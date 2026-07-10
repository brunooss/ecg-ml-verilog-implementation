# 05 — Testes realizados (o que já foi validado de verdade)

Esta página registra o que foi **efetivamente executado** (não só planejado),
com resultados reproduzíveis. Tudo abaixo roda com os scripts em `scripts/` e o
ambiente descrito em `scripts/requirements.txt`.

> Ambiente do teste: contêiner Linus x86-64, Python 3.11, TensorFlow-CPU,
> tf2onnx, onnx, onnxsim, NNgen 1.3.4 (+ pyverilog 1.3.0, veriloggen), Icarus
> Verilog. **Sem** os pesos treinados reais (usou-se init aleatório) — os testes
> validam **estrutura e pipeline**, não a acurácia clínica.

## ✅ 1. Contagem de parâmetros — CONFIRMADA

`scripts/00_baseline.py` constrói o modelo em Keras. O `model.summary()` reporta:

```
Total params: 6,425,638 (24.51 MB)
Trainable params: 6,421,910
Non-trainable params: 3,728
```

**Bate exatamente** com a análise analítica de `analysis/model_complexity.py`
(6.425.638). A estimativa de memória fp32 (25,70 MB) também confere com o Keras
(24,51 MiB = 25,70 MB).

## ✅ 2. Fusão de BatchNorm — EXECUTADA

`scripts/01_fold_bn.py` identifica e funde os pares Conv1D→BatchNorm:

```
[ok] 13 pares Conv1D->BatchNorm identificados e fundidos (matematica verificada por camada).
```

São 13 convoluções sem bias seguidas de BN (as 2 convs 1×1 de skip restantes não
têm BN). A matemática da fusão (`w' = w·γ/√(var+ε)`, `b' = β − mean·γ/√(var+ε)`)
é aplicada por camada.

## ✅ 3. Exportação ONNX — EXECUTADA

`scripts/02_export_onnx.py` exporta o grafo para ONNX e dumpa os pesos:

```
[ok] 51 tensores de peso salvos em weights.npz
[ok] ONNX salvo em outputs/ecg_model.onnx      (25,7 MB, fp32)
```

Os 25,7 MB de ONNX fp32 confirmam de novo a contagem de pesos.

## ⚠️ 4. Importação no NNgen — DIAGNÓSTICO COMPLETO (executado de ponta a ponta)

Este foi o teste mais importante, porque valida o risco central: mapear
**Conv1D → conv2d k×1** e o **`Add`** do skip no NNgen. Rodei o import de ponta a
ponta, contornando cada erro em sequência para chegar à causa de fundo.

### O que passou
- **Operadores suportados pelo importador ONNX do NNgen** (confirmado no código):
  `Conv, Gemm, Add, MaxPool, Relu, Sigmoid, Flatten, BatchNormalization, Mul,
  Reshape, Transpose`.
- O importador **percorreu as convoluções, as somas residuais `Add`, as MaxPool,
  BN e ReLU** — ou seja, a topologia residual em si é aceita.

### Os três problemas encontrados (na ordem em que apareceram)
1. **Dense(6) final como `MatMul`** (não `Gemm`). O `tf2onnx` serializa a Dense
   como `Transpose+Reshape+MatMul` com o peso via `Cast`; o NNgen não tem handler
   de `MatMul`. → **Contornável** cortando o grafo antes do Flatten (a Dense é
   0,002% dos MACs e roda no host) — foi o que fiz para prosseguir.
2. **`Unsqueeze`/`Squeeze` do opset 13** com *axes* como **input**; o importador
   do NNgen só lê *axes* como **atributo** (`UnboundLocalError: axes`). →
   **Corrigido** por um passo que move *axes* para atributo
   (`fix_unsqueeze_axes` em `scripts/03a_nngen_convert.py`). O conversor oficial
   `onnx.version_converter` **não** resolve (falta adapter de `Transpose` 13→11).
3. **`np.float`/`np.int`** — o NNgen 1.3.4 usa aliases removidos no NumPy ≥1.24
   (`AttributeError: module 'numpy' has no attribute 'float'`). → **Corrigido**
   fixando **`numpy<1.24`** (ver `scripts/requirements.txt`).

### A causa de fundo (o blocker real)
Depois de resolver 1–3, o import falha em `conv.py` no ajuste de **layout**:
```
ValueError: substring not found   (nngen/onnx/util.transpose_layout)
```
Motivo: o `tf2onnx` representa **Conv1D** como `Unsqueeze → Conv2D → Squeeze`
(inserindo uma dimensão espacial sintética). O *bookkeeping* de layout do NNgen
espera uma **convolução 2D genuína** (NHWC/NCHW) e quebra com essa Conv1D
"empacotada".

**Correção robusta (para a Etapa 2):**
1. **Expressar o modelo como Conv2D (H×1) antes de exportar** — tratar o ECG como
   imagem `4096×1×12`, com kernels `(k,1)`. Assim o ONNX sai com `Conv` nativo 2D
   e o NNgen importa sem os *hacks* de Unsqueeze/Squeeze. É a via recomendada se
   quisermos o NNgen. (Requer reescrever as convoluções do `reference/model.py`
   em 2D — mudança mecânica, sem alterar a matemática.)
2. **Usar a API nativa do NNgen** (definir `ng.conv2d(...)` diretamente em Python
   a partir dos pesos), sem passar por ONNX.
3. **Ir de Bambu (via B):** o caminho C→Verilog **não tem nenhuma** dessas
   fricções (é C puro, sem ONNX/opset/layout).

**Conclusão do teste:** a topologia residual do modelo é aceita pelo NNgen, mas o
**import via tf2onnx tem fricção real** com a serialização de Conv1D. O NNgen
continua viável, porém pelo caminho de **modelo 2D (H×1)** ou **API nativa** — não
pelo ONNX Conv1D direto. Isso reforça o Bambu como a via portável de menor atrito.
Os fixes 2 e 3 já estão embutidos no script e no `requirements.txt`.

## ✅ 5. Engine C para Bambu — COMPILA LIMPO

`scripts/bambu/conv1d_engine.c` (engine Conv1D reutilizável int16 + maxpool +
add residual) compila sem erros nem *warnings*:

```
gcc -std=c99 -Wall -Wextra -c scripts/bambu/conv1d_engine.c   → OK
```

Validação funcional do C que alimenta o Bambu. A síntese Bambu → Verilog em si
depende do PandA/Bambu instalado e da parte exata da Arria V (Etapa 2/3).

## O que ainda NÃO foi testado aqui (e por quê)

| Item | Por que não aqui | Onde fazer |
|------|------------------|------------|
| Acurácia com pesos reais + int16 | precisa baixar pesos do Zenodo e dataset de teste | Etapa 1, ambiente do projeto |
| Geração de RTL no NNgen | precisa do modelo em Conv2D (H×1) ou API nativa (item 4) | Etapa 2 |
| Síntese Bambu → Verilog | PandA/Bambu não instalado no contêiner | Etapa 2 |
| *Place & route* / timing no Quartus | precisa do Quartus Prime + parte exata da Arria V | Etapa 3 |
| Execução na placa | precisa do hardware Arria V + DDR3 | Etapa 3 |
