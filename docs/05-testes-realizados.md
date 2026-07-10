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

## ⚠️ 4. Importação no NNgen — BACKBONE OK, CAUDA PRECISA DE AJUSTE

Este foi o teste mais importante, porque valida o risco que eu havia sinalizado
(mapear **Conv1D → conv2d k×1** e o **`Add`** do skip no NNgen). Resultado real:

- **O NNgen importa toda a espinha convolucional sem erro.** O importador
  percorreu, sem falhar, todas as **Conv1D** (mapeadas para conv2d k×1), as
  somas residuais **`Add`**, as **MaxPool**, **BatchNormalization**, **ReLU**,
  **Transpose** e **Flatten/Reshape**. Confirmado inspecionando o `func_map` do
  importador (ops suportadas: `Conv, Gemm, Add, MaxPool, Relu, Sigmoid, Flatten,
  BatchNormalization, Mul, Reshape, Transpose`).
- **O import só falha na Dense(6) final** — que é **0,002% dos MACs** do modelo.
  Causa: o `tf2onnx` serializa a Dense como `Transpose + Reshape + MatMul` (com o
  peso passando por um `Cast`), enquanto o front-end do NNgen espera um `Flatten`
  limpo + `Gemm` com peso **constante**. Erros observados, em sequência:
  `KeyError: 'MatMul'` → (após trocar por `Gemm`) `object of type 'variable' has
  no len()` no importador de `Gemm`/`Reshape`.

**Como resolver (documentado para a Etapa 2):**
1. **Recomendado — separar a cauda:** deixar o NNgen acelerar o *backbone*
   convolucional (que é ~100% da computação) e rodar a **Dense(6) final no host**.
   É trivial (5120×6 = 30 k MACs) e elimina toda a fricção de serialização.
2. **Alternativa — higiene de grafo:** re-exportar o modelo garantindo um
   `Flatten` op nativo + `Gemm` com peso como *initializer* constante (sem
   `Cast`/`Transpose`/`Reshape`), ou substituir a Dense por uma **Conv1D 1×1**
   (que o NNgen mapeia como `Conv`).

**Conclusão do teste:** a viabilidade do NNgen para este modelo está
**praticamente confirmada** — o difícil (todo o backbone Conv1D + skips) passou;
o que resta é higiene da camada final, com dois caminhos claros de solução.

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
| Geração de RTL completa no NNgen | depende de resolver a cauda (item 4) | Etapa 2 |
| Síntese Bambu → Verilog | PandA/Bambu não instalado no contêiner | Etapa 2 |
| *Place & route* / timing no Quartus | precisa do Quartus Prime + parte exata da Arria V | Etapa 3 |
| Execução na placa | precisa do hardware Arria V + DDR3 | Etapa 3 |
