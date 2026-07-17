# Porte do modelo de diagnóstico automático de ECG para FPGA (Verilog)

Pesquisa e engenharia para portar o modelo de diagnóstico automático de ECG de
12 derivações de [Ribeiro et al. (Nature Communications, 2020)](https://www.nature.com/articles/s41467-020-15432-4)
— repositório [`antonior92/automatic-ecg-diagnosis`](https://github.com/antonior92/automatic-ecg-diagnosis),
em TensorFlow/Keras — para **Verilog RTL** executável em uma FPGA **Intel/Altera
Arria V**.

O objetivo é rodar **o modelo em hardware** (o *forward pass* com pesos e bias já
treinados e congelados), de forma o mais **independente de fornecedor** possível,
para reaproveitar a metodologia em outras placas.

> 🟢 **Novo por aqui / sem background técnico?** Comece por
> [`docs/06-explicacao-para-leigos.md`](docs/06-explicacao-para-leigos.md) — explica
> o projeto inteiro (modelo, camadas, ferramentas e o que já foi feito) em
> linguagem simples.

---

## 1. Conclusões

### 1.1 O modelo
- **~6,43 M de parâmetros** e **~1,83 GMAC por inferência** (janela de ~10 s de
  ECG). *Confirmado* rodando o Keras (`Total params: 6,425,638`) e pela análise
  analítica em `analysis/`.
- Topologia enxuta: 1 convolução de entrada + **4 blocos residuais** + 1 camada
  densa. Só 5 tipos de operação: Conv1D (inclui 1×1), MaxPool1D, Add, ReLU e uma
  Dense final com sigmoide.

### 1.2 A restrição que define o projeto
- Os **pesos não cabem na memória interna** da Arria V (~1,7–2,1 MB de M10K)
  contra 6,4 MB (int8) a 25,7 MB (fp32). → **DDR3 externa é obrigatória**, com um
  acelerador **layer-by-layer** que reusa um único engine de Conv1D e faz
  streaming dos pesos. A placa do projeto tem DDR3 de sobra: são necessários só
  ~**16–20 MB** e a banda exigida (~1,3–90 MB/s) fica muito abaixo do disponível
  (~1,6–4 GB/s). Ver [`docs/04-arria-v-recursos.md`](docs/04-arria-v-recursos.md).
- **Latência folgada:** uma inferência a cada ~10 s é trivial → dá para usar
  poucos DSPs e um projeto serializado. O gargalo é memória/banda, não cálculo.

### 1.3 Precisão: **int16**
Escolhida por ser a opção segura "de graça": como pesos, banda e DSP estão todos
folgados, o custo do int16 sobre o int8 é irrelevante, enquanto o int16 preserva
a acurácia (AUC/F1) com muito mais margem — o que importa num modelo clínico.
int8 fica como otimização opcional futura. Justificativa em
[`docs/03-recomendacao-e-pipeline.md`](docs/03-recomendacao-e-pipeline.md).

### 1.4 Alternativas de porte — o que serve e o que não serve
| Via | Serve para Arria V? | Motivo |
|-----|:---:|--------|
| **OpenVINO** | ❌ | plugin FPGA só existiu p/ Arria 10 e foi descontinuado (2020); entrega bitstream fechado, não Verilog |
| **OpenCL (Intel FPGA SDK)** | ❌ | descontinuado; sem BSP para Arria V |
| **oneAPI (DPC++ FPGA)** | ❌ | só Arria 10/Stratix 10/Agilex; suporte a FPGA deprecado em 2025.1 |
| **Porte manual RTL** | ⚠️ | tecnicamente possível, mas escrever 6,4 M de params à mão é inviável — só p/ blocos-referência |
| **hls4ml** | ⚠️ | funciona, mas **não emite RTL**: depende de back-end de HLS de fornecedor (Intel HLS/oneAPI) → contraria "ser geral" |
| **NNgen** (ONNX→Verilog) | ✅ | emite RTL direto (sem HLS de fornecedor), já com DMA/DDR3 e quantização inteira |
| **PandA/Bambu** (C→Verilog) | ✅ | HLS open-source e agnóstico de fornecedor |

Detalhes em [`docs/02-alternativas-de-porte.md`](docs/02-alternativas-de-porte.md).

### 1.5 Caminho escolhido
**Duas vias HLS paralelas, ambas gerais e sem back-end de fornecedor:**
**NNgen** (ONNX → Verilog) e **PandA/Bambu** (C → Verilog). hls4ml fica como
comparação opcional. Em ambas, dois pré-processamentos comuns: **fundir a
BatchNorm** nas convoluções e **quantizar para int16**.

### 1.6 O que já foi testado de verdade (neste ambiente)
Executado com TensorFlow + NNgen (resultados completos em
[`docs/05-testes-realizados.md`](docs/05-testes-realizados.md)):
- ✅ Contagem de parâmetros confirmada (6.425.638, bate com a análise).
- ✅ Fusão de BatchNorm: 13 pares Conv1D→BN fundidos.
- ✅ Exportação para ONNX (25,7 MB fp32).
- ✅ **NNgen gera Verilog RTL de verdade.** Pela API nativa, gerei o RTL de uma
  camada de convolução do modelo (64→128, kernel 16×1, int16): **61.572 linhas**,
  **parse OK no Icarus Verilog**, com **interface AXI4 para a DDR3**. Excerto em
  [`rtl/`](rtl/). É o bloco reutilizável do acelerador.
- ⚠️ O import do modelo **inteiro** via ONNX (tf2onnx) tem fricção com o NNgen
  1.3.4: reescrevi o modelo em **Conv2D (H×1)** e o problema de layout sumiu, mas
  o `padding='SAME'` de convoluções com *stride* ainda diverge do Keras no `Add`
  residual → por isso a **API nativa** é o caminho. Diagnóstico completo em
  [`docs/05`](docs/05-testes-realizados.md) §4–6.
- ✅ Engine C do Bambu compila limpo (`gcc -Wall -Wextra`) — via **sem** nenhuma
  dessas fricções de ONNX.
- ✅ **Simulação funcional com RAM (DDR3 simulada): PASS.** O acelerador gerado
  pelo NNgen foi simulado de ponta a ponta no Icarus Verilog: um modelo de RAM
  AXI4 ([`rtl/sim/axi_ram_model.v`](rtl/sim/axi_ram_model.v)) faz o papel da
  DDR3 (sinal + pesos pré-carregados via `$readmemh`), o testbench
  ([`rtl/sim/tb_ecg_conv_layer.v`](rtl/sim/tb_ecg_conv_layer.v)) programa o
  acelerador via AXI4-Lite e a **saída bate palavra a palavra (match exato)**
  com o golden calculado em NumPy (`ng.eval`). Ver §2.3 e
  [`docs/05`](docs/05-testes-realizados.md) §7.

---

## 2. Como testar / reproduzir

### 2.1 Só a análise de complexidade (sem dependências pesadas)
```bash
python3 analysis/model_complexity.py     # imprime params/MACs por camada
```

### 2.2 Frente de software completa (build → fold BN → ONNX)
Requer o ambiente de `scripts/requirements.txt`. Recomenda-se um venv (pyverilog
exige `setuptools<58` no Python 3.11):
```bash
python3 -m venv env && . env/bin/activate
pip install "setuptools<58" wheel
pip install -r scripts/requirements.txt
sudo apt-get install -y iverilog        # p/ simular o RTL depois

# 0) baseline + golden reference (com pesos reais do Zenodo, ou sem p/ smoke test)
python3 scripts/00_baseline.py --weights model.hdf5

# 1) fusao de BatchNorm nas convolucoes
python3 scripts/01_fold_bn.py --weights model.hdf5

# 2) exporta ONNX (entrada do NNgen) + dump de pesos
python3 scripts/02_export_onnx.py --weights model.hdf5
```
> Os pesos treinados (`model.hdf5`) vêm do
> [Zenodo doi:10.5281/zenodo.3625017](https://doi.org/10.5281/zenodo.3625017).
> Sem `--weights`, os scripts rodam com init aleatório (validam o pipeline, não a
> acurácia).

### 2.3 Simulação do acelerador com RAM AXI4 (DDR3 simulada)
Não precisa do TensorFlow — só `nngen`, `numpy<1.24` e `iverilog`:
```bash
pip install "setuptools<58" && pip install "numpy<1.24" nngen==1.3.4
sudo apt-get install -y iverilog

cd sim
make check                                  # gera dados+RTL, simula e verifica
make check DIMS="--H 128 --Cin 16 --Cout 16"   # instância maior
make wave                                   # idem, com dump VCD
```
O fluxo: `scripts/04_make_sim_data.py` monta o grafo (mesmo operador do
`03b`), gera o RTL da instância de teste, a **imagem da RAM** (`ram_init.hex`,
com sinal + pesos + bias nos endereços que o próprio NNgen aloca) e o **golden**
(`ng.eval`). O testbench emula o host: habilita interrupção, escreve no
registrador `START` via AXI4-Lite, espera o `irq`, e compara a região de saída
da RAM com o golden — o critério é **match exato** (int16, sem tolerância).
Por padrão usa dimensões reduzidas (H=64, 8→8 canais, kernel 16×1) para a
simulação terminar em segundos; a instância com as dimensões reais da camada
(4096, 64→128) é a mesma geração com `DIMS`, mas leva horas no iverilog.

---

## 3. Como fazer a conversão (modelo → Verilog)

### 3.1 Via A — NNgen (ONNX → Verilog RTL + AXI/DMA)
```bash
python3 scripts/03a_nngen_convert.py --onnx outputs/ecg_model.onnx --bitwidth 16
```
Gera o Verilog do acelerador (PEs + memória on-chip + DMA + AXI4), que sintetiza
no Quartus para a Arria V. **Sem** HLS de fornecedor.

**Ajustes conhecidos (diagnosticados executando):** o script já embute os fixes
de opset (`Unsqueeze` axes→atributo) e o `requirements.txt` fixa `numpy<1.24`. A
solução de fundo para o NNgen consumir as convoluções é **exportar o modelo como
Conv2D (H×1)** (ECG como imagem 4096×1×12) ou usar a **API nativa do NNgen** — o
ONNX Conv1D do tf2onnx não casa com o layout esperado. Diagnóstico completo em
[`docs/05-testes-realizados.md`](docs/05-testes-realizados.md) §4. Sem isso, a via
**Bambu** (§3.2) é a de menor atrito.

### 3.2 Via B — PandA/Bambu (C → Verilog)
```bash
bambu scripts/bambu/conv1d_engine.c --top-fname=conv1d_layer \
      --device-name=<parte_arria_v> --clock-period=10 \
      --generate-tb=tb.xml --simulate -v3
```
`conv1d_engine.c` é o engine de Conv1D reutilizável (int16, BN fundida) que cobre
todas as camadas iterando camada a camada. Detalhes em
[`scripts/bambu/README.md`](scripts/bambu/README.md).

### 3.3 Validação
Ambas as vias geram *testbench*; simule com Icarus Verilog e compare a saída
contra `outputs/golden_output.npy` (a referência fp32 do baseline). O critério é
casar dentro da tolerância de quantização int16.

---

## 4. Como implementar na Arria V

1. **Pré-processar o modelo** (§2 e §3): fold BN + quantização int16 + geração do
   RTL (NNgen ou Bambu) + blob de pesos para a DDR3.
2. **Montar o projeto no Quartus Prime (Standard)** para a **parte exata** da sua
   Arria V. Integrar:
   - o acelerador (RTL gerado);
   - o **controlador DDR3** (hard memory controller da Arria V) para os pesos;
   - o caminho de I/O do ECG (host via AXI/PCIe, UART, ou memória pré-carregada);
   - o *wrapper* de topo que sequencia as camadas.
3. **Fechar timing** a ~100 MHz (folgado, dada a latência-alvo) e fazer
   *place & route*.
4. **Validar em hardware** contra o golden reference; medir latência e uso de
   recursos (LUT/DSP/M10K/banda DDR3).
5. **Otimizar** o paralelismo do engine (nº de DSPs) e o *tiling* ao orçamento.

Plano por etapas com marcos verificáveis em
[`docs/03-recomendacao-e-pipeline.md`](docs/03-recomendacao-e-pipeline.md).

> **Posso rodar isso por você?** A **frente de software** (§2 e a maior parte do
> §3 — build, fold, ONNX, import no NNgen, compilação do C) eu executo numa
> sessão como esta, e já executei (ver [`docs/05`](docs/05-testes-realizados.md)).
> O que **não** dá para fazer aqui é a **síntese no Quartus e a execução na
> placa** (§4 passos 2–4): exigem o toolchain Intel licenciado e o **hardware
> Arria V físico**, que este ambiente não tem. Para essas etapas eu gero e valido
> os artefatos (RTL, testbench, scripts de projeto); você roda no Quartus/placa.

---

## 5. Estrutura do repositório

```
.
├── README.md
├── analysis/
│   ├── model_complexity.py      # params/MACs por camada (sem TensorFlow)
│   └── model_complexity.txt
├── docs/
│   ├── 01-modelo-e-analise.md   # arquitetura, complexidade, viabilidade
│   ├── 02-alternativas-de-porte.md
│   ├── 03-recomendacao-e-pipeline.md
│   ├── 04-arria-v-recursos.md   # recursos da Arria V + estimativa de DDR3
│   ├── 05-testes-realizados.md  # o que já foi executado e validado
│   ├── 06-explicacao-para-leigos.md  # o projeto inteiro em linguagem simples
│   └── referencias.md
├── rtl/
│   ├── ecg_conv_layer_interface.v    # excerto do Verilog gerado (NNgen)
│   └── sim/
│       ├── axi_ram_model.v      # RAM AXI4 (DDR3 simulada, $readmemh)
│       └── tb_ecg_conv_layer.v  # testbench: host AXI-Lite + verificacao
├── sim/
│   └── Makefile                 # make check -> gera, simula e verifica
├── scripts/
│   ├── requirements.txt
│   ├── 00_baseline.py           # build + golden reference
│   ├── 01_fold_bn.py            # fusao de BatchNorm
│   ├── 02_export_onnx.py        # export ONNX + dump de pesos
│   ├── 03a_nngen_convert.py     # Via A: ONNX -> Verilog (NNgen)
│   ├── 03b_nngen_native_layer.py # Via A: RTL de 1 camada (API nativa)
│   ├── 04_make_sim_data.py      # imagem da RAM + golden p/ simulacao
│   └── bambu/                   # Via B: C -> Verilog (Bambu)
│       ├── conv1d_engine.c
│       └── README.md
└── reference/
    └── model.py                 # copia do model.py original (Ribeiro et al.)
```

## 6. Próximos passos e pontos abertos

**Próximos passos técnicos (executáveis em software, sem hardware):**
1. **NNgen:** encadear as ~15 convoluções pela **API nativa** (já provada gerando
   RTL de uma camada — ver [`docs/05`](docs/05-testes-realizados.md) §6 e
   [`rtl/`](rtl/)), somar os *skips* e emitir o Verilog do backbone completo.
   A infraestrutura de simulação (RAM AXI4 + testbench + golden, §2.3) já está
   pronta e **passando com match exato** para uma camada — para o backbone é
   reusar o mesmo harness com a nova imagem de memória.
2. **Pesos reais na simulação:** trocar os valores aleatórios do
   `04_make_sim_data.py` pelos pesos int16 do `weights.npz` (fold BN feito) e
   uma janela real de ECG — mesma mecânica, valida a quantização de verdade.
3. **Acurácia:** baixar os pesos treinados (Zenodo) + o CODE-test e medir a
   **queda de AUC/F1 com int16** — a validação mais importante para a pesquisa.
4. **Bambu:** empacotar `weights.npz` em int16 + tabela de camadas e sintetizar
   o `conv1d_engine.c` (precisa do PandA/Bambu instalado).

**Pontos abertos (dependem de você, não bloqueiam o software):**
- **Parte/kit exato da Arria V** e quanto de DDR3 o board tem — para *place &
  route* e timing.
- **Interface de I/O** do ECG (host/AXI/UART/memória) — define o *wrapper* de topo.
