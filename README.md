# Porte do modelo de diagnóstico automático de ECG para FPGA (Verilog)

Pesquisa e engenharia para portar o modelo de diagnóstico automático de ECG
de 12 derivações de [Ribeiro et al. (Nature Communications, 2020)](https://www.nature.com/articles/s41467-020-15432-4)
— repositório [`antonior92/automatic-ecg-diagnosis`](https://github.com/antonior92/automatic-ecg-diagnosis),
implementado em TensorFlow/Keras — para **Verilog RTL** executável em uma FPGA
**Intel/Altera Arria V**.

O objetivo é rodar **o modelo em hardware** (o *forward pass* com os pesos e
bias já treinados e congelados), de forma o mais **independente de fornecedor**
possível, para que a mesma metodologia possa ser reaproveitada em outras placas.

> **Estado atual:** fase de pesquisa e decisão de arquitetura. Este repositório
> reúne a análise do modelo, o levantamento das alternativas de porte e uma
> recomendação de pipeline. Ainda **não** há RTL gerado — os próximos passos
> dependem de decisões descritas em [`docs/03-recomendacao-e-pipeline.md`](docs/03-recomendacao-e-pipeline.md).

## TL;DR das conclusões

1. **O modelo é grande para FPGA.** ~**6,43 milhões de parâmetros** e
   ~**1,83 GMAC por inferência** (uma janela de ECG de ~10 s). Só os pesos, em
   int8, ocupam ~6,4 MB — **muito acima** da memória interna de uma Arria V GX
   (~1,7 MB de blocos M10K no A7). **Conclusão:** os pesos têm de ficar em
   **DRAM externa (DDR3)** e ser carregados por streaming; uma arquitetura
   "tudo desenrolado no chip" (como a usada em modelos minúsculos) é inviável
   aqui. Ver [`docs/01-modelo-e-analise.md`](docs/01-modelo-e-analise.md).

2. **OpenVINO está descartado** para a Arria V. O plugin FPGA do OpenVINO só
   existiu para Arria 10 (PAC/Mustang) e foi *descontinuado* já em 2020; nunca
   suportou a Arria V. Ver [`docs/02-alternativas-de-porte.md`](docs/02-alternativas-de-porte.md).

3. **OpenCL / oneAPI também não servem** como você suspeitava. O Intel FPGA SDK
   for OpenCL foi descontinuado e não tem BSP para Arria V; o oneAPI só suporta
   Arria 10 / Stratix 10 / Agilex e teve o suporte a FPGA *deprecado* a partir
   de 2025.1.

4. **Porte manual "à mão"** de um modelo de 6,4 M de parâmetros é inviável — não
   por causa do *número de camadas* (são poucas: 1 conv inicial + 4 blocos
   residuais + 1 densa), mas pela quantidade de aritmética, memória e controle
   de dataflow. Só valeria como referência de blocos isolados.

5. **O caminho recomendado é HLS** (modelo → C/C++/RTL → Verilog), em **duas vias
   paralelas, ambas gerais e sem back-end de fornecedor** (decisão da rodada 1):
   - **NNgen** (ONNX → Verilog + core AXI): gera RTL direto, já com DMA para DRAM
     e quantização inteira — melhor ajuste à necessidade de memória externa;
   - **PandA/Bambu** (C/C++ → Verilog): HLS open-source e agnóstico de fornecedor.

   **hls4ml** fica como comparação opcional: ele **não emite RTL** e depende de um
   back-end de HLS de fornecedor (Intel HLS/oneAPI), contrariando o critério de
   ser geral. A comparação completa está em
   [`docs/02-alternativas-de-porte.md`](docs/02-alternativas-de-porte.md) e
   [`docs/03-recomendacao-e-pipeline.md`](docs/03-recomendacao-e-pipeline.md).

## Estrutura do repositório

```
.
├── README.md                        # este arquivo
├── docs/
│   ├── 01-modelo-e-analise.md       # arquitetura, contagem de params/MACs, viabilidade
│   ├── 02-alternativas-de-porte.md  # OpenVINO, OpenCL/oneAPI, manual, HLS (hls4ml/Bambu/NNgen)
│   ├── 03-recomendacao-e-pipeline.md# pipeline recomendado, plano por etapas, decisões abertas
│   ├── 04-arria-v-recursos.md       # recursos da Arria V e orçamento de hardware
│   └── referencias.md               # todas as fontes consultadas
├── analysis/
│   ├── model_complexity.py          # calcula params/MACs por camada (não precisa de TensorFlow)
│   └── model_complexity.txt         # saída congelada do script
└── reference/
    └── model.py                     # cópia do model.py original (Ribeiro et al.) p/ referência
```

## Como reproduzir a análise de complexidade

```bash
python3 analysis/model_complexity.py
```

Não requer TensorFlow — as formas das camadas são derivadas analiticamente a
partir da definição em `reference/model.py`.

## Decisões que dependem de você

Estão consolidadas no fim de
[`docs/03-recomendacao-e-pipeline.md`](docs/03-recomendacao-e-pipeline.md).
**Rodada 1 já decidida:** modelo **completo** (6 saídas, 4096×12); precisão
**int16**; duas vias **gerais** de porte (**NNgen + Bambu**); DDR3 externa (a
Arria V do projeto tem de sobra — estimativa em §4.4). Restam abertas, sem
bloquear o início: a **parte/kit exato** da Arria V e a **interface de I/O**.
