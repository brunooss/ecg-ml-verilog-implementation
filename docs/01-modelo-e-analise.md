# 01 — O modelo e sua análise de complexidade

## 1.1 O que o modelo faz

O modelo de Ribeiro et al. classifica **ECG de 12 derivações** em 6
anormalidades cardíacas (bloqueio AV de 1º grau, bloqueio de ramo direito,
bloqueio de ramo esquerdo, bradicardia sinusal, fibrilação atrial e taquicardia
sinusal).

| Item | Valor |
|------|-------|
| Entrada | tensor `(N, 4096, 12)` — ~10 s de ECG a 400 Hz, 12 derivações, escala 1e-4 V |
| Saída | tensor `(N, 6)` — probabilidade (sigmoide) por anormalidade |
| Framework | TensorFlow 2.2 / Keras (há um branch legado em TF 1.15) |
| Pesos treinados | disponíveis no [Zenodo (doi:10.5281/zenodo.3625017)](https://doi.org/10.5281/zenodo.3625017), formato `.hdf5` |

A definição está em `reference/model.py` (cópia fiel do `model.py` original).

## 1.2 Arquitetura

É uma **ResNet unidimensional** (1D-CNN residual). Poucas camadas de alto nível,
mas cada uma bastante "pesada":

```
Entrada (4096, 12)
  └─ Conv1D(64, k=16, stride=1)  → BN → ReLU              # bloco de entrada (stem)
  └─ ResidualUnit → (1024, 128)                            # bloco residual 1
  └─ ResidualUnit → (256, 196)                             # bloco residual 2
  └─ ResidualUnit → (64, 256)                              # bloco residual 3
  └─ ResidualUnit → (16, 320)                              # bloco residual 4
  └─ Flatten (5120) → Dense(6, sigmoide)                   # cabeça de classificação
```

Cada **ResidualUnit** contém:
- um ramo principal com **duas Conv1D** de kernel 16 (a 2ª com *stride* = fator de
  downsampling, tipicamente 4), cada uma seguida de BatchNorm + ReLU + Dropout;
- um **skip connection** com MaxPooling (downsampling) e, quando o nº de canais
  muda, uma **Conv1D 1×1** para casar a dimensão de canais;
- uma soma (`Add`) do ramo principal com o skip.

Detalhes relevantes para hardware:
- **Sem bias nas convoluções** (`use_bias=False`) — o bias é absorvido pela
  BatchNorm seguinte.
- **BatchNorm** em inferência é uma transformação afim por canal
  (`y = gamma*(x-mean)/sqrt(var+eps) + beta`) que pode ser **fundida (folded)**
  na convolução anterior, virando `w' = w * scale` e um bias por canal. Isso
  **elimina a BN do datapath** em hardware — passo padrão de otimização.
- **Dropout** é *no-op* em inferência (removido).
- Ativação **sigmoide** só na saída (6 valores) — barata; pode ser LUT ou
  até dispensada se só precisarmos do *score* pré-sigmoide (a sigmoide é
  monotônica, então limiares podem ser aplicados no pré-ativação).

## 1.3 Contagem de parâmetros e operações

Gerado por `analysis/model_complexity.py` (derivação analítica, sem TensorFlow):

```
layer            out_samples  out_ch        params            MACs
------------------------------------------------------------------
stem_conv               4096      64        12,288      50,331,648
stem_bn                 4096      64           256               0
res1_skip_1x1           1024     128         8,192       8,388,608
res1_conv1              4096     128       131,072     536,870,912
res1_conv2              1024     128       262,144     268,435,456
res2_skip_1x1            256     196        25,088       6,422,528
res2_conv1              1024     196       401,408     411,041,792
res2_conv2               256     196       614,656     157,351,936
res3_skip_1x1             64     256        50,176       3,211,264
res3_conv1               256     256       802,816     205,520,896
res3_conv2                64     256     1,048,576      67,108,864
res4_skip_1x1             16     320        81,920       1,310,720
res4_conv1                64     320     1,310,720      83,886,080
res4_conv2                16     320     1,638,400      26,214,400
dense                      1       6        30,726          30,720
------------------------------------------------------------------
TOTAL                                    6,425,638   1,826,125,824
```

(BatchNorms omitidas acima por brevidade; somam ~7,5 k parâmetros e 0 MAC úteis
após a fusão.)

| Métrica | Valor |
|---------|-------|
| **Parâmetros totais** | **6.425.638 (~6,43 M)** |
| **MACs por inferência** | **1.826.125.824 (~1,83 GMAC)** → ~3,65 GFLOP |
| Memória de pesos @ int8 | **~6,43 MB** |
| Memória de pesos @ int16 | ~12,85 MB |
| Memória de pesos @ fp32 | ~25,70 MB |

Esse valor de ~6,4 M de parâmetros bate com o número reportado para este modelo
na literatura, o que valida a contagem.

## 1.4 Implicações diretas para a FPGA

Este é o ponto mais importante da análise, e ele restringe fortemente as opções
de porte:

1. **Os pesos NÃO cabem na memória interna.** Uma Arria V GX A7 tem ~13.660 Kb
   (~1,67 MB) de M10K; a maior (B7) tem ~2,5 MB. Os pesos exigem 6,4 MB (int8)
   a 25,7 MB (fp32). **Portanto os pesos precisam morar em DRAM externa (DDR3)**
   e ser trazidos por streaming/tiling. Ver [`04-arria-v-recursos.md`](04-arria-v-recursos.md).
   → Isso **inviabiliza** o fluxo "modelo totalmente desenrolado no chip" que
   ferramentas como o hls4ml usam para modelos minúsculos (kB de pesos). Aqui é
   obrigatório um acelerador com **reuso de PEs + DMA de pesos**.

2. **1,83 GMAC é muito, mas a latência-alvo é folgada.** Uma inferência a cada
   ~10 s (tempo de uma janela de ECG) é trivial em termos de vazão. Mesmo a
   ~100 MHz com um único MAC seríamos lentos, mas com algumas dezenas/centenas
   de DSPs em paralelo a inferência sai em poucos ms — **não há pressão de
   throughput**, o que permite uma arquitetura pequena e serializada, reusando
   um núcleo de convolução para todas as camadas.

3. **Quantização é praticamente obrigatória — precisão escolhida: int16.** fp32 é
   caro em DSP e memória. Com *post-training quantization* em **int16** os pesos
   ficam em ~12,85 MB (na DDR3) preservando a acurácia com boa margem; int8
   (~6,4 MB) fica como otimização opcional futura. A escolha do int16 é
   justificada em [`03-recomendacao-e-pipeline.md`](03-recomendacao-e-pipeline.md#justificativa-da-precisão-int16).
   Ainda assim, é preciso validar a **perda de acurácia (AUC/F1)** após a
   quantização com os dados de teste do repositório original.

4. **A topologia é amigável a um acelerador genérico de CNN 1D.** Só há 5 tipos
   de operação: Conv1D (incl. 1×1), MaxPool1D, Add, ReLU e uma Dense final. Um
   único *engine* de Conv1D parametrizável (com caminhos para pooling/add) cobre
   o modelo inteiro iterando camada a camada — arquitetura clássica de
   *layer-by-layer accelerator*.

## 1.5 O número de camadas é um problema? (resposta à sua dúvida)

Não. A preocupação de que "o porte fica exponencialmente mais difícil com mais
camadas" se aplica ao **porte manual** de RTL escrito à mão, onde cada camada
vira lógica dedicada. Mas:

- este modelo tem **poucas camadas de alto nível** (1 conv + 4 blocos + 1 densa,
  ~15 convoluções no total);
- a dificuldade real não é o número de camadas e sim o **volume de dados/pesos**
  (6,4 M de parâmetros), que exige DRAM e controle de dataflow.

A solução idiomática — tanto no porte manual quanto no HLS — é **não** instanciar
uma camada por bloco de hardware, e sim ter **um engine reutilizável** que
processa cada camada em sequência lendo pesos da DRAM. Com essa abordagem, o
custo de hardware é **independente do número de camadas** — o que muda é só a
tabela de configuração (dimensões por camada) e o tempo de execução. Isso torna
o porte perfeitamente viável.
