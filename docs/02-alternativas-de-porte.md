# 02 — Alternativas de porte para a Arria V

Levantamento das vias para levar o modelo Keras até Verilog rodando numa
Arria V, com verificação atualizada do que **de fato** suporta essa placa. As
fontes estão em [`referencias.md`](referencias.md).

## Resumo executivo (tabela de decisão)

| Via | Suporta Arria V? | Gera Verilog? | Independente de fornecedor? | Adequada a modelo de 6,4 M params? | Veredito |
|-----|:---:|:---:|:---:|:---:|:---|
| **OpenVINO (plugin FPGA)** | ❌ nunca (só Arria 10) | ❌ (bitstream fechado) | ❌ | — | **Descartado** |
| **Intel FPGA SDK for OpenCL** | ❌ sem BSP p/ Arria V | via Quartus, opaco | ❌ | parcial | **Descartado** |
| **oneAPI (DPC++ FPGA)** | ❌ (só A10/S10/Agilex) | via Quartus, opaco | ❌ | sim, mas… | **Descartado** p/ Arria V |
| **Porte manual RTL** | ✅ (é só Verilog) | ✅ | ✅ | inviável de escrever à mão | **Só p/ blocos-referência** |
| **hls4ml → HLS → RTL** | ⚠️ indireto (ver abaixo) | ✅ (via backend HLS) | ⚠️ médio | ⚠️ precisa io_stream + DRAM | **Candidato** |
| **PandA/Bambu (C→Verilog)** | ✅ (RTL genérico) | ✅ | ✅ (open-source) | ✅ com engine + DMA | **Candidato forte** |
| **NNgen (ONNX→Verilog+AXI)** | ✅ (RTL genérico) | ✅ | ✅ | ✅ já usa DRAM/DMA | **Candidato forte** |

Detalhamento a seguir.

---

## 2.1 OpenVINO — descartado

Sua suspeita está correta, e é ainda mais restritiva do que você pensava:

- O plugin FPGA do OpenVINO **nunca suportou a Arria V**. Ele existiu apenas para
  a **Arria 10** (nas placas *Intel Vision Accelerator Design / Mustang-F100-A10*
  e *Intel PAC with Arria 10 GX*).
- Mesmo para a Arria 10, a Intel **descontinuou** o plugin FPGA: ele saiu dos
  *releases* padrão já na versão **2020.2**, sobrevivendo só na LTS **2020.3.x**,
  e depois foi removido de vez (a Intel migrou para soluções de DL baseadas em
  FPGA de próxima geração).
- Além disso, o OpenVINO **não entrega Verilog**: ele carrega *bitstreams*
  pré-compilados e fechados numa placa suportada — o oposto do objetivo do
  projeto (RTL aberto e portável).

**Conclusão:** inaplicável à Arria V e contrário ao objetivo. Descartado.

## 2.2 OpenCL (Intel FPGA SDK for OpenCL) — descartado

- O **Intel FPGA SDK for OpenCL foi descontinuado** (PDN da Intel) e a Intel
  recomenda migrar para o oneAPI.
- Não há **BSP (Board Support Package)** disponível para Arria V — há BSPs para
  Arria 10, mas não para a família V. Sem BSP, o fluxo OpenCL não roda na placa.
- O fluxo OpenCL também não é "Verilog portável": ele compila para um *bitstream*
  específico do BSP, então não atende ao critério de independência.

**Conclusão:** descartado — sem BSP para Arria V e produto morto.

## 2.3 oneAPI (DPC++/C++ com FPGA add-on) — descartado para Arria V

- O oneAPI é o sucessor do OpenCL, mas suporta apenas **Arria 10, Stratix 10 e
  Agilex** — **Arria V não está na lista**.
- Pior: o suporte a FPGA do oneAPI foi **deprecado** e **deixa de existir a
  partir do compilador 2025.1** (as versões que ainda têm FPGA são até 2025.0).

**Conclusão:** não suporta Arria V e está em fim de vida. Descartado.

> **Observação importante sobre HLS na Arria V:** o **Intel HLS Compiler (`i++`)**
> — a ferramenta de C++→RTL da Intel, diferente de OpenCL/oneAPI — **lista a
> Arria V (e a Cyclone V) entre as famílias válidas do `-march`** (`ArriaV`,
> `CycloneV`, `Arria10`, `Cyclone10GX`, `MAX10`, `StratixV`, `Stratix10`). Mas
> só as edições/versões **até a 19.1 (Standard Edition, Quartus Prime Standard)**
> mantêm esse suporte; a partir da 20.1 a Standard Edition não traz mais o HLS, e
> a Pro Edition foca em famílias novas. **Portanto, se a via for HLS da Intel, o
> alvo prático é o Intel HLS Compiler Standard Edition ≤19.1 + Quartus Prime
> Standard.** Confirmar na sua instalação de Quartus qual versão você tem.

## 2.4 Porte manual em Verilog — só como referência

- **Viável tecnicamente?** Sim — Verilog roda em qualquer FPGA, inclusive Arria V.
- **Viável em esforço?** Não para o modelo inteiro. O gargalo **não é o número de
  camadas** (são poucas — ver [`01-modelo-e-analise.md`](01-modelo-e-analise.md)),
  e sim escrever, verificar e depurar à mão: o engine de convolução, o
  buffering, o controle de dataflow, a quantização, o DMA para DDR3 e o
  *sequenciamento* das 15 convoluções. Isso é meses de trabalho e altamente
  sujeito a erros.
- **Onde faz sentido:** escrever à mão **blocos isolados** (ex.: um MAC-array de
  Conv1D, um controlador DDR3) como *baseline* de comparação, ou para otimizar
  pontos críticos que o HLS gerar mal. Não como estratégia principal.

## 2.5 HLS (o caminho recomendado) — três variantes

A ideia que você levantou — **modelo → linguagem HLS intermediária (C/C++) →
Verilog** — é a correta. Existem três formas concretas:

### (a) hls4ml (Keras → HLS → RTL)
- Ferramenta consolidada (CERN/fastmachinelearning) que converte modelos Keras/
  QKeras/ONNX em projetos HLS e daí em Verilog/VHDL.
- Tem **backends Intel**: o *Quartus* (mirando o **Intel HLS**, hoje deprecado) e
  o *oneAPI* (mirando oneAPI 2025.0). Ambos suportam `io_parallel` e `io_stream`.
- **Ponto de atenção crítico:** o hls4ml foi pensado para modelos **muito
  pequenos** (física de partículas, kB de pesos) totalmente desenrolados no chip
  com `io_parallel`. Nosso modelo tem 6,4 M de params → **obrigatoriamente
  `io_stream`** e provavelmente **particionamento em subgrafos** (o hls4ml tem
  suporte a *subgraphs* para síntese em partes). Mesmo assim, manter 6,4 MB de
  pesos exige repensar armazenamento (DRAM), o que não é o caso de uso natural
  da ferramenta. **É factível, mas exige trabalho de engenharia não-trivial.**
- **Alvo de placa:** o backend Quartus mira o Intel HLS; combinar com Quartus
  Prime Standard ≤19.1 para chegar à Arria V (ver observação em 2.3).

### (b) PandA/Bambu (C/C++ → Verilog, open-source)
- **Bambu** é um compilador HLS **open-source** (GPL, Politecnico di Milano) que
  transforma C/C++ em Verilog/VHDL **agnóstico de fornecedor** — o RTL gerado
  sintetiza em Quartus para Arria V normalmente.
- Combina bem com um **gerador de código C de rede neural**: escreve-se (ou
  gera-se) o forward pass do modelo em C limpo (loops de convolução, buffers) e o
  Bambu produz o RTL. Já foi usado para implementar redes neurais em FPGA.
- **Vantagem:** máxima portabilidade e independência de fornecedor — exatamente o
  objetivo do projeto. **Desvantagem:** você controla (e escreve) mais do
  dataflow/quantização em C do que no hls4ml, que já traz camadas prontas.

### (c) NNgen (ONNX → Verilog + core AXI com DMA)
- **NNgen** é um compilador open-source que, a partir de um modelo (via
  **importador ONNX**), gera **Verilog RTL + IP-XACT** de um acelerador
  *all-inclusive*: PEs, memória on-chip, rede on-chip, **controlador DMA** e
  lógica de controle, com interface **AXI4**.
- **Encaixa na nossa maior restrição:** ele já assume **pesos em DRAM externa
  carregados por DMA** e trabalha com **quantização inteira** (int8/int16 com
  acumulador int64) — que é justamente o que nosso modelo de 6,4 MB exige.
- **Pontos de atenção:** o conjunto de operadores é focado em **conv2d, matmul,
  max_pool, reshape, relu**. Nosso modelo é **Conv1D** — que é matematicamente um
  caso particular de conv2d (kernel `Kx1`), então mapeável, mas exige adaptar as
  formas. Confirmar suporte a `Add` (skip) e à fusão de BatchNorm.

---

## 2.6 Recomendação (resumo — detalhes em `03-recomendacao-e-pipeline.md`)

Ordem de preferência sugerida, dado o objetivo de **portabilidade** + a
**restrição de memória** de 6,4 MB de pesos:

1. **NNgen** — porque já resolve nativamente a parte mais difícil (pesos em DRAM
   via DMA + quantização inteira + core AXI), gera Verilog puro e é
   independente de fornecedor. Melhor ajuste ao problema real.
2. **Bambu + gerador de C** — máxima portabilidade e controle; mais esforço de
   engenharia no dataflow, mas sem amarras de fornecedor.
3. **hls4ml (backend Quartus/oneAPI)** — mais "pronto" para camadas de NN, porém
   forçando `io_stream`/subgrafos e adaptando o armazenamento de pesos para um
   modelo bem maior do que o seu caso de uso típico; amarrado ao toolchain Intel.

Em todos os casos, dois pré-processamentos do modelo são comuns e recomendados:
**(i) fundir a BatchNorm nas convoluções** e **(ii) quantizar para int8/int16**
com validação de acurácia.
