# Briefing para gerar a apresentação (parcial) em slides

> **Para a outra sessão do Claude:** este documento é o insumo para você montar
> um deck de slides de **status parcial** desta pesquisa. Abaixo há (A) fatos e
> números canônicos, (B) uma estrutura de deck sugerida slide a slide com o
> conteúdo, e (C) notas de tom/estilo. Pode reorganizar, mas **não invente
> números** — use os desta página. Contexto é uma pesquisa científica em
> andamento; o deck é um **relatório de progresso**, não um produto final.

---

## A. Fatos e números canônicos (fonte da verdade)

**Objetivo do projeto**
- Portar o modelo de diagnóstico automático de ECG de 12 derivações de Ribeiro
  et al. (Nature Communications, 2020; repo `antonior92/automatic-ecg-diagnosis`,
  TensorFlow/Keras) para **Verilog RTL** rodando numa FPGA **Intel/Altera Arria V**.
- Rodar o *forward pass* (modelo com pesos/bias congelados) em hardware.
- Requisito-chave: ser o mais **independente de fornecedor** possível (por isso
  Verilog, e não ferramentas fechadas), para reaproveitar em outras placas.

**O modelo (ResNet 1D)**
- Entrada: `(4096, 12)` — ~10 s de ECG a 400 Hz, 12 derivações.
- Saída: `(6)` — probabilidade (sigmoide) de 6 anormalidades cardíacas.
- Topologia: 1 conv de entrada + **4 blocos residuais** + Flatten + Dense(6).
  Operações: Conv1D (incl. 1×1), MaxPool1D, Add (skip), ReLU, Dense/sigmoide.
- **Parâmetros: 6.425.638 (~6,43 M)** — confirmado no Keras.
- **~1,83 GMAC por inferência** (~3,65 GFLOP).
- Pesos: fp32 ~25,7 MB · int16 ~12,85 MB · int8 ~6,43 MB.

**A restrição que define a arquitetura**
- Memória interna da Arria V: ~1,7 MB (GX A7) a ~2,1 MB (GX B3) de M10K.
- Pesos **não cabem on-chip** → **DDR3 externa obrigatória** + streaming.
- DDR3 necessária: só **~16–20 MB**; banda exigida ~1,3–90 MB/s contra ~1,6–4 GB/s
  disponíveis → **nem capacidade nem banda são gargalo**.
- Latência-alvo folgada (1 exame/~10 s) → acelerador **pequeno e serializado**,
  um único **engine de Conv1D reutilizado camada a camada** (custo de hardware
  independe do nº de camadas). Gargalo é memória, não cálculo.

**Decisões tomadas**
- Modelo: **completo** (sem poda).
- Precisão: **int16** (custo irrelevante sobre int8 aqui; preserva acurácia —
  importante em contexto clínico).
- Toolchain: **duas vias HLS, ambas gerais e sem back-end de fornecedor** —
  **NNgen** (ONNX→Verilog) e **PandA/Bambu** (C→Verilog). hls4ml descartado como
  via principal por depender de back-end de fornecedor (não emite RTL).

**Alternativas avaliadas (e por que caíram)**
- **OpenVINO**: só Arria 10, descontinuado (2020), entrega bitstream fechado. ❌
- **OpenCL (Intel FPGA SDK)**: descontinuado; sem BSP para Arria V. ❌
- **oneAPI**: só Arria 10/Stratix 10/Agilex; FPGA deprecado em 2025.1. ❌
- **Porte manual RTL**: inviável p/ 6,4 M de params; só p/ blocos-referência. ⚠️
- **hls4ml**: funciona, mas depende de back-end de fornecedor (Intel HLS/oneAPI). ⚠️
- **NNgen / Bambu**: geram Verilog universal; escolhidos. ✅

**O que já foi EXECUTADO e validado (não é plano — rodou de verdade)**
1. ✅ Contagem de parâmetros confirmada no Keras: **6.425.638**.
2. ✅ Fusão de BatchNorm nas convoluções: 13 pares Conv→BN fundidos.
3. ✅ Exportação para ONNX (25,7 MB fp32).
4. ✅ Modelo reescrito em **Conv2D (H×1)** — idêntico (mesmos 6,43 M params) —
   para o NNgen consumir sem os *hacks* de Conv1D.
5. ✅ **RTL Verilog gerado pelo NNgen (API nativa)** para uma camada real do
   modelo (Conv 64→128, kernel 16×1, int16): **61.572 linhas**, **parse OK no
   Icarus Verilog**, com **interface AXI4 mestre para DDR3**. É o "motor"
   reutilizável do acelerador.
6. ✅ Engine de Conv1D em C para Bambu compila limpo (`gcc -Wall -Wextra`).

**Fricções encontradas e resolvidas no caminho do NNgen (honestidade técnica)**
- Dense final vinha como `MatMul` (não `Gemm`) → contornado.
- `Unsqueeze` opset 13 (axes como input) → fix embutido (axes→atributo).
- NNgen 1.3.4 quebra com NumPy ≥1.24 → fixado `numpy<1.24`.
- Conv1D "empacotado" (Unsqueeze→Conv2D→Squeeze) não casava com layout →
  resolvido reescrevendo em Conv2D.
- Padding 'SAME' de convs com stride diverge do Keras no Add residual →
  contornado pela **API nativa** (stride/padding explícitos).

**O que falta (mecânico, sem incógnitas de viabilidade)**
1. Encadear as ~15 convoluções pela API nativa do NNgen → RTL do backbone completo.
2. Baixar pesos reais (Zenodo) + CODE-test e medir **queda de AUC/F1 em int16**.
3. Síntese Bambu → Verilog (via B, comparação).
4. Integração na Arria V (Quartus + DDR3 + I/O), timing, execução na placa —
   exige toolchain Intel + hardware físico (não feito neste ambiente).

**Pontos abertos (não bloqueiam software)**
- Parte/kit exato da Arria V e tamanho da DDR3 do board.
- Interface de I/O do ECG (host/AXI/UART/memória).

---

## B. Estrutura de deck sugerida (slide a slide)

Deck de ~12–14 slides, relatório de progresso. Conteúdo por slide:

1. **Capa** — "Porte do modelo de diagnóstico de ECG para FPGA (Arria V) —
   Relatório parcial". Subtítulo: modelo TensorFlow/Keras → Verilog RTL.
2. **Objetivo e motivação** — rodar a IA de ECG em hardware; por que FPGA/Verilog
   (portabilidade, independência de fornecedor, aplicação portátil/embarcada).
3. **O modelo** — entrada `(4096,12)` → saída `(6)`; ResNet 1D; diagrama das
   camadas (stem + 4 blocos residuais + Dense); as 5 operações.
4. **Números que definem o projeto** — 6,43 M params; 1,83 GMAC/inferência;
   tabela de tamanho dos pesos (fp32/int16/int8).
5. **A restrição central** — pesos (6–26 MB) não cabem na memória interna
   (~1,7 MB) → DDR3 externa + engine reutilizável layer-by-layer. (Gráfico de
   barras: pesos vs memória on-chip.)
6. **Decisões** — modelo completo; **int16** (com justificativa); duas vias
   gerais (NNgen + Bambu).
7. **Alternativas avaliadas** — tabela: OpenVINO/OpenCL/oneAPI/manual/hls4ml/
   NNgen/Bambu × serve para Arria V? × emite Verilog? × sem back-end de fornecedor?
8. **Estratégia/pipeline** — diagrama: Keras → (fold BN + quantize int16) →
   {NNgen: ONNX→Verilog} e {Bambu: C→Verilog} → Quartus → Arria V + DDR3.
9. **O que já foi executado** — checklist com os 6 itens validados (marcar ✅).
10. **Destaque: RTL gerado** — a camada Conv 64→128 int16; 61.572 linhas; parse
    OK no iverilog; interface AXI4/DDR3. (Pode mostrar o excerto do módulo.)
11. **Fricções técnicas superadas** — bullets curtos (Conv1D→2D, opset, numpy,
    padding) mostrando o trabalho de engenharia real e honesto.
12. **Orçamento de hardware na Arria V** — DDR3 ~16–20 MB, banda folgada,
    latência folgada; gargalo é memória, não cálculo.
13. **Próximos passos** — backbone completo em RTL; validação de acurácia int16;
    síntese Bambu; integração na placa.
14. **Fechamento** — status: viabilidade comprovada; ferramentas certas; RTL do
    bloco-chave já gerado e verificado.

---

## C. Tom e estilo
- Público: acadêmico/pesquisa, mas incluir 1 slide acessível (o modelo/motivação).
- Ser **honesto sobre o parcial**: o que está provado vs o que falta. Não afirmar
  que o modelo inteiro já roda em FPGA (não roda ainda) — o que está feito é o
  bloco-chave em RTL + toda a análise/decisões.
- Usar os números exatos desta página. Evitar superlativos.
- Se gerar visualizações, uma boa é a barra "pesos (6,4 MB int8) vs memória
  on-chip (1,7 MB)" — comunica a restrição central de imediato.
- Referências: Ribeiro et al. 2020; NNgen; PandA/Bambu; docs deste repo
  (`docs/01`–`06`, `rtl/`).
