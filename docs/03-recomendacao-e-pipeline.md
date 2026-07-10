# 03 — Recomendação e pipeline proposto

## 3.1 Estratégia recomendada

Dado (a) o objetivo de **independência de fornecedor**, (b) a restrição dura de
**6,4 MB de pesos que não cabem no chip** e (c) a **latência-alvo folgada** (uma
inferência por janela de ~10 s), a recomendação é um **pipeline HLS de duas
etapas** com um acelerador **layer-by-layer** que reusa um único engine de
Conv1D e lê os pesos de **DDR3 externa**.

Concretamente, dois caminhos paralelos que podem ser prototipados e comparados:

```
                         ┌─────────────────────────────────────────────┐
   Keras (.hdf5)         │  PRÉ-PROCESSAMENTO (comum às duas vias)      │
   Ribeiro et al.  ─────▶│  1. fold BatchNorm → Conv (w', bias)         │
                         │  2. remover Dropout (no-op em inferência)    │
                         │  3. quantizar int16 (PTQ) + validar AUC/F1   │
                         │  4. exportar pesos + (ONNX ou C headers)     │
                         └───────────────┬──────────────┬──────────────┘
                                         │              │
                    VIA A (recomendada)  │              │  VIA B (portável)
                                         ▼              ▼
                   ┌─────────────────────────┐   ┌───────────────────────────┐
                   │ NNgen (ONNX → RTL)      │   │ Forward pass em C/C++     │
                   │ gera acelerador c/ DMA  │   │ (engine Conv1D + buffers) │
                   │ + AXI + Verilog + IP    │   │        │                  │
                   └───────────┬─────────────┘   │        ▼                  │
                               │                 │ PandA/Bambu (C → Verilog) │
                               │                 └───────────┬───────────────┘
                               ▼                             ▼
                   ┌───────────────────────────────────────────────────────┐
                   │  Verilog RTL  →  Quartus Prime (Standard) → Arria V    │
                   │  + DDR3 controller (pesos) + host/AXI para I/O do ECG  │
                   └───────────────────────────────────────────────────────┘
```

**Por que duas vias em paralelo:** a Via A (NNgen) tende a chegar mais rápido a
um acelerador funcional porque já resolve DMA+quantização; a Via B (Bambu) é o
plano de contingência totalmente aberto, sem amarras, útil se o mapeamento
Conv1D/skip no NNgen esbarrar em limitações de operadores.

> **hls4ml** fica como **comparação opcional**: além de exigir `io_stream` +
> subgrafos e adaptar o armazenamento de pesos, ele **depende de um back-end de
> HLS de fornecedor** (não emite RTL) — o que contraria o critério de ser geral
> e sem back-end próprio. Só entra se você pedir explicitamente. Ver §3.5.

## 3.2 Plano por etapas (marcos verificáveis)

### Etapa 0 — Reprodução e *baseline* de software (sem hardware ainda)
- [ ] Baixar os pesos treinados (Zenodo) e rodar `predict.py` original.
- [ ] Congelar um vetor de teste (entrada `(4096,12)` → saída `(6)`) como
      *golden reference* para validar o hardware bit-a-bit depois.

### Etapa 1 — Preparação do modelo para hardware
- [ ] Implementar a **fusão BatchNorm→Conv** e remover Dropout; verificar que a
      saída não muda (fp32).
- [ ] **Quantização pós-treino em int16** (decidido); medir queda de **AUC/F1**
      no conjunto de teste do repositório e confirmar que é aceitável.
- [ ] Exportar o modelo quantizado em **ONNX** (para NNgen) e pesos crus +
      *header* C (para Bambu).

### Etapa 2 — Protótipo de acelerador (as duas vias)
- [ ] **Via A:** importar o ONNX no NNgen; mapear Conv1D como conv2d `Kx1`;
      resolver `Add` (skip) e MaxPool; gerar Verilog + testbench; simular contra
      o *golden reference*.
- [ ] **Via B:** escrever o forward pass em C (engine Conv1D reusável, buffers em
      BRAM, pesos vindos de um ponteiro para DRAM); rodar Bambu; simular RTL.

### Etapa 3 — Integração na Arria V
- [ ] Instanciar **controlador DDR3** (pesos) + caminho de entrada do ECG
      (AXI/UART/…); *place & route* no Quartus Prime **Standard** para a parte
      exata da placa; fechar *timing*.
- [ ] Validar em hardware contra o *golden reference*; medir latência e recursos.

### Etapa 4 — Otimização
- [ ] Ajustar paralelismo do engine (nº de MACs/DSPs), *tiling* e largura de
      banda de DRAM ao *budget* de latência.

## 3.3 Orçamento de recursos (estimativa preliminar)

Ver números completos em [`04-arria-v-recursos.md`](04-arria-v-recursos.md). Em resumo:

- **Pesos:** 6,4 MB (int8) → **DDR3 externa obrigatória**. On-chip guarda só o
  *tile* de pesos e ativações da camada atual.
- **Ativações:** o maior mapa é a saída do stem, `4096×64` = 262 k elementos →
  256 KB @ int8. Cabe on-chip com *double buffering* modesto; se apertar, também
  vai para DRAM.
- **DSPs:** a Arria V GX A7 tem ~800 blocos DSP variable-precision. Com dezenas a
  poucas centenas de MACs paralelos, 1,83 GMAC sai em poucos ms a ~100–150 MHz —
  **muito** dentro do orçamento de latência de 10 s.
- **Conclusão:** o gargalo de projeto é **memória/banda**, não computação.

## 3.4 Riscos e pontos a confirmar

| Risco | Mitigação |
|-------|-----------|
| Queda de acurácia com int8 | testar int16; quantização por canal; validar AUC/F1 |
| Operadores Conv1D/skip no NNgen | mapear Conv1D→conv2d Kx1; se `Add` faltar, via Bambu |
| Versão do Intel HLS que suporta Arria V (≤19.1) | Via A/B (NNgen/Bambu) geram RTL puro e evitam essa dependência |
| Controlador DDR3 e banda | latência folgada permite acesso serializado simples |
| Parte exata da Arria V desconhecida | confirmar com você (ver decisões abaixo) |

## 3.5 Decisões já tomadas (rodada 1)

| # | Decisão | Resposta | Consequência de projeto |
|---|---------|----------|-------------------------|
| 1 | Modelo-alvo | **Modelo completo** (6 saídas, entrada 4096×12, ~6,43 M params) | Sem poda; dataflow layer-by-layer + DDR3 obrigatórios |
| 2 | Placa / DDR3 | **Arria V com DDR3 externa** (capacidade a confirmar, "sobrando") | Pesos e I/O na DDR3; ver estimativa em [`04-arria-v-recursos.md`](04-arria-v-recursos.md) (§4.4) |
| 3 | Precisão | **int16** (decidido — ver justificativa abaixo) | Pesos ~12,85 MB na DDR3; acumulador int32/int48; menor risco de acurácia |
| 4 | Toolchain | **Duas vias, ambas gerais e sem back-end de fornecedor: NNgen + Bambu** | hls4ml rebaixado a comparação opcional (depende de back-end de fornecedor) |

### Justificativa da precisão (int16)

int16 é a escolha certa aqui, e por um motivo que dispensa trade-off:

- **O custo de área/memória do int16 sobre o int8 é pequeno neste caso.** Os
  pesos vão para a DDR3 de qualquer forma (nem int8 cabe on-chip), e há DDR3
  "sobrando" — então dobrar 6,4 MB → 12,85 MB é irrelevante para a capacidade e
  para a banda (ver §4.5: a banda exigida é da ordem de dezenas de MB/s, ~100×
  abaixo do que uma DDR3 entrega).
- **Não há pressão de latência** (uma inferência a cada ~10 s), então o eventual
  uso de 2 DSPs por MAC em vez de empacotar 2–3 MACs int8 por DSP também é
  irrelevante — sobram DSPs de sobra.
- **Em troca, int16 preserva a acurácia** (AUC/F1) com margem muito maior que
  int8, reduzindo o risco de a quantização degradar o diagnóstico — que é o
  ponto sensível de um modelo clínico.

Ou seja: como memória, banda e DSP estão todos folgados, **paga-se quase nada
por escolher a precisão mais segura**. Mantemos int8 apenas como *otimização
opcional* futura, se algum dia houver pressão de área.

### Por que NNgen + Bambu (e não hls4ml) para atender "sem back-end próprio"

- **NNgen** gera **Verilog RTL diretamente** (via Veriloggen) — não usa nenhum
  HLS de fornecedor. O RTL + core AXI sintetiza no Quartus para a Arria V.
- **Bambu (PandA)** é HLS **open-source e agnóstico de fornecedor**: C/C++ →
  Verilog genérico, que também sintetiza no Quartus para a Arria V.
- **hls4ml NÃO emite RTL** — emite C++ de HLS que exige um back-end de fornecedor
  (no caminho Intel, o Intel HLS, que só cobre a Arria V até a 19.1). Portanto
  ele **tem, por natureza, um back-end de fornecedor** — contrário ao critério
  "geral". Fica como comparação opcional apenas se você quiser.

## 3.6 Decisões ainda em aberto (não bloqueantes)

Não travam o início do trabalho, mas vou precisar delas mais à frente:

1. **Parte/kit exato da Arria V** (ex.: `5AGX…`, GX/SX, A7/B3) e **quanto de
   DDR3** o board tem — para fechar *place & route* e *timing* na Etapa 3.
2. **Interface de I/O**: como o ECG entra e o resultado sai (host via PCIe/AXI,
   UART, memória pré-carregada) — define o *wrapper* de topo na Etapa 3.

Nada disso impede começar as Etapas 0–2 (baseline de software, fusão de BN,
quantização int16 e os dois protótipos de acelerador em simulação).
