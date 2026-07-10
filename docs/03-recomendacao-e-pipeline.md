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
                         │  3. quantizar int8/int16 (PTQ) + validar AUC │
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

> **hls4ml** fica como terceira via / comparação: útil se você quiser reaproveitar
> as camadas prontas da ferramenta, mas exige forçar `io_stream` + subgrafos e
> adaptar o armazenamento de pesos — mais atrito para este tamanho de modelo.

## 3.2 Plano por etapas (marcos verificáveis)

### Etapa 0 — Reprodução e *baseline* de software (sem hardware ainda)
- [ ] Baixar os pesos treinados (Zenodo) e rodar `predict.py` original.
- [ ] Congelar um vetor de teste (entrada `(4096,12)` → saída `(6)`) como
      *golden reference* para validar o hardware bit-a-bit depois.

### Etapa 1 — Preparação do modelo para hardware
- [ ] Implementar a **fusão BatchNorm→Conv** e remover Dropout; verificar que a
      saída não muda (fp32).
- [ ] **Quantização pós-treino** (int8 e int16); medir queda de **AUC/F1** no
      conjunto de teste do repositório. Definir a menor precisão aceitável.
- [ ] Exportar o modelo quantizado em **ONNX** (para NNgen) e/ou pesos crus +
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

## 3.5 Decisões que dependem de você (para a próxima rodada)

Preciso destas definições para afinar o plano e começar a Etapa 0/1:

1. **Modelo-alvo:** portamos o modelo original completo (6 saídas, entrada
   4096×12), ou há intenção de **podar/reduzir** (menos canais, menos derivações,
   janela menor) para caber com folga? A poda reduz muito o custo de hardware.
2. **Placa exata:** qual **parte** da Arria V (ex.: `5AGXMA…`, GX/SX, A7/B3…) e
   qual **kit/board**? Isso fixa memória, DSPs e a existência de DDR3 no board.
3. **Precisão aceitável:** qual queda de AUC/F1 é tolerável na sua pesquisa?
   (define int8 vs int16 vs manter fp32 em partes sensíveis).
4. **Toolchain preferido:** priorizamos a via **open-source/portável**
   (NNgen/Bambu) — coerente com o objetivo de generalidade — ou você quer que eu
   também prototipe a via **hls4ml/Intel HLS** para comparação?
5. **Interface de I/O:** como o ECG entra e o resultado sai na sua montagem
   (host via PCIe/AXI, UART, memória pré-carregada)? Define o *wrapper* de topo.

Com as respostas (principalmente 1, 2 e 4) eu avanço para a Etapa 0/1 com código.
