# Via B — PandA/Bambu (C → Verilog)

`conv1d_engine.c` é o **engine de Conv1D reutilizável** (int16, BatchNorm já
fundida, acumulador int32) que o Bambu sintetiza em Verilog genérico para a
Arria V. A mesma lógica cobre as ~15 convoluções do modelo iterando camada a
camada — o custo de hardware **independe do número de camadas** (ver
[`../../docs/01-modelo-e-analise.md`](../../docs/01-modelo-e-analise.md) §1.5).

## Fluxo

1. **Pré-processar o modelo** (uma vez, em software):
   - `python3 ../01_fold_bn.py --weights model.hdf5` — funde BatchNorm nas convs.
   - `python3 ../02_export_onnx.py --weights model.hdf5` — dumpa `weights.npz`.
   - Quantizar `weights.npz` para int16 e gerar uma **tabela de configuração por
     camada** (n_in, c_in, c_out, k, stride, out_shift) + o blob de pesos que vai
     para a DDR3. (script de empacotamento a escrever na Etapa 2.)

2. **Sintetizar o engine com o Bambu:**
   ```bash
   bambu conv1d_engine.c \
     --top-fname=conv1d_layer \
     --device-name=<parte_arria_v> \   # ex.: 5AGXFB3H4F35C4 — CONFIRMAR o board
     --clock-period=10 \               # 100 MHz
     --generate-tb=testbench.xml \     # testbench p/ validar vs golden
     --simulate -v3
   ```
   O Bambu é HLS **open-source e agnóstico de fornecedor**: o Verilog gerado
   sintetiza no Quartus para a Arria V, sem depender de nenhum HLS proprietário.

3. **Integrar** o engine num top-level que: (a) lê a tabela de camadas, (b) faz
   DMA dos pesos da DDR3 por camada, (c) mantém as ativações em BRAM, (d)
   sequencia `conv1d_layer` / `maxpool1d` / `add_residual` na ordem do modelo.

## Status testado

`conv1d_engine.c` **compila limpo** com `gcc -std=c99 -Wall -Wextra` (validação
funcional do C). A síntese Bambu em si precisa do PandA/Bambu instalado e da
parte exata da Arria V — passos da Etapa 2/3 no ambiente do projeto.
