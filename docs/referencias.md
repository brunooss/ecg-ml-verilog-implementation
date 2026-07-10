# Referências

## Modelo e artigo original
- Ribeiro et al., "Automatic diagnosis of the 12-lead ECG using a deep neural
  network", *Nature Communications* 11, 1760 (2020).
  https://www.nature.com/articles/s41467-020-15432-4
- Repositório de código: https://github.com/antonior92/automatic-ecg-diagnosis
- Pré-print: https://arxiv.org/abs/1904.01949
- Pesos treinados (Zenodo): https://doi.org/10.5281/zenodo.3625017

## OpenVINO / FPGA (Arria 10, descontinuado)
- Release Notes OpenVINO 2020 (remoção do plugin FPGA Arria 10 a partir de
  2020.2; LTS 2020.3):
  https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2020.html
- Release Notes OpenVINO 2020.3 LTS:
  https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2020-3.html

## OpenCL / oneAPI (sem Arria V)
- Intel FPGA SDK for OpenCL — Support Center (descontinuação, BSPs):
  https://www.intel.com/content/www/us/en/support/programmable/support-resources/design-guidance/opencl-software-support.html
- Discussão Arria V GX BSP para OpenCL:
  https://community.altera.com/discussions/boards-and-dev-kits/arria-v-gx-board-support-package-for-opencl-quartus-standard-v19-1/103904
- FPGA Support Package para oneAPI DPC++/C++ (famílias suportadas; deprecação
  do suporte a FPGA):
  https://www.intel.com/content/www/us/en/developer/tools/oneapi/fpga.html
- Lista de FPGAs suportadas com oneAPI (Arria 10 / Stratix 10 / Agilex):
  https://community.intel.com/t5/Intel-High-Level-Design/list-of-supported-FPGA-with-oneAPI/td-p/1282988

## Intel HLS Compiler (i++) — suporta Arria V até 19.1
- Overview HLS Compiler Standard Edition 19.1:
  https://www.intel.com/content/www/us/en/docs/programmable/683306/19-1/overview-of-the-standard-edition.html
- Command options (`-march` com famílias ArriaV/CycloneV/…):
  https://www.intel.com/content/www/us/en/docs/programmable/683310/19-1/standard-edition-command-options.html
- "Can I use Cyclone V with the Intel HLS compiler?" (19.1 é a última Standard
  com suporte às famílias V):
  https://malt.zendesk.com/hc/en-us/articles/900006639383-Can-I-use-Cyclone-V-with-the-Intel-HLS-High-Level-Synthesis-compiler
- Comparação PRO / STANDARD / LITE:
  https://www.intel.com/content/dam/www/central-libraries/us/en/documents/quartus-prime-compare-editions-guide.pdf

## hls4ml
- Repositório: https://github.com/fastmachinelearning/hls4ml
- Paper 2025 (backends Quartus/oneAPI, io_stream, subgrafos):
  https://arxiv.org/abs/2512.01463 · https://arxiv.org/html/2512.01463v1
- oneAPI backend (exige oneAPI 2025.0; 2025.1 não funciona):
  https://fastmachinelearning.org/hls4ml/advanced/oneapi.html
- CNNs em FPGA com hls4ml: https://arxiv.org/pdf/2101.05108

## PandA / Bambu (HLS open-source, C→Verilog)
- Projeto PandA-Bambu (Politecnico di Milano): https://panda.dei.polimi.it/
- Exemplo de ANN em FPGA via Bambu:
  https://github.com/socks2309/neural-network-fpga

## NNgen (ONNX → Verilog + AXI, open-source)
- Repositório: https://github.com/NNgen/nngen
- README (operadores, ONNX import, DMA/DRAM, quantização int16/int64):
  https://github.com/NNgen/nngen/blob/develop/README.md
- Backend Veriloggen: https://github.com/PyHDI/veriloggen

## Arria V — recursos de hardware
- Arria V Device Overview (AV-51001), tabelas de recursos GX/SX:
  https://www.mouser.com/datasheet/2/612/av_51001-1623623.pdf
- Overview da família Arria V:
  https://media.digikey.com/pdf/Data%20Sheets/Altera%20PDFs/Arria_V_Family.pdf
- DSP variable-precision (Arria V / Cyclone V):
  https://people.ece.cornell.edu/land/courses/ece5760/DE1_SOC/DSP_wp-01159-arriav-cyclonev-dsp.pdf
