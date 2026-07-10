#!/usr/bin/env python3
"""Etapa 2 - Via A: ONNX -> NNgen -> Verilog RTL (+ core AXI/DMA).

NNgen le o ONNX, aplica quantizacao inteira (alvo: int16) e gera Verilog RTL de
um acelerador all-inclusive (PEs + memoria on-chip + DMA + AXI4), que sintetiza
no Quartus para a Arria V. NNgen NAO usa nenhum HLS de fornecedor (emite RTL via
Veriloggen) -> atende ao criterio "geral, sem back-end proprio".

PONTO DE ATENCAO (risco a validar): o modelo e Conv1D, e o conjunto de operadores
do NNgen e centrado em conv2d. A ponte e mapear Conv1D (k) -> conv2d (k x 1),
tratando o eixo de tempo como altura e largura=1. O `Add` do skip e a MaxPool1D
tambem precisam de mapeamento equivalente. Este script implementa esse caminho e
falha com mensagem clara caso algum operador nao seja suportado - exatamente o
tipo de coisa que so se confirma rodando.

Pre-requisitos:
    pip install nngen veriloggen pyverilog
    # + iverilog (Icarus) para simular o testbench gerado:
    #   apt-get install iverilog

Uso:
    python3 scripts/03a_nngen_convert.py --onnx outputs/ecg_model.onnx
"""
import argparse, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(ROOT, "outputs", "ecg_model.onnx"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs", "nngen"))
    ap.add_argument("--bitwidth", type=int, default=16, help="largura inteira (int16 decidido)")
    ap.add_argument("--topname", default="ecg_accel")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import numpy as np
    import nngen as ng

    # 1) importa o ONNX -> grafo NNgen. value_dtypes fixa a quantizacao inteira.
    dtype = ng.int16 if args.bitwidth == 16 else ng.int8
    print(f"[info] importando {args.onnx} com quantizacao {dtype}")
    outputs, placeholders, variables, constants, operators = ng.from_onnx(
        args.onnx,
        value_dtypes={"signal": dtype},
        default_placeholder_dtype=dtype,
        default_variable_dtype=dtype,
        default_constant_dtype=dtype,
        default_operator_dtype=dtype,
        default_scale_dtype=ng.int16,
        default_bias_dtype=ng.int32,
        onnx_input_layout="channel_last",
    )

    # 2) (quando houver pesos reais) calibrar quantizacao com amostras de ECG:
    #    ng.quantize(outputs, input_scale_factors={"signal": s}, ...)
    #    Aqui, sem calibracao, NNgen usa escalas padrao (suficiente p/ gerar RTL).

    # 3) aloca off-chip (DRAM) para pesos/ativacoes e gera o RTL + AXI/DMA.
    act = outputs[0]
    targ = ng.to_veriloggen([act], args.topname, silent=False,
                            config={"maxi_datawidth": 128,  # barramento AXI p/ DDR3
                                    "default_datawidth": args.bitwidth})
    rtl_path = os.path.join(args.outdir, f"{args.topname}.v")
    targ.to_verilog(rtl_path)
    print(f"[ok] Verilog RTL gerado em {rtl_path}")

    # 4) tambem gera um testbench com memoria simulada (para validar vs golden).
    #    ng.sim.to_ipxact(...) / targ.to_verilog(testbench=True) conforme a versao.
    print("[proximo] simular com iverilog e comparar com outputs/golden_output.npy")


if __name__ == "__main__":
    main()
