#!/usr/bin/env python3
"""Etapa 2 - Via A: ONNX -> NNgen -> Verilog RTL (+ core AXI/DMA).

NNgen le o ONNX, aplica quantizacao inteira (alvo: int16) e gera Verilog RTL de
um acelerador all-inclusive (PEs + memoria on-chip + DMA + AXI4), que sintetiza
no Quartus para a Arria V. NNgen NAO usa nenhum HLS de fornecedor (emite RTL via
Veriloggen) -> atende ao criterio "geral, sem back-end proprio".

ESTADO (validado executando - ver docs/05-testes-realizados.md):
  O NNgen importa TODO o backbone convolucional (Conv1D->conv2d, Add/skip,
  MaxPool, BN, ReLU). Ao rodar o import de ponta a ponta, foram encontrados e
  tratados tres pontos de compatibilidade:
    1. Unsqueeze/Squeeze opset 13 (axes como input) -> este script converte
       axes para ATRIBUTO (fix_unsqueeze_axes) para o importador antigo do NNgen.
    2. NNgen 1.3.4 usa np.float/np.int (removidos no numpy novo) -> exige
       numpy<1.24 (ver scripts/requirements.txt).
    3. LIMITACAO DE FUNDO: o tf2onnx serializa Conv1D como Unsqueeze->Conv2D->
       Squeeze (dim sintetica), e o bookkeeping de layout do NNgen quebra nisso.
       A correcao robusta e expressar o modelo como Conv2D (H x 1) ANTES de
       exportar (ECG como imagem 4096x1x12) OU usar a API nativa do NNgen.
       Enquanto isso, a via B (Bambu, C puro) nao tem nenhuma dessas fricoes.

Uso:
    python3 scripts/03a_nngen_convert.py --onnx outputs/ecg_model.onnx \
            [--output-tensor <nome>]   # corta o grafo (ex.: so o backbone)
"""
import argparse, os


def fix_unsqueeze_axes(in_path, out_path):
    """Converte Unsqueeze/Squeeze do estilo opset-13 (axes como input) para o
    estilo opset-11 (axes como atributo), que e o que o importador do NNgen le."""
    import onnx
    from onnx import helper, numpy_helper
    m = onnx.load(in_path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    n = 0
    for node in m.graph.node:
        if node.op_type in ("Unsqueeze", "Squeeze") and len(node.input) == 2:
            if node.input[1] in inits:
                axes = [int(x) for x in inits[node.input[1]].tolist()]
                del node.input[1]
                node.attribute.append(helper.make_attribute("axes", axes))
                n += 1
    onnx.save(m, out_path)
    print(f"[fix] {n} Unsqueeze/Squeeze: axes -> atributo -> {out_path}")
    return out_path


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(ROOT, "outputs", "ecg_model.onnx"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs", "nngen"))
    ap.add_argument("--bitwidth", type=int, default=16, help="largura inteira (int16 decidido)")
    ap.add_argument("--topname", default="ecg_accel")
    ap.add_argument("--output-tensor", default=None,
                    help="corta o grafo neste tensor (ex.: so o backbone conv)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import numpy as np
    import onnx
    import nngen as ng

    onnx_path = args.onnx
    # corte opcional do grafo (ex.: backbone ate o ultimo ReLU antes do Flatten)
    if args.output_tensor:
        import onnx.utils
        cut = os.path.join(args.outdir, "cut.onnx")
        onnx.utils.extract_model(onnx_path, cut, input_names=["signal"],
                                 output_names=[args.output_tensor])
        onnx_path = cut
    # aplica o fix de axes (Unsqueeze/Squeeze opset 13 -> atributo)
    onnx_path = fix_unsqueeze_axes(onnx_path, os.path.join(args.outdir, "prepared.onnx"))

    # 1) importa o ONNX -> grafo NNgen. value_dtypes fixa a quantizacao inteira.
    dtype = ng.int16 if args.bitwidth == 16 else ng.int8
    print(f"[info] importando {onnx_path} com quantizacao {dtype}")
    outputs, placeholders, variables, constants, operators = ng.from_onnx(
        onnx_path,
        value_dtypes={"signal": dtype},
        default_placeholder_dtype=dtype,
        default_variable_dtype=dtype,
        default_constant_dtype=dtype,
        default_operator_dtype=dtype,
        default_scale_dtype=ng.int16,
        default_bias_dtype=ng.int32,
        # layout: manter os defaults do NNgen (ONNX vem NCHW do tf2onnx;
        # NNgen trabalha internamente em NHWC). Passar strings invalidas aqui
        # causava "ValueError: substring not found" em transpose_layout.
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
