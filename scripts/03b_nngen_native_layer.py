#!/usr/bin/env python3
"""Etapa 2 - Via A (API nativa do NNgen): gera Verilog RTL de UMA camada de
convolucao do modelo, com as dimensoes reais, usando a API nativa do NNgen.

Motivacao (ver docs/05 secao 4): o import via ONNX (tf2onnx) do modelo Conv1D/2D
esbarra em lacunas do front-end do NNgen 1.3.4 (padding 'SAME' de convolucoes com
stride nao reproduz as formas do Keras -> o Add residual diverge). A API NATIVA
contorna isso: definimos a conv diretamente, com stride/padding explicitos, e o
NNgen emite o RTL. Este script prova que o NNgen gera Verilog sintetizavel para o
nosso operador-alvo (Conv 'k x 1' int16), que e o bloco reutilizavel do
acelerador layer-by-layer.

Uso:
    python3 scripts/03b_nngen_native_layer.py
Saida:
    outputs/nngen/ecg_conv_layer.v   (Verilog RTL)
"""
import argparse, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs", "nngen"))
    ap.add_argument("--topname", default="ecg_conv_layer")
    # dimensoes de uma camada representativa (1a conv do bloco residual 1):
    # entrada H=4096 amostras, W=1, C_in=64 -> C_out=128, kernel (16,1), 'same'
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--Cin", type=int, default=64)
    ap.add_argument("--Cout", type=int, default=128)
    ap.add_argument("--K", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import nngen as ng

    dt = ng.int16
    # tensores no layout NHWC (nativo do NNgen)
    x = ng.placeholder(dt, shape=(1, args.H, 1, args.Cin), name="signal")
    w = ng.variable(dt, shape=(args.Cout, args.K, 1, args.Cin), name="conv_w")
    b = ng.variable(ng.int32, shape=(args.Cout,), name="conv_b")

    # Conv2D 'k x 1' com padding SAME (stride 1). rshift requantiza int32->int16.
    y = ng.conv2d(x, w, strides=(1, 1, 1, 1), padding="SAME",
                  bias=b, act_func=ng.relu,
                  dtype=dt, sum_dtype=ng.int32, name="conv")

    # aloca off-chip (DRAM) para pesos e ativacoes, e gera o RTL + AXI/DMA
    targ = ng.to_veriloggen([y], args.topname, silent=True,
                            config={"maxi_datawidth": 128,
                                    "default_datawidth": 16})
    rtl_path = os.path.join(args.outdir, f"{args.topname}.v")
    with open(rtl_path, "w") as f:
        f.write(targ.to_verilog())
    n_lines = sum(1 for _ in open(rtl_path))
    print(f"[ok] Verilog RTL gerado: {rtl_path} ({n_lines} linhas)")
    print("     modulo top:", args.topname,
          f"| conv {args.Cin}->{args.Cout}, kernel {args.K}x1, H={args.H}, int16")


if __name__ == "__main__":
    main()
