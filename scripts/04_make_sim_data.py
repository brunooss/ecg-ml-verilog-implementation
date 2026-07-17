#!/usr/bin/env python3
"""Etapa 3 - Simulacao com RAM: gera o RTL da instancia de teste, a imagem de
memoria (RAM simulada) e a saida de referencia (golden) para o testbench.

O acelerador gerado pelo NNgen nao recebe amostras por streaming direto: ele
le sinal + pesos de uma DDR3 externa via AXI4. Na simulacao, a DDR3 e
substituida por rtl/sim/axi_ram_model.v, carregada com a imagem hex produzida
por este script. O fluxo:

  1. monta o MESMO grafo do scripts/03b (conv KxL int16, SAME, bias, ReLU),
     por padrao em dimensoes reduzidas para a simulacao terminar em segundos;
  2. fixa pesos/bias/entrada com valores reprodutiveis (seed) e magnitudes
     que nao saturam o int16 (verificacao exata, sem tolerancia);
  3. gera o RTL (ng.to_veriloggen) -> outputs/nngen/<topname>.v
     - os enderecos DRAM de cada tensor (x.addr, w.addr, y.addr) vem do
       proprio NNgen, entao imagem de memoria e RTL ficam sempre coerentes;
  4. calcula o golden em software com ng.eval (mesma semantica do RTL);
  5. escreve:
        sim/data/ram_init.hex   imagem da RAM (1 palavra de 128 bits/linha)
        sim/data/golden.hex     saida esperada (1 palavra de 16 bits/linha)
        sim/data/sim_config.vh  parametros para o testbench (`defines)

Uso:
    python3 scripts/04_make_sim_data.py                  # dims reduzidas
    python3 scripts/04_make_sim_data.py --H 4096 --Cin 64 --Cout 128
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def to_hex_image(mem_bytes):
    """bytes -> linhas hex de 128 bits (little-endian: byte 0 = bits [7:0])."""
    assert len(mem_bytes) % 16 == 0
    lines = []
    for i in range(0, len(mem_bytes), 16):
        word = mem_bytes[i:i + 16]
        lines.append(word[::-1].hex())
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "sim", "data"))
    ap.add_argument("--rtldir", default=os.path.join(ROOT, "outputs", "nngen"))
    ap.add_argument("--topname", default="ecg_conv_layer_sim")
    # dimensoes reduzidas por padrao: mesmo operador (conv Kx1 int16 SAME,
    # bias, ReLU), menos canais para a simulacao no iverilog ser rapida.
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--Cin", type=int, default=8)
    ap.add_argument("--Cout", type=int, default=8)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=20_000_000,
                    help="watchdog do testbench, em ciclos")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.rtldir, exist_ok=True)

    import nngen as ng

    # o barramento tem 128 bits; com C multiplo de 8 cada linha do tensor
    # ocupa beats inteiros e nao ha padding de alinhamento entre linhas.
    for name, c in (("Cin", args.Cin), ("Cout", args.Cout)):
        if (c * 2) % 16 != 0:
            raise SystemExit(f"{name}={c}: use multiplo de 8 (alinhamento do "
                             f"barramento de 128 bits)")

    # 1) grafo identico ao 03b ------------------------------------------------
    dt = ng.int16
    x = ng.placeholder(dt, shape=(1, args.H, 1, args.Cin), name="signal")
    w = ng.variable(dt, shape=(args.Cout, args.K, 1, args.Cin), name="conv_w")
    b = ng.variable(ng.int32, shape=(args.Cout,), name="conv_b")
    y = ng.conv2d(x, w, strides=(1, 1, 1, 1), padding="SAME",
                  bias=b, act_func=ng.relu,
                  dtype=dt, sum_dtype=ng.int32, name="conv")

    # 2) valores reprodutiveis, sem saturacao ---------------------------------
    # |acc| <= 2*8*(K*Cin) + |bias| ; para K=16,Cin=8: <= 2048+64 << 32767
    rng = np.random.default_rng(args.seed)
    wv = rng.integers(-2, 3, size=w.shape, dtype=np.int64)
    bv = rng.integers(-64, 65, size=b.shape, dtype=np.int64)
    xv = rng.integers(-8, 9, size=x.shape, dtype=np.int64)
    w.set_value(wv)
    b.set_value(bv)

    # 3) RTL + enderecos ------------------------------------------------------
    targ = ng.to_veriloggen([y], args.topname, silent=True,
                            config={"maxi_datawidth": 128,
                                    "default_datawidth": 16})
    rtl_path = os.path.join(args.rtldir, f"{args.topname}.v")
    with open(rtl_path, "w") as f:
        f.write(targ.to_verilog())

    out_bytes = int(np.prod(y.shape)) * 2
    print(f"[ok] RTL: {rtl_path} ({sum(1 for _ in open(rtl_path))} linhas)")
    print(f"     enderecos DRAM: y(out)=0x{y.addr:x}  x(signal)=0x{x.addr:x}"
          f"  w=0x{w.addr:x}  b=0x{b.addr:x}")

    # 4) golden em software (mesma semantica de quantizacao do RTL) -----------
    golden = ng.eval([y], signal=xv)[0].reshape(-1).astype(np.int64)
    assert golden.size * 2 == out_bytes

    # 5) imagem de memoria + arquivos de simulacao ----------------------------
    param = ng.export_ndarray([y])            # blob com w e b ja nos offsets
    param_base = min(w.addr, b.addr)

    mem_end = max(param_base + len(param),
                  x.addr + xv.size * 2,
                  y.addr + out_bytes)
    mem_bytes = 1 << max(15, int(np.ceil(np.log2(mem_end))))  # >= 32 KiB

    mem = np.zeros(mem_bytes, dtype=np.uint8)
    mem[param_base:param_base + len(param)] = param
    xb = xv.astype("<i2").tobytes()
    mem[x.addr:x.addr + len(xb)] = np.frombuffer(xb, dtype=np.uint8)

    ram_init = os.path.join(args.outdir, "ram_init.hex")
    with open(ram_init, "w") as f:
        f.write("\n".join(to_hex_image(mem.tobytes())) + "\n")

    golden_path = os.path.join(args.outdir, "golden.hex")
    with open(golden_path, "w") as f:
        for v in golden:
            f.write(f"{int(v) & 0xFFFF:04x}\n")

    cfg_path = os.path.join(args.outdir, "sim_config.vh")
    with open(cfg_path, "w") as f:
        f.write(f"""// gerado por scripts/04_make_sim_data.py - nao editar a mao
// conv {args.Cin}->{args.Cout}, kernel {args.K}x1, H={args.H}, int16, seed={args.seed}
`define DUT_MODULE     {args.topname}
`define RAM_INIT_FILE  "data/ram_init.hex"
`define GOLDEN_FILE    "data/golden.hex"
`define OUT_ADDR       32'h{y.addr:08x}
`define OUT_WORDS16    {golden.size}
`define MEM_BYTES      {mem_bytes}
`define TIMEOUT_CYCLES {args.timeout}
""")

    print(f"[ok] imagem da RAM: {ram_init} ({mem_bytes} bytes)")
    print(f"[ok] golden:        {golden_path} ({golden.size} palavras int16)")
    print(f"[ok] config do tb:  {cfg_path}")
    print(f"     saida esperada em 0x{y.addr:x}..0x{y.addr + out_bytes - 1:x}"
          f" | golden: min={golden.min()} max={golden.max()}"
          f" nao-zero={int((golden != 0).sum())}/{golden.size}")


if __name__ == "__main__":
    main()
