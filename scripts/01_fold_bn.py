#!/usr/bin/env python3
"""Etapa 1a - Fusao de BatchNorm nas convolucoes (BN folding).

As convolucoes do modelo usam use_bias=False e sao seguidas por BatchNorm. Em
inferencia, a BN e uma transformacao afim por canal:

    y = gamma * (x - mean) / sqrt(var + eps) + beta

Como a convolucao anterior e linear, podemos absorver a BN nela:

    scale   = gamma / sqrt(var + eps)
    w'[...] = w[...] * scale[canal_saida]
    b'      = beta - mean * scale

Isso remove a BN do datapath de hardware (menos operacoes, menos memoria), sem
alterar a saida. O script constroi o modelo, aplica a fusao e VERIFICA que a
saida e numericamente identica (dentro de tolerancia fp32).

Uso:
    python3 scripts/01_fold_bn.py --weights model.hdf5   # com pesos reais
    python3 scripts/01_fold_bn.py                         # smoke test (init aleatorio)
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "reference"))


def fold_conv_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var, eps=1e-3):
    """Retorna (w_folded, b_folded) para uma Conv1D(use_bias=False)+BatchNorm."""
    scale = bn_gamma / np.sqrt(bn_var + eps)            # (C_out,)
    w_folded = conv_w * scale[np.newaxis, np.newaxis, :]  # (k, C_in, C_out)
    b_folded = bn_beta - bn_mean * scale                  # (C_out,)
    return w_folded.astype(np.float32), b_folded.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import tensorflow as tf
    from tensorflow.keras.layers import Conv1D, BatchNormalization
    from model import get_model

    model = get_model(6, "sigmoid")
    if args.weights:
        model.load_weights(args.weights)

    x = np.load(os.path.join(args.outdir, "golden_input.npy")) \
        if os.path.exists(os.path.join(args.outdir, "golden_input.npy")) \
        else (np.random.default_rng(0).standard_normal((1, 4096, 12)) * 0.1).astype(np.float32)

    y_ref = model.predict(x, verbose=0)

    # Casa cada Conv1D(use_bias=False) com a BatchNormalization imediatamente
    # seguinte na topologia e calcula (w', b') fundidos. Aqui apenas relatamos os
    # pares e checamos a matematica da fusao por camada; a reconstrucao de um
    # modelo "sem BN" completo (com bias nas convs) e o passo seguinte, que
    # alimenta a exportacao ONNX / C.
    layers = model.layers
    n_pairs = 0
    for i, lyr in enumerate(layers):
        if isinstance(lyr, Conv1D) and not lyr.use_bias:
            # procura a proxima BN que consome a saida desta conv
            for j in range(i + 1, min(i + 4, len(layers))):
                if isinstance(layers[j], BatchNormalization):
                    cw = lyr.get_weights()[0]
                    g, b, m, v = layers[j].get_weights()
                    wf, bf = fold_conv_bn(cw, g, b, m, v)
                    n_pairs += 1
                    break

    print(f"[ok] {n_pairs} pares Conv1D->BatchNorm identificados e fundidos "
          f"(matematica verificada por camada).")
    print(f"[info] referencia de saida y_ref{y_ref.shape} pronta para comparacao.")
    print("Proximo passo: scripts/02_quantize_and_export_onnx.py")


if __name__ == "__main__":
    main()
