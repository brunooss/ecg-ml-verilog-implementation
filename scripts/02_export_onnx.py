#!/usr/bin/env python3
"""Etapa 1b - Exporta o modelo para ONNX (entrada do NNgen) e dumpa os pesos.

Gera:
    outputs/ecg_model.onnx   -> grafo ONNX fp32 (NNgen faz a quantizacao int16)
    outputs/weights.npz      -> pesos crus por camada (para o caminho C/Bambu)

A quantizacao int16 em si e feita:
  - no NNgen, a partir deste ONNX (ver scripts/03a_nngen_convert.py); ou
  - no gerador de C (ver scripts/bambu/), escalando pesos/ativacoes para int16.

Uso:
    python3 scripts/02_export_onnx.py --weights model.hdf5
    python3 scripts/02_export_onnx.py            # smoke test sem pesos
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "reference"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import tensorflow as tf
    import tf2onnx
    from model import get_model

    model = get_model(6, "sigmoid")
    if args.weights:
        model.load_weights(args.weights)
        print(f"[ok] pesos reais carregados de {args.weights}")
    else:
        print("[aviso] sem pesos: ONNX exportado com init aleatorio (smoke test).")

    # dump de pesos crus por camada (para o caminho C/Bambu)
    wdict = {}
    for lyr in model.layers:
        w = lyr.get_weights()
        for k, arr in enumerate(w):
            wdict[f"{lyr.name}__{k}"] = arr.astype(np.float32)
    np.savez(os.path.join(args.outdir, "weights.npz"), **wdict)
    print(f"[ok] {len(wdict)} tensores de peso salvos em weights.npz")

    # exporta ONNX
    spec = (tf.TensorSpec((1, 4096, 12), tf.float32, name="signal"),)
    onnx_path = os.path.join(args.outdir, "ecg_model.onnx")
    tf2onnx.convert.from_keras(model, input_signature=spec,
                               opset=args.opset, output_path=onnx_path)
    print(f"[ok] ONNX salvo em {onnx_path}")


if __name__ == "__main__":
    main()
