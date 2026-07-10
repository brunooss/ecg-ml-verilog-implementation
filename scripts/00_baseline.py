#!/usr/bin/env python3
"""Etapa 0 - Baseline de software (golden reference).

Constroi o modelo de Ribeiro et al., (opcionalmente) carrega os pesos treinados
e gera um par (entrada, saida) de referencia em fp32 que servira para validar,
bit-a-bit, o hardware gerado nas etapas seguintes.

Uso:
    python3 scripts/00_baseline.py --weights caminho/para/model.hdf5
    python3 scripts/00_baseline.py            # sem pesos: usa init aleatorio (so p/ smoke test)

Saidas (em outputs/):
    golden_input.npy   -> (1, 4096, 12) float32
    golden_output.npy  -> (1, 6)        float32  (probabilidades pos-sigmoide)
    model_summary.txt  -> keras summary (confirma ~6,43 M parametros)
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "reference"))  # importa model.py original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="model.hdf5 do Zenodo (opcional)")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from model import get_model  # reference/model.py

    model = get_model(n_classes=6, last_layer="sigmoid")

    # Confirma a contagem de parametros (deve bater com ~6,43 M da nossa analise)
    summ = []
    model.summary(print_fn=lambda l: summ.append(l))
    with open(os.path.join(args.outdir, "model_summary.txt"), "w") as f:
        f.write("\n".join(summ))
    print("\n".join(summ[-4:]))

    if args.weights:
        model.load_weights(args.weights)
        print(f"[ok] pesos carregados de {args.weights}")
    else:
        print("[aviso] SEM pesos: saida nao tem significado clinico (apenas smoke test).")

    rng = np.random.default_rng(args.seed)
    # Entrada realista: ~10 s de ECG a 400 Hz, escala 1e-4 V. Aqui um ruido
    # gaussiano pequeno so para ter um vetor deterministico de referencia.
    x = (rng.standard_normal((1, 4096, 12)) * 0.1).astype(np.float32)
    y = model.predict(x, verbose=0).astype(np.float32)

    np.save(os.path.join(args.outdir, "golden_input.npy"), x)
    np.save(os.path.join(args.outdir, "golden_output.npy"), y)
    print(f"[ok] golden_input {x.shape} e golden_output {y.shape} salvos em {args.outdir}")
    print("saida:", np.array2string(y, precision=5))


if __name__ == "__main__":
    main()
