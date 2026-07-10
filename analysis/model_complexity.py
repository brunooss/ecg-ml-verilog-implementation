#!/usr/bin/env python3
"""Analytic layer-by-layer parameter and MAC count for the Ribeiro ECG ResNet.
No TensorFlow needed - reproduces the shapes defined in model.py."""

def conv1d(name, n_in_samples, c_in, c_out, k, stride, bias=False):
    n_out = -(-n_in_samples // stride)  # 'same' padding ceil div
    params = k * c_in * c_out + (c_out if bias else 0)
    macs = n_out * k * c_in * c_out
    return name, n_out, c_out, params, macs

rows = []
def add(r): rows.append(r)

# stem
add(conv1d("stem_conv", 4096, 12, 64, 16, 1)); n, c = 4096, 64
add(("stem_bn", n, c, 4*c, 0))

def resunit(prefix, n_in, c_in, n_out, c_out, k=16):
    ds = n_in // n_out
    out = []
    # skip: maxpool (no params) + optional 1x1 conv
    if c_in != c_out:
        out.append(conv1d(prefix+"_skip_1x1", n_in, c_in, c_out, 1, ds))
    # main conv1 (stride 1) then conv2 (stride ds)
    out.append(conv1d(prefix+"_conv1", n_in, c_in, c_out, k, 1))
    out.append((prefix+"_bn1", n_in, c_out, 4*c_out, 0))
    out.append(conv1d(prefix+"_conv2", n_in, c_out, c_out, k, ds))
    out.append((prefix+"_bn2", n_out, c_out, 4*c_out, 0))
    return out

for r in resunit("res1", 4096, 64, 1024, 128): add(r)
for r in resunit("res2", 1024, 128, 256, 196): add(r)
for r in resunit("res3", 256, 196, 64, 256): add(r)
for r in resunit("res4", 64, 256, 16, 320): add(r)

# flatten + dense
flat = 16 * 320
add(("dense", 1, 6, flat*6 + 6, flat*6))

tot_p = tot_m = 0
print(f"{'layer':<16}{'out_samples':>12}{'out_ch':>8}{'params':>14}{'MACs':>16}")
print("-"*66)
for name, ns, ch, p, m in rows:
    tot_p += p; tot_m += m
    print(f"{name:<16}{ns:>12}{ch:>8}{p:>14,}{m:>16,}")
print("-"*66)
print(f"{'TOTAL':<16}{'':>12}{'':>8}{tot_p:>14,}{tot_m:>16,}")
print(f"\nTotal parameters : {tot_p:,} (~{tot_p/1e6:.2f} M)")
print(f"Total MACs/infer : {tot_m:,} (~{tot_m/1e9:.2f} GMAC)  -> ~{2*tot_m/1e9:.2f} GFLOP")
print(f"Weights @ int8   : ~{tot_p/1e6:.2f} MB   @ int16: ~{2*tot_p/1e6:.2f} MB   @ fp32: ~{4*tot_p/1e6:.2f} MB")
