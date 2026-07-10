# rtl/ — RTL Verilog gerado

## `ecg_conv_layer_interface.v`
**Excerto** (interface do módulo) do Verilog gerado pelo NNgen (API nativa) para
uma camada de convolução do modelo: **Conv 64→128, kernel 16×1, int16, padding
SAME**, com interface **AXI4 mestre** para a DDR3.

- RTL completo: **61.572 linhas** (2,8 MB), **parse OK no Icarus Verilog**
  (`iverilog -t null -Wall` → exit 0).
- O `.v` completo não é versionado (fica em `outputs/`, gitignored). **Gere-o**
  com:
  ```bash
  python3 scripts/03b_nngen_native_layer.py
  ```

## Por que a API nativa (e não o import ONNX direto)
O import do modelo completo via ONNX (tf2onnx → NNgen) esbarra em lacunas do
front-end do NNgen 1.3.4 — em particular o `padding='SAME'` de convoluções com
*stride* não reproduz as formas do Keras, fazendo o `Add` residual divergir (ver
[`../docs/05-testes-realizados.md`](../docs/05-testes-realizados.md) §4). A **API
nativa** contorna isso: definimos cada operação com *stride*/*padding* explícitos.
Esta camada gerada é o **bloco reutilizável** do acelerador *layer-by-layer* — o
mesmo módulo, reconfigurado por camada, cobre todo o backbone.

## Próximo passo
Encadear as ~15 convoluções do modelo pela API nativa (cada uma com suas
dimensões e *stride*), somar os *skips* e gerar o RTL do backbone completo, depois
simular contra o *golden reference* (`outputs/golden_output.npy`).
