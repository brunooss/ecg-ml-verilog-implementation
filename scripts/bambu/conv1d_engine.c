/* Etapa 2 - Via B: engine de Conv1D reutilizavel em C para HLS (PandA/Bambu).
 *
 * Ideia central (ver docs/01 e 03): NAO instanciar uma camada por bloco de
 * hardware. Ha UM engine de Conv1D parametrizavel que processa cada camada em
 * sequencia, lendo os pesos da DRAM externa. Assim o custo de hardware independe
 * do numero de camadas - o que muda e so a tabela de configuracao por camada.
 *
 * Precisao: int16 (decidido). Acumulador em int32 para nao estourar.
 * BatchNorm ja fundida na convolucao (ver scripts/01_fold_bn.py): cada camada
 * tem pesos w' e um bias b' por canal de saida.
 *
 * Este arquivo e escrito em C limpo, sinteti­zavel pelo Bambu:
 *     bambu conv1d_engine.c --top-fname=conv1d_layer \
 *           --device-name=5AGXFB3H4F35C4 --clock-period=10 \
 *           --generate-tb=testbench.xml -v3
 * (device-name = a parte Arria V do seu board; ajuste conforme decisao aberta.)
 *
 * O RTL Verilog gerado sintetiza no Quartus para a Arria V (RTL generico,
 * sem back-end de fornecedor).
 */
#include <stdint.h>

typedef int16_t data_t;   /* ativacoes e pesos em int16 */
typedef int32_t acc_t;    /* acumulador */

/* Um passo de convolucao 1D "same padding" com stride, para UMA camada.
 *   in      : [n_in][c_in]                 ativacoes de entrada (em BRAM/on-chip)
 *   weights : [k][c_in][c_out]             pesos ja fundidos (streaming da DRAM)
 *   bias    : [c_out]                      bias por canal (BN fundida)
 *   out     : [n_out][c_out]               ativacoes de saida
 * Layout linearizado (row-major) para facilitar o acesso por HLS.
 */
void conv1d_layer(const data_t *in, const data_t *weights, const acc_t *bias,
                  data_t *out,
                  int n_in, int c_in, int c_out, int k, int stride,
                  int out_shift /* deslocamento p/ requantizar int32->int16 */)
{
    const int n_out = (n_in + stride - 1) / stride;   /* ceil, 'same' */
    const int pad = (k - 1) / 2;

    for (int o = 0; o < n_out; o++) {
        const int center = o * stride;
        for (int oc = 0; oc < c_out; oc++) {
            acc_t acc = bias[oc];
            for (int kk = 0; kk < k; kk++) {
                const int idx = center + kk - pad;
                if (idx < 0 || idx >= n_in) continue;   /* zero-pad */
                const data_t *in_row = in + (long)idx * c_in;
                const data_t *w_row  = weights + ((long)kk * c_in) * c_out + oc;
                for (int ic = 0; ic < c_in; ic++) {
                    acc += (acc_t)in_row[ic] * (acc_t)w_row[(long)ic * c_out];
                }
            }
            /* ReLU + requantizacao para int16 (saturada) */
            acc_t v = acc >> out_shift;
            if (v < 0) v = 0;                 /* ReLU (todas as camadas ocultas) */
            if (v > 32767) v = 32767;
            out[(long)o * c_out + oc] = (data_t)v;
        }
    }
}

/* max pooling 1D (skip connection com downsampling), 'same' */
void maxpool1d(const data_t *in, data_t *out,
               int n_in, int c, int pool, int stride)
{
    const int n_out = (n_in + stride - 1) / stride;
    for (int o = 0; o < n_out; o++) {
        for (int ch = 0; ch < c; ch++) {
            data_t m = -32768;
            for (int p = 0; p < pool; p++) {
                int idx = o * stride + p;
                if (idx < n_in) {
                    data_t x = in[(long)idx * c + ch];
                    if (x > m) m = x;
                }
            }
            out[(long)o * c + ch] = m;
        }
    }
}

/* soma residual elementwise com saturacao int16 */
void add_residual(const data_t *a, const data_t *b, data_t *out, long n)
{
    for (long i = 0; i < n; i++) {
        acc_t s = (acc_t)a[i] + (acc_t)b[i];
        if (s > 32767) s = 32767;
        if (s < -32768) s = -32768;
        out[i] = (data_t)s;
    }
}
