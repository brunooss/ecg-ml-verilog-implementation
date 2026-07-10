# 04 — Recursos da Arria V e orçamento de hardware

A **Arria V** (nó de 28 nm, ALM com LUT fraturável de 8 entradas + 4 registradores)
tem dois tamanhos de RAM interna: blocos **M10K** (10 Kb cada) para arrays
maiores e **MLAB** (memória distribuída). O DSP é do tipo **variable-precision**,
configurável em 9×9, 18×18, 27×27 e 36×36 — ou seja, um bloco DSP pode empacotar
**vários MACs de baixa precisão** (bom para int8).

## 4.1 Recursos de duas variantes GX de referência

| Recurso | Arria V GX **A7** | Arria V GX **B3** |
|---------|------------------:|------------------:|
| Elementos lógicos (LEs) | ~242 K | ~362 K |
| ALMs | 91.680 | 136.880 |
| Registradores | 366.720 | 547.520 |
| Blocos **M10K** | 1.366 (**13.660 Kb ≈ 1,67 MB**) | 1.726 (**17.260 Kb ≈ 2,11 MB**) |
| MLAB | 1.448 (2.317 Kb) | 2.098 (3.357 Kb) |
| Blocos DSP (variable-precision) | **800** | 1.045 |

(Valores do *Arria V Device Overview*, AV-51001. A parte exata da sua placa
ajusta esses números — confirmar; ver decisão 2 em
[`03-recomendacao-e-pipeline.md`](03-recomendacao-e-pipeline.md).)

## 4.2 Confronto com as necessidades do modelo

| Necessidade do modelo | Valor | Cabe on-chip? |
|---|---:|---|
| Pesos @ fp32 | 25,7 MB | ❌ (15× a M10K do A7) |
| Pesos @ int16 | 12,85 MB | ❌ |
| Pesos @ int8 | 6,43 MB | ❌ (~4× a M10K do A7) |
| Maior mapa de ativação (stem, 4096×64 @ int8) | 256 KB | ✅ (com folga) |
| Maior *tensor* de pesos de 1 camada (res4_conv2, 320×320×16 @ int8) | ~1,64 MB | ⚠️ perto do limite — usar *tiling* |

**Leitura:** nenhuma configuração de pesos cabe inteira no chip → **DDR3 externa
é obrigatória**. As **ativações** cabem on-chip, então o dataflow natural é:
manter ativações da camada em M10K, **fazer streaming dos pesos** da DDR3 pelo
engine, e escrever a saída de volta em M10K (ou DDR3 se necessário).

## 4.3 Orçamento de computação (por que a latência não é problema)

- Total: **1,83 GMAC** por inferência.
- A Arria V GX A7 tem **800 DSPs**; em int8, cada DSP variable-precision faz
  ~2–3 MACs, então há teto de **~1.600–2.400 MAC/ciclo** se totalmente usado.
- Mesmo com um engine modesto de, digamos, **128 MACs** a **100 MHz** =
  12,8 GMAC/s → 1,83 GMAC ≈ **~143 ms** por inferência. Com uma janela de ECG a
  cada ~10 s, isso é **~70× mais rápido que o necessário**.
- Ou seja: dá para usar **poucos DSPs**, priorizar simplicidade/área e ainda
  sobrar folga enorme. O projeto pode ser **serial e pequeno**.

## 4.4 Estimativa de uso da DDR3 (precisão escolhida: int16)

Com a decisão de usar **int16** (ver [`03-recomendacao-e-pipeline.md`](03-recomendacao-e-pipeline.md#justificativa-da-precisão-int16)),
segue o dimensionamento do que precisa morar na DDR3 e da banda exigida.

### Capacidade

| O que | Tamanho @ int16 | Observação |
|-------|----------------:|------------|
| **Pesos do modelo** | **~12,85 MB** | 6,43 M params × 2 B; principal ocupante |
| Bias por canal (BN fundida) | ~5 KB | ~2,4 k canais somados × 2 B |
| Buffer de entrada (1 ECG) | ~98 KB | 4096×12 × 2 B |
| *Spill* de ativações (se necessário) | ~0,5–2 MB | maior mapa 4096×64 = 512 KB; margem p/ *double buffer* |
| **Total prático** | **~16–20 MB** (com alinhamento/folga) | |

**Conclusão:** ~**16–20 MB** bastam. Qualquer DDR3 de board Arria V (tipicamente
256 MB a 1 GB) é **enorme** para essa necessidade — confirma seu "sobrando". Não
há restrição de capacidade.

### Banda

Num esquema layer-by-layer, cada peso é lido da DDR3 **uma vez por inferência**
(as ativações da camada ficam on-chip). Logo o tráfego dominante é ~12,85 MB de
pesos por inferência.

| Cenário | Banda exigida |
|---------|--------------:|
| 1 inferência a cada 10 s (janela de ECG) | **~1,3 MB/s** |
| 1 inferência a cada ~140 ms (compute-bound, ~128 MACs @100 MHz) | **~90 MB/s** |

Uma DDR3 modesta na Arria V (ex.: interface de 16–32 bits a DDR3-800/1066)
entrega **~1,6–4 GB/s**. Ou seja, a banda exigida fica **~20× a ~3000× abaixo**
do disponível — **banda não é gargalo** em nenhum cenário realista. Isso permite
um controlador DDR3 simples com acesso essencialmente sequencial (*burst*
amigável, pré-busca do próximo *tile* de pesos durante o cálculo do atual).

## 4.5 Implicações de arquitetura

1. **Um engine de Conv1D reutilizável**, iterando camada a camada (não um bloco
   por camada). Custo de hardware independente do nº de camadas.
2. **Controlador DDR3** para pesos (a Arria V tem *hard memory controllers* para
   DDR3; confirmar no board). Padrão de acesso sequencial simples — banda não é
   crítica dada a latência folgada.
3. **Buffers em M10K** para ativações da camada atual + *tile* de pesos.
4. **Quantização int8** como alvo primário (menor pegada de DRAM e melhor uso do
   DSP), com **int16** como rede de segurança se a acurácia cair.
5. **BatchNorm fundida** e **sigmoide** só na saída (LUT pequena ou aplicada em
   software no host).
