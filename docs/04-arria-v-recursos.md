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

## 4.4 Implicações de arquitetura

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
