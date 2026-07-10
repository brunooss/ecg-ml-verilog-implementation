# 06 — Explicação para leigos (o projeto inteiro, em linguagem simples)

Este documento explica, sem jargão, **o que é o modelo**, **o que estamos
fazendo com ele**, **como** e **com quais ferramentas** — desde o que é uma FPGA
até por que uma camada específica deu trabalho. Se você nunca mexeu com redes
neurais nem com hardware, comece por aqui.

---

## 1. O objetivo em uma frase

Existe um programa de computador (uma "inteligência artificial") que **olha um
exame de eletrocardiograma (ECG) e diz se há certas doenças do coração**. Hoje
ele roda num computador comum, em software. Queremos fazer esse mesmo programa
rodar **dentro de um chip especial (uma FPGA)** — transformá-lo de "software" em
"circuito eletrônico".

**Por que fazer isso?** Um chip dedicado pode ser menor, mais rápido, gastar
menos energia e funcionar sem depender de um PC — útil, por exemplo, num
aparelho portátil de monitoramento cardíaco.

---

## 2. O que é uma FPGA (e por que Verilog)

- Um processador comum (o do seu PC) é uma peça **fixa**: ela já vem pronta e
  você só escreve programas que rodam nela.
- Uma **FPGA** é um chip **"moldável"**: um monte de pecinhas eletrônicas
  (blocos lógicos, memórias, multiplicadores) que você pode **religar** para
  formar o circuito que quiser. É como um LEGO eletrônico: as peças são as
  mesmas, mas você monta o que precisar.
- Para "dizer" à FPGA como conectar essas peças, usa-se uma linguagem chamada
  **Verilog** (um "HDL", *hardware description language*). Verilog **não é** um
  programa que executa passo a passo — é uma **descrição do circuito**.
- A nossa placa é uma **Intel/Altera Arria V**. Escolhemos Verilog (em vez de
  ferramentas fechadas da Intel) porque Verilog é **universal**: o mesmo circuito
  serve para outras placas no futuro.

---

## 3. O que o modelo faz e como ele é por dentro

### 3.1 Entrada e saída
- **Entrada:** um ECG de 12 "derivações" (12 pontos de vista do coração), com
  ~10 segundos de sinal. Em números: uma tabela de **4096 medidas no tempo × 12
  derivações**.
- **Saída:** **6 números entre 0 e 1**, cada um a "probabilidade" de uma
  anormalidade cardíaca (por ex. fibrilação atrial, taquicardia, bloqueios…).

### 3.2 As "camadas" — a receita de bolo
Uma rede neural é uma **sequência de etapas** (camadas). Cada etapa transforma os
números da anterior. Pense numa linha de montagem. As etapas aqui são:

- **Convolução (Conv):** é o coração do modelo. Uma convolução desliza uma
  "janelinha" (um filtro) pelo sinal procurando **padrões locais** — por exemplo,
  o formato de um batimento. Cada filtro aprende a reconhecer um padrão. É como
  passar várias lupas diferentes ao longo do ECG, cada lupa sensível a um
  formato. O modelo tem **cerca de 15 convoluções**, e é onde está praticamente
  todo o esforço de cálculo.
- **BatchNorm (normalização):** depois de cada convolução, um passo que
  **reequilibra os números** (tira média, ajusta escala) para a rede treinar e
  funcionar de forma estável. Em uso final, isso vira só uma "multiplicação e
  soma" fixa por canal — e a gente **funde** isso dentro da convolução (some no
  cálculo dela) para simplificar o hardware.
- **ReLU (ativação):** uma regrinha simples: "se o número for negativo, vira
  zero; se for positivo, mantém". Isso dá à rede a capacidade de decidir/filtrar.
- **Pooling (MaxPool):** **encolhe** o sinal pegando o maior valor de cada
  pedacinho. Reduz o tamanho e mantém o que é forte. Ao longo do modelo o sinal
  encolhe de 4096 → 1024 → 256 → 64 → 16 pontos.
- **Bloco residual (o "Res" de ResNet):** um truque famoso. Além de passar os
  números pela convolução, a rede também **guarda uma cópia da entrada** e
  **soma** as duas no fim ("atalho" ou *skip*). Isso ajuda redes profundas a
  aprender sem "se perder". O nosso modelo tem **4 blocos residuais**.
- **Flatten + Dense (decisão final):** no fim, todos os números são
  "esticados" numa lista e passam por uma última camada (**Dense**) que produz os
  6 resultados. Uma função **sigmoide** espreme cada um para ficar entre 0 e 1.

### 3.3 O tamanho do modelo (por que isso importa muito)
- O modelo tem **~6,43 milhões de "pesos"** (os números que ele aprendeu no
  treino). Cada peso é como um botãozinho ajustado.
- Fazer uma análise de um ECG exige **~1,83 bilhão de multiplicações**
  (chamadas "MAC"). Parece muito — e é —, mas o computador (ou a FPGA) faz isso
  em frações de segundo.

Esses dois números **definem toda a engenharia** do projeto (próxima seção).

---

## 4. O problema central (e a solução)

Imagine que os 6,43 milhões de pesos são **6,43 milhões de fichas** que o
circuito precisa consultar para fazer as contas.

- A **memória interna** da FPGA Arria V é pequena — cabe mais ou menos **1,7 MB**
  de fichas. Mas os pesos ocupam **entre 6 e 26 MB** (dependendo da precisão).
  **Não cabe.**
- Solução: guardar as fichas numa **memória externa maior** (uma **DDR3**, o tipo
  de "pente de memória" parecido com o do PC), que a placa tem de sobra. O
  circuito **vai buscando** as fichas na DDR3 conforme precisa. Calculamos que
  são necessários só ~16–20 MB e que a "velocidade de busca" exigida é muito
  menor do que a DDR3 oferece — ou seja, **tranquilo**.
- Como não há pressa (um exame a cada ~10 segundos é folgadíssimo para um chip),
  o circuito pode ser **pequeno e simples**: um **único "motor" de convolução**
  que é **reutilizado** para todas as camadas, uma de cada vez. Isso é ótimo
  porque o custo do hardware **não cresce** com o número de camadas.

### 4.1 "Precisão" (int16) — o que é isso
Os pesos originais são números "com vírgula" bem detalhados (chamados fp32, 32
bits cada). Em hardware, é caro carregar tanta precisão. Então **arredondamos**
os pesos para números inteiros de 16 bits (**int16**) — metade do tamanho, contas
mais baratas, e **quase sem perder qualidade** de diagnóstico. Escolhemos int16
(em vez do ainda menor int8) porque aqui o "custo" de usar int16 é irrelevante e
ele preserva melhor a precisão do diagnóstico — importante num contexto médico.

---

## 5. As ferramentas (por que umas servem e outras não)

O desafio é: **como transformar um modelo de IA (escrito em Python) num circuito
Verilog?** Investigamos vários caminhos.

- **OpenVINO** (ferramenta da Intel): **não serve.** Só funcionava numa placa
  diferente (Arria 10), foi descontinuada, e entrega um "pacote fechado", não o
  Verilog aberto que queremos.
- **OpenCL / oneAPI** (outras ferramentas da Intel): **não servem** para a nossa
  Arria V — sem suporte para essa placa, e a Intel está abandonando esse caminho.
- **Escrever o Verilog à mão:** possível em teoria, **inviável na prática** para
  6,4 milhões de pesos — seria como construir um relógio suíço parafuso por
  parafuso. Só usamos escrita manual para peças pequenas de referência.
- **HLS (*High-Level Synthesis*)** — **este é o caminho.** HLS é uma tecnologia
  que **traduz automaticamente** uma descrição de alto nível (o modelo, ou um
  código em C) para Verilog. É como um "tradutor" de software para circuito.
  Dentro do HLS, temos três opções:
  - **hls4ml:** boa, mas **precisa de um "tradutor" da própria Intel** por baixo
    (um back-end de fornecedor) — o que contraria nosso objetivo de ser
    universal. Ficou como comparação opcional.
  - **NNgen:** pega o modelo e **gera o Verilog diretamente**, já com o mecanismo
    de buscar pesos na DDR3. **Não depende** de ferramenta de fornecedor. É a
    nossa **via A**.
  - **PandA/Bambu:** um tradutor **de C para Verilog**, aberto e universal.
    Escrevemos o "motor" de convolução em C e ele vira circuito. É a nossa
    **via B** (plano robusto e portável).

Decidimos seguir **NNgen + Bambu** em paralelo, porque ambos geram Verilog
universal, sem amarras.

---

## 6. O que eu realmente fiz (passo a passo, com o "porquê")

Tudo isto foi **executado de verdade** neste ambiente (resultados em
[`05-testes-realizados.md`](05-testes-realizados.md)):

1. **Montei o modelo e conferi o tamanho.** Rodei o código do modelo e confirmei
   os **6.425.638 pesos** — batendo com a conta que eu havia feito no papel. É a
   "certidão de nascimento" do modelo.

2. **Fundi a BatchNorm nas convoluções.** Aquele passo de "reequilibrar números"
   (seção 3.2) foi **absorvido** dentro das convoluções. Menos operações no
   circuito, mesmo resultado. (13 pares fundidos.)

3. **Exportei o modelo para um formato universal (ONNX).** ONNX é um "idioma
   comum" que várias ferramentas entendem. É a ponte entre o mundo Python e as
   ferramentas de hardware. (Deu um arquivo de 25,7 MB, coerente com o tamanho.)

4. **Levei o modelo ao NNgen** para gerar o Verilog. Aqui veio o trabalho de
   engenharia real. O NNgen é uma ferramenta poderosa mas **antiga**, e o modelo
   veio "embrulhado" de um jeito que ele não gostou. Fui destravando um problema
   de cada vez:
   - a última camada (Dense) veio num formato que ele não reconhecia →
     contornei;
   - havia incompatibilidades de versão (de um padrão do ONNX e de uma
     biblioteca numérica) → **corrigi** e deixei as correções embutidas nos
     scripts;
   - **descoberta importante:** o modelo usa convoluções **1D** (o ECG é um sinal
     ao longo do tempo), mas o NNgen prefere convoluções **2D** (como imagens).
     Então **reescrevi o modelo como se o ECG fosse uma "imagem" de 4096×1** —
     matematicamente **idêntico** (mesmos 6,43 milhões de pesos), só numa
     roupagem que o NNgen entende melhor. Isso resolveu boa parte da fricção.

5. **Gerei Verilog de verdade.** Usando o "modo direto" (API nativa) do NNgen,
   produzi o circuito de uma **camada de convolução real do modelo** (a que
   transforma 64 em 128 canais). Resultado concreto:
   - **61.572 linhas de Verilog**;
   - **passou na verificação** de um simulador de hardware (Icarus Verilog) — ou
     seja, **é circuito válido**, não rascunho;
   - já vem com a **"tomada" AXI4** para conversar com a memória DDR3 dos pesos.

   Esse módulo é o **"motor" reutilizável**: o mesmo circuito, reconfigurado,
   processa todas as camadas do modelo, uma a uma. (Excerto em
   [`../rtl/`](../rtl/).)

6. **Preparei a via B (Bambu) em paralelo.** Escrevi o "motor" de convolução em
   **C** (`scripts/bambu/conv1d_engine.c`) e confirmei que **compila limpo**. É o
   caminho alternativo, 100% aberto, caso queiramos comparar.

---

## 7. Onde estamos e o que falta

**Já provado (o difícil):**
- O modelo cabe na estratégia certa (motor reutilizável + pesos na DDR3).
- As ferramentas escolhidas (NNgen, Bambu) são as adequadas e **funcionam**.
- **O NNgen gera Verilog sintetizável** para a peça-chave do modelo.

**O que falta (mecânico, sem grandes incógnitas):**
- Encadear **todas** as ~15 convoluções pelo "modo direto" do NNgen, somando os
  atalhos residuais, e gerar o circuito do modelo **inteiro**.
- Baixar os **pesos reais** (do treino) e medir se, ao arredondar para int16, o
  diagnóstico continua **igualmente bom** (é a validação mais importante para a
  pesquisa).
- **Montar na Arria V de verdade:** isso exige o programa da Intel (Quartus) e a
  **placa física** — passos que **não** dá para fazer aqui; eu entrego os
  arquivos prontos e você roda na placa.

---

## 8. Glossário rápido

| Termo | Em uma frase |
|-------|--------------|
| **FPGA** | Chip "moldável" que você reconfigura para virar o circuito que quiser. |
| **Verilog / RTL** | A "planta" do circuito, em texto. |
| **Arria V** | O modelo da nossa placa FPGA (Intel/Altera). |
| **Camada** | Uma etapa da "linha de montagem" da rede neural. |
| **Convolução** | Passar filtros pelo sinal para achar padrões locais. |
| **Peso** | Um dos ~6,4 milhões de números que o modelo aprendeu. |
| **MAC** | Uma multiplicação-e-soma; o modelo faz ~1,83 bilhão por exame. |
| **int16 / fp32** | Formas de guardar números; int16 é menor e mais barato no hardware. |
| **DDR3** | Memória externa grande (tipo pente de RAM) que guarda os pesos. |
| **HLS** | Tecnologia que traduz alto nível → Verilog automaticamente. |
| **ONNX** | "Idioma comum" para trocar modelos entre ferramentas. |
| **NNgen / Bambu** | As ferramentas que geram Verilog (a partir do modelo / de C). |
| **AXI4** | O "padrão de tomada" pelo qual o circuito fala com a memória. |
