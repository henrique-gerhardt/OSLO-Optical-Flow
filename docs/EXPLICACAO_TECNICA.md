# OSLO-RAFT explicado — o modelo, os experimentos e o que os resultados significam

<!-- SCOPE-BANNER -->
> **STATUS 2026-08-14 — read `docs/plans/LITERATURE_SCOPE.md` before quoting anything here.**
> Two independent corrections apply across this project's documents. (1) The FLOW360
> forward-flow convention was inverted, so every FLOW360 result recorded before
> 2026-08-04 is void. (2) A literature check on 2026-08-14 found that several things
> treated here as ours already exist in print: the geodesic (SEPE) metric, polar/equatorial
> stratification, rotation-robustness evaluation in panoramic vision, and matched-backbone
> comparison across panoramic representations. `LITERATURE_SCOPE.md` is the register of what
> we may and may not claim.
>
> **This file specifically:** READER-FACING AND OUT OF DATE. FLOW360 results are VOID and the uniformity claim has been demoted to one dataset plus a displacement threshold. Do not hand this to anyone before it is rewritten.


Documento de leitura para alguém com formação técnica (programação, ML básico) mas sem
familiaridade com fluxo óptico esférico, grades HEALPix ou a literatura de vídeo 360°.
Todo jargão é definido na primeira aparição. Pressupõe-se conhecido apenas o que é fluxo
óptico; o resto é construído do zero.

Fontes: `docs/OSLO_RAFT_DOCS.md`, `docs/THESIS_CH4_DRAFT.md`,
`docs/THESIS_REGIME_ARGUMENT.md`, `docs/plans/UNIVERSALITY_TABLE.md`,
`docs/plans/DECISION_GATE.md`, `docs/plans/ROADMAP_SEMINARIO.md` e o relatório de
andamento (`../OSLO-Optical-Flow-progress-report/src/main.tex`).

---

## 0. Dicionário mínimo

| Termo | O que é, sem jargão |
| --- | --- |
| **ERP** (equirretangular) | O formato padrão de vídeo 360°: a esfera "desenrolada" num retângulo, latitude no eixo Y, longitude no eixo X. É o mapa-múndi de Mercator do vídeo. |
| **Distorção polar** | No ERP, uma linha de pixels perto do polo cobre um pedaço minúsculo da esfera, mas ocupa a largura inteira da imagem. O polo é replicado em centenas de colunas. Uma janela de convolução 3×3 significa coisas fisicamente diferentes no equador e no polo. |
| **HEALPix** | Uma forma de picar a esfera em células de **área exatamente igual** (usada em cosmologia para mapas do céu). Cada célula vira um "nó". Não há retângulo, não há latitude privilegiada. |
| **Nó** | Um ponto de amostragem sobre a esfera. O análogo do pixel, mas sem grade retangular: os vizinhos são dados por um grafo, não por índices `i±1, j±1`. |
| **Resolução `r`** | Nível da hierarquia HEALPix. Cada nível divide cada célula em 4. `r4` = 3.072 nós, `r6` = 49.152, `r7` = 196.608. Espaçamento típico entre vizinhos: 3,67° / 0,92° / 0,46°. |
| **Plano tangente** | O plano que encosta na esfera num ponto. Um vetor de movimento sobre a esfera é representado nesse plano local, com duas componentes: *leste* e *norte*. |
| **exp / log map** | As duas conversões entre "vetor no plano tangente" e "ponto de chegada sobre a esfera". `exp`: dado o ponto de partida e o vetor, onde chega. `log`: dado partida e chegada, qual o vetor. É o análogo esférico de somar/subtrair coordenadas. |
| **Erro geodésico** | Distância angular (em graus de arco) entre o ponto onde o modelo diz que o conteúdo foi parar e onde ele realmente foi. Não depende de latitude — é a métrica "justa" na esfera. |
| **EPE** | *End-point error*, a métrica que a literatura publica: erro em **pixels ERP**. Infla artificialmente perto dos polos (um mesmo erro angular custa mais pixels lá). |
| **Zero-flow** | O preditor trivial: "nada se moveu, em lugar nenhum". É a linha de base contra a qual tudo aqui é medido. |
| **Ativos (`active>θ`)** | O subconjunto de nós cujo movimento verdadeiro passa de θ graus. É onde há de fato o que estimar. |
| **RAFT** | A arquitetura dominante de fluxo óptico em imagens de perspectiva (2020): correlação de todos-contra-todos em resolução reduzida + refinamento iterativo por uma GRU. O OSLO-RAFT é a recriação disso sobre a esfera. |
| **GRU / ConvGRU** | Unidade recorrente com portas: um bloco que mantém um "estado" e o atualiza a cada iteração, decidindo o quanto lembrar e o quanto sobrescrever. "Conv" = a atualização olha também os vizinhos espaciais. |

---

## 1. Por que uma rede esférica, e não mais uma adaptação do ERP

A literatura de fluxo 360° tem quatro linhas, e todas fazem a mesma coisa por baixo:
**estimam sobre um raster de pixels obtido projetando a esfera.** Elas diferem só em como
compensam a distorção — deformando o kernel da convolução, reprojetando em faces de
cubo/icosaedro, aumentando os dados com distorções sintéticas, ou acrescentando um segundo
ramo em vista ortogonal para consertar os polos.

O espaço complementar estava vazio: **e se a grade de estimação for esférica desde o
início?** Sem raster, não há latitude privilegiada, não há costura, e a convolução tem o
mesmo significado métrico em qualquer lugar da esfera. Essa é a hipótese que o OSLO-RAFT
testa.

A premissa geométrica declarada no início do projeto era mais forte: *amostrar em células
de área igual (HEALPix) é o que compra precisão nos polos*. Guarde essa frase — a Seção 6.6
mostra o experimento controlado que a derrubou, e o que sobreviveu no lugar dela.

---

## 2. A arquitetura, peça por peça

### 2.1 A ideia estrutural: três grades, não uma

O primeiro insight do projeto (e o que separa a versão que funciona da que não funcionava)
é que **"onde eu olho a imagem", "onde eu estimo o movimento" e "onde eu meço o erro" são
três coisas distintas**, e amarrá-las mata o modelo.

| Papel | Resolução | Nós | Espaçamento | Carrega |
| --- | --- | --- | --- | --- |
| **Retina** (entrada) | r7 | 196.608 | 0,458° | os dois quadros; os encoders rodam aqui |
| **Estimação** (saída) | r4 | 3.072 | 3,665° | correlação, estado da GRU, campo de fluxo |
| **Supervisão** (avaliação) | r6 | 49.152 | 0,916° | alvos da perda, métricas, saída final |

Por que assim: a RAFT original também estima em 1/8 da resolução, mas **vê** pixels em
resolução plena. A primeira versão do OSLO-RAFT amostrava a imagem *na própria grade de
estimação* (r4, ~3,7° entre nós) — ou seja, jogava fora todo o detalhe fino da imagem antes
de tentar casar detalhe fino. Foi uma das três falhas de projeto que a Seção 6.2 descreve.

A restrição é `estimação < supervisão ≤ retina`. Para referência: as fontes ERP têm
≈ 0,35°/pixel, então r7 já amostra um pouco acima do raster de origem — r8 seria desperdício.

### 2.2 Como se faz "convolução" quando não existe grade retangular

Numa imagem, convolução é: para cada pixel, pegue os 8 vizinhos em posições fixas
(nordeste, norte, ...), multiplique cada um pelo seu peso e some. A posição relativa é a
mesma no mundo inteiro, e é isso que dá sentido a "o peso do vizinho de cima".

Na esfera não há "vizinho de cima" com índice fixo. A solução herdada do projeto OSLO é a
**SDPAConv** (*Sphere Directional and Position-Aware convolution*): para cada nó existe uma
lista pré-computada de vizinhos, cada um com um **peso de interpolação** que codifica sua
direção e distância angular relativa ao nó central. A convolução é
`saída = Σ_k peso_k · (vizinho_k · W_k)`, com um banco de pesos `W` por posição direcional.
Na prática: uma convolução sobre um grafo, mas onde a geometria da esfera entra nos pesos,
não é ignorada. É pré-computada uma vez por grade e reusada.

### 2.3 Como se representa "movimento" na esfera

Duas representações, e a conversão entre elas importa:

- **Vetor tangente** `(leste, norte)` no plano tangente do nó — é o que a rede prevê;
- **Ponto de chegada** `e ∈ S²` — para onde o conteúdo daquele nó foi parar.

`e = exp(p, f)` e `f = log(p, e)`. A **perda é calculada no ponto de chegada**, não no vetor
tangente, justamente para que erros de ida-e-volta entre as duas formas não entrem no
gradiente.

A distância entre dois pontos unitários tem uma armadilha numérica que custou uma
reemissão de tabela inteira: a forma ingênua `acos(a·b)` é estacionária quando `a·b → 1`, o
que em float32 cria um **piso de erro de 0,028°** — abaixo disso ela não consegue medir.
Trocou-se pela forma **haversine**, `d = 2·arcsin(‖a−b‖/2)`, exata em todo o intervalo.
Isso não é cosmético: 42,8% dos nós do FLOW360 se movem 0,0103°, muito abaixo daquele piso,
então o `acos` estava **inflando a linha de base de fluxo nulo** e portanto favorecendo
todo método medido contra ela. Ver Seção 6.7.

### 2.4 O caminho dos dados

```
Quadro 1 (ERP) ─┐                    ┌─ encoder de características ─┐
                ├─ amostra na retina ┤                              ├→ CORRELAÇÃO (r4)
Quadro 2 (ERP) ─┘        (r7)        └─ encoder de características ─┘        │
                                                                             ↓
                          encoder de contexto (só quadro 1) ────────────→ ConvGRU (r4) ×8
                                                                             ↓
                                                            campo de fluxo tangente (r4)
                                                                             ↓
                                                       upsampler convexo → r6 → perda
                                        └──────── consulta de novo a correlação a cada iteração
```

**(a) Encoder de características (siamês).** Blocos residuais de SDPAConv que descem da
retina r7 até a grade de estimação r4, agrupando 4 filhos em 1 pai a cada nível
(32 → 64 → 96 canais, GroupNorm). "Siamês" = os mesmos pesos processam os dois quadros, o
que é obrigatório: descritores de coisas iguais precisam sair iguais para poderem ser
comparados. A saída é um vetor descritor por nó — a "assinatura visual" daquele pedacinho
da esfera.

**(b) Encoder de contexto.** Roda **só no quadro 1**, com pesos próprios, e produz duas
coisas: o estado oculto inicial da GRU (via `tanh`) e um vetor de contexto injetado em toda
iteração (via `relu`). Ele carrega o que a rede sabe sobre a *cena* (bordas, textura,
estrutura) independentemente de correspondência. É também a razão de um resultado incômodo
do Ato I: se o modelo consegue ir bem só com esse caminho, ele nunca precisou olhar o
quadro 2.

**(c) Volume de correlação.** O produto interno normalizado (cosseno) entre o descritor do
nó `i` no quadro 1 e o descritor do nó `j` no quadro 2, para todos os pares. Valor alto =
"esses dois pedaços da esfera se parecem". Fluxo óptico é, no fundo, achar o `j` que casa
com cada `i`; a correlação é o mapa dessa evidência. Existe também uma **pirâmide de
correlação**: agregando o eixo do quadro 2 (r4→r3→r2→r1) obtém-se busca grossa e busca
fina simultaneamente — o análogo de procurar primeiro no bairro e depois na rua.

**(d) A consulta esférica (*lookup*) — o coração geométrico.** Na RAFT plana, a cada
iteração se lê o volume de correlação numa janela ao redor da posição atual estimada. Na
esfera:

1. do fluxo atual `f_i`, computa-se o ponto de chegada `e_i = exp(p_i, f_i)`;
2. ao redor de `e_i` monta-se um **estêncil de 17 pontos**: o centro mais dois anéis
   geodésicos concêntricos de 8 pontos cada, com raio proporcional ao espaçamento do nível
   da pirâmide;
3. as características do quadro 2 são **interpoladas** nessas direções contínuas (pesos por
   distância inversa sobre os 3 nós mais próximos) e correlacionadas com `f1[i]`.

O detalhe que decide tudo: a primeira versão fazia `ang2pix` — "arredonde `e_i` para o nó
mais próximo". Isso torna a resposta da correlação **constante por pedaços** no fluxo: se o
fluxo muda menos que meio nó, o resultado da consulta é literalmente idêntico, então o
gradiente que chega ali é **zero quase em todo lugar**. Como o movimento típico do dataset
é 0,03 de um nó, a correlação era mecanicamente incapaz de aprender qualquer coisa. A versão
interpolada responde suavemente. Testes unitários confirmam: sob fluxo de 0,3 nó, a consulta
antiga é bit a bit idêntica e a nova varia.
(Por linearidade, interpolar descritores e depois fazer o produto interno é igual a
interpolar o próprio volume de correlação — o mesmo truque da RAFT, feito "preguiçosamente"
para nunca materializar a matriz N×N, que a r6 custaria 9,66 GB.)

**(e) ConvGRU sobre o grafo.** O refinador iterativo. Recebe (característica de correlação +
fluxo tangente atual + contexto), atualiza um estado oculto agregando os 8 vizinhos de cada
nó, e emite um **incremento** de fluxo `Δ(leste, norte)`. 8 iterações no treino, 12 na
avaliação. Cada iteração re-consulta a correlação na nova posição — é isso que permite
refinar sem construir pirâmides de *warping* explícitas.

Detalhe deliberado: a cabeça que emite `Δ` é **inicializada em zero**. Consequência: no
passo zero o fluxo é exatamente 0 em todo nó, e como `upsample_convexo(0) = 0`, a predição
inicial é *exatamente* o preditor trivial. O modelo começa de onde a linha de base está e
tem que ganhar cada grau a partir dali. Isso é verificado em todo teste de fumaça.

**(f) Upsampler convexo.** O fluxo vive em r4 (3.072 nós), mas a perda e as métricas vivem
em r6 (49.152). Cada nó pai prevê, via softmax, pesos sobre sua vizinhança de 1 salto para
cada um dos seus 16 descendentes r6 — uma interpolação **aprendida**, não bilinear fixa. A
combinação é feita por **transporte paralelo** dos vetores tangentes (não se pode
simplesmente somar vetores de planos tangentes diferentes; e mediar pontos de chegada
quebraria o contrato de partida em zero da alínea anterior).

Esse componente acabou sendo mais importante do que parecia: trocá-lo por "copie o valor do
nó central" custa **7,5% de erro nos polos** contra 2,5% no equador (FlowScape), e 5,4%
contra 1,6% no FLOW360. Ou seja, boa parte do ganho polar do modelo vem dessa suavização
aprendida — que é **arquitetura**, não geometria da grade.

**(g) Perda.** Soma geodésica sequencial no estilo RAFT, com peso geométrico crescente
(γ = 0,8), de modo que iterações finais pesam mais mas as intermediárias ainda recebem
sinal:

```
L = Σ_i γ^(N−i) · d( exp(p, f_i) , e* )
```

**(h) Aumento de dados: rotação SO(3).** Uma rotação 3D aleatória é aplicada ao par inteiro
— posições dos nós e campo verdadeiro juntos. Numa grade de área igual isso é **exato**: não
há reamostragem dependente de latitude, ao contrário do que aconteceria girando um ERP. É a
simetria natural do problema e o modelo é exposto a ela por inteiro.

**Orçamento.** 1.558.768 parâmetros. A RAFT-large tem ~5,3 milhões, e vem pré-treinada em
corpora grandes (Chairs → Things → Sintel/KITTI/HD1K). Toda comparação aqui é assimétrica
nessa direção.

### 2.5 Quatro emendas que parecem detalhes e não são

Cada uma foi forçada por um teste que falhou; cada uma é individualmente necessária (o
portão de CPU da Seção 6.2 não passa sem qualquer uma delas):

1. **Características de casamento sem posição.** As coordenadas `xyz` do nó ficam **só** no
   encoder de contexto. Com `xyz` nas características de casamento, todo nó se parece
   consigo mesmo mais do que com qualquer outro — um viés embutido de fluxo zero. Suspeita
   forte de ter contribuído para a correlação inerte do Ato I.
2. **Correlação em unidades de cosseno.** Havia um `/√C` herdado sobre produtos internos já
   normalizados: a correlação chegava na GRU com magnitude ≤ 0,1 contra contexto de ordem 1.
   A GRU literalmente não ouvia o sinal.
3. **Perda auxiliar de casamento** (InfoNCE com alvo suave sobre a janela do estêncil, peso
   0,5, aquecida sozinha no primeiro quarto do treino). Descoberta dura: treino
   ponta-a-ponta **nunca** desenvolve características casáveis em nenhum orçamento testado —
   é um problema de ovo e galinha (sem características casáveis, a correlação não dá sinal;
   sem sinal, as características não aprendem a casar). Com supervisão direta de casamento,
   as características chegam ao seu piso de entropia em ~300 passos.
4. **Atalho da correlação direto para a entrada da GRU.** Mesmo com características
   perfeitas — uma sonda de regressão linear decodifica o estêncil para fluxo com cosseno
   0,99 — o caminho "codificador de movimento → GRU" sozinho não se alinha ao sinal em
   orçamentos pequenos. Com o atalho, aprende em ~450 passos.

---

## 3. O protocolo de avaliação (que é, ele próprio, uma contribuição)

Três decisões, e cada uma muda conclusões:

**(1) Tudo é reportado como melhoria percentual sobre `zero-flow`.** Prever movimento nulo
em todo nó é o mínimo absoluto que qualquer estimador precisa superar. Parece um
espantalho — não é. Numa esfera majoritariamente estática, a primeira obrigação é **não
alucinar movimento**. A varredura sistemática de seis artigos de fluxo 360° (lidos em texto
completo, com busca por `zero flow`, `trivial`, `identity`, `constant`, `density`,
`abstain`…) encontrou **nenhum** que reporte um preditor trivial. Isso é o que abre a
Seção 6.5.

**(2) Desagregação em três eixos.** Por **região** (polos |lat|>60°, equador |lat|<30°,
costura longitudinal), por **limiar de atividade** (>0,25° / >0,5° / >1,0°) e por **faixa de
deslocamento** (intervalos disjuntos). O terceiro eixo é o que permite comparar datasets
diferentes *na mesma magnitude de movimento* em vez de comparar médias.

**(3) Média ponderada por ângulo sólido.** Bug encontrado tarde: a média sobre nós só
coincide com a média sobre a esfera numa grade de área igual. A máscara polar cobre 13,4%
dos nós HEALPix e 32,8% dos nós de uma grade equiangular — cada agregado respondia a uma
pergunta diferente em cada perna do experimento. Agora cada nó pesa a área exata da sua
célula. Em HEALPix os pesos são uniformes, então **todo número já publicado se reproduz sem
alteração** — mas comparações entre grades passaram a ser legítimas.

---

## 4. Os dois regimes — a chave para ler todo o resto

| Conjunto | Origem | Deslocamento mediano | Regime |
| --- | --- | --- | --- |
| **FLOW360** | renderização naturalista | **0,132°** (≈0,28 px ERP) | sub-pixel |
| **FlowScape** | CARLA, direção urbana | **2,467°** | movimento amplo |
| Replica360 | cenas internas | ~12° | movimento amplo |

Quase **vinte vezes** de diferença. Isso não é detalhe de dataset, é a divisão do problema
em dois:

- **Regime de movimento amplo.** Dezenas de pixels de deslocamento por toda a esfera. É para
  isso que *todo* benchmark 360° publicado foi construído — herdado do fluxo de perspectiva
  (FlyingChairs/Things → Sintel/KITTI), onde a dificuldade de interesse é casamento de longo
  alcance e oclusão.
- **Regime de vídeo real.** Quadros consecutivos a 30–60 fps. A aritmética de taxa de quadros
  torna esse regime universal: **qualquer coisa que não seja rápida e próxima da câmera se
  move menos de um pixel entre quadros.** No FLOW360, ~76% da esfera fica abaixo de 0,25° e
  ~88% abaixo de 0,5°. O campo é majoritariamente estático, com movedores esparsos e
  espacialmente estruturados (paralaxe, movimento articulado), sob mudança real de aparência
  entre quadros.

Os dois são renderizados — a diferença entre eles **não** é sintético contra real. É
estrutura e magnitude de movimento. A tese estuda o segundo regime, que é onde qualquer
câmera 360° cai no instante em que grava, e para o qual não existe benchmark nem método
dedicado.

---

## 5. A trajetória experimental (o que se descobriu, na ordem em que doeu)

### 5.1 Ato I — a correlação estava inerte, e o modelo era um "prior de aparência"

Primeira versão: grade única (entrada = estimação), consulta por `ang2pix`, sem perda
auxiliar. Ela treina de forma estável, decora 10 pares até erro quase zero, e no conjunto
real bate o zero-flow em **+2,9%** nos nós ativos. Parecia um começo modesto e saudável.

Aí veio a escada de ablações, e cada degrau devolveu o **mesmo +2,9%**:

| # | Experimento | ativos>0,25° | Veredito |
| --- | --- | ---: | --- |
| 0 | linha de base r4 | +2,90% | aprende, mas limitado |
| 1 | estima r4 → supervisiona r6 | +2,82% | o upsampler só interpola |
| 2 | correlação em r5 (metade do espaçamento) | +2,81% | nada muda |
| 3 | correlação local em r6 | +2,80% | **refuta a hipótese de resolução** |
| 4 | **correlação zerada** | +2,90% | **a correlação é inerte** |
| 5 | **contexto zerado** | +2,90% | **o quadro 2 não é usado** |
| 7 | cabeça diferencial (Lucas-Kanade) | **−4,43%** | ativamente errada |

As linhas 4 e 5 são o golpe. Zerar a correlação deixa o resultado **bit a bit idêntico** —
e como o contexto só vê o quadro 1, esse modelo é literalmente um modelo *de um quadro só*
que já alcança o teto inteiro. O espelho também vale: desligando o contexto, sobra só a
correlação, e ela também converge para +2,9%. Dois caminhos de entrada disjuntos chegam ao
mesmo campo. **Conclusão: +2,9% é o teto de um prior de aparência — a rede aprendeu "onde
costuma haver movimento nesse tipo de cena" e nunca fez correspondência.**

A cabeça diferencial (uma solução de Lucas-Kanade esférica sobre características aprendidas)
foi construída como teste decisivo: ela depende de `f2` estruturalmente e **não pode** cair
no prior. Resultado: pior que não prever nada. E o diagnóstico de confiança fechou o
paradigma — o fluxo produzido é **anti-correlacionado** com o movimento verdadeiro
(cos −0,07 no geral, **−0,18 nos 5% de nós mais confiantes**): mais errado exatamente onde
as características são mais nítidas. Nenhuma fração de *gating* por confiança bate zero.

Diagnóstico do Ato I: "o movimento é sub-nó em qualquer grade que caiba na GPU". Correto nos
sintomas, **errado na causa** — como o Ato II mostrou.

### 5.2 Ato II — a correlação passa a funcionar, e o modelo ganha nos polos

O post-mortem identificou três defeitos separáveis, não um limite físico: (i) retina amarrada
à estimação, (ii) consulta por `ang2pix` sem gradiente, (iii) nada faz as características
aprenderem a casar. As Seções 2.1, 2.4(d) e 2.5(3) descrevem os três consertos.

Portão de CPU antes de gastar GPU: recuperar uma rotação **sub-nó** (0,3× o espaçamento de
estimação) a partir de textura sintética multi-banda. Modelo completo: similaridade de
direção **0,997**. Controle com correlação ablada: **−0,019**. Primeira versão do OSLO-RAFT
em que a correlação faz trabalho.

Em movimento real amplo (Replica360, mediana ~12°): **1,56° de erro contra 13,53° do
zero-flow — +88,4%** (três sementes: 1,74° ± 0,25, +87,1 ± 1,9%). O controle honesto: treinar
do zero *com a correlação ablada* converge para **paridade exata com zero-flow (−0,1%)** —
ou seja, **100% do ganho passa pela correlação.**

E o confronto direto com o proxy da literatura (RAFT-large congelada aplicada ao ERP):

| Erro geodésico (graus) | Replica360: OSLO | Replica360: RAFT | FlowScape: OSLO | FlowScape: RAFT |
| --- | ---: | ---: | ---: | ---: |
| Global | 1,56 | **1,16** | 1,158 | **0,872** |
| Polos | **2,88** | 3,65 | **3,147** | 3,650 |
| Razão polos/equador | **2,37×** | 5,36× | **4,33×** | 10,4× |

Leitura honesta, dois lados: **perde no global nos dois conjuntos, ganha nos polos nos dois,
e a vantagem de uniformidade se repete com valor quase idêntico** (2,3× em Replica360, 2,4×
em FlowScape) — em dois datasets de origem completamente diferente, com o mesmo par de
modelos. É a replicação que sustenta a afirmação: **geometria esférica compra precisão polar
e homogeneidade; não compra precisão média.**

### 5.3 Ato III — o mesmo modelo, no regime sub-pixel, não bate "não fazer nada"

Levar o modelo do Ato II para o FLOW360 falha, e falha duas vezes, inclusive depois de
remover o confundidor óbvio (na mistura, alvos de 0,1° contribuíam ~1% do gradiente contra
os 13° do Replica; um *fine-tune* só em FLOW360, com perda ponderada por movimento e
currículo de ângulo pequeno, não salvou).

Então veio o experimento que reorientou o projeto inteiro. **Mesmo checkpoint, mesmos
quadros, mesma escala de movimento, uma variável:**

| o quadro 2 é… | resultado |
| --- | ---: |
| o quadro 1 **reamostrado** no movimento verdadeiro | **+80,6%** (resíduo ≈ 0,04° ≈ 0,12 px ERP) |
| o **quadro 2 real** renderizado | **−32,2%** |

Um modelo de 1,5M de parâmetros rastreia movimento até **um décimo de um nó da retina** —
uma ordem de magnitude abaixo da sua grade de entrada — quando a constância de brilho vale
exatamente. Nos mesmos pares reais, não consegue bater ficar parado.

Isso eliminou de uma vez: "a grade não resolve", "os rótulos são ruins", "faltou sinal de
treino", "a métrica é artefato". Sobrou uma explicação — e por vários meses ela foi lida como
**robustez a mudança de aparência entre quadros** (sombreamento, anti-aliasing, oclusão, dois
renders independentes).

### 5.4 A decomposição — a causa não é a aparência, é a **estrutura do campo**

A leitura acima estava errada, e o quadrado de decomposição mostra o porquê. Quatro células,
duas variáveis (campo de movimento: rotação global vs. campo real; aparência: reamostrada vs.
real):

| | aparência limpa | aparência real |
| --- | ---: | ---: |
| **campo de rotação** | +80,6% | +44,4% |
| **campo real** | **−72,5%** | −32,2% |

Leia a coluna e a linha:

- trocar **estrutura de campo** custa ~153–162 pontos;
- trocar **aparência** custa ~36–42 pontos;
- **a estrutura domina por ~4×**;
- e, contra-intuitivamente, a aparência **inverte de sinal** no campo real (+40): o ruído
  amortece a confiança da correlação, e numa esfera quase parada uma previsão amortecida está
  mais perto de zero, portanto melhor.

A curva de magnitude fecha o argumento (mesmo conjunto, mesma aparência nas duas pernas,
variando só a escala do campo 20×):

| deslocamento mediano | campo de rotação (coerente) | campo real (esparso) |
| ---: | ---: | ---: |
| 0,11° | +1,7% | −32,1% |
| 0,57° | +43,1% | −59,0% |
| 2,29–2,59° | **+84,1%** | **−70,4%** |

As curvas **divergem**: aumentar a magnitude do campo real não resgata nada, **piora**. O vão
cresce de 34 para 155 pontos. Com aparência idêntica, conjunto idêntico e modelo idêntico, a
única variável restante é a estrutura do campo.

E o que "estrutura" quer dizer concretamente: no FlowScape, um nó que se desloca 0,7°
pertence a um campo coerente de ego-movimento — os vizinhos também se movem, e a GRU pode
agregar essa evidência espacialmente. No FLOW360, o mesmo 0,7° é um objeto isolado sobre uma
esfera parada. **Não há nada para agregar.**

Notícia boa, e ela é o que salva o resultado de ser puramente negativo: o eixo é
**treinável**. 5 mil passos treinando na família de campo-real-limpo levam aquela perna de
−72,5% para **+42,7%**. Não é limite arquitetural, é lacuna de dados e currículo.

### 5.5 O resultado positivo, e o seu escopo exato

A campanha final (mistura de dados incluindo um gerador sintético próprio de objetos
independentes, mais nuisance modelada, mais média móvel exponencial dos pesos para domar a
variância — que sozinha reduziu o desvio em 9×) chega a:

- **+4,2% nos nós ativos >0,5°** (desvio 0,9 ponto) no conjunto de **validação** do FLOW360,
  o primeiro positivo consolidado sobre pares reais no regime sub-pixel — acima do teto de
  prior de aparência do Ato I (+2,9%);
- com **controle de correlação**: ablar a correlação faz o erro colapsar de 0,28° para 54°.
  Casamento genuíno, não prior.

E o escopo, declarado sem maquiagem: **esse número não atravessa para o conjunto de teste**,
onde a mesma métrica dá −4,0%. A decomposição por faixa de deslocamento atribui ~42% da
diferença à composição distinta dos dois cortes e ~58% a falha de generalização. A afirmação
sustentável é: *o ganho existe dentro do mesmo pool de sequências e não atravessa cortes.*

---

## 6. Os resultados, e como lê-los

### 6.1 No movimento amplo, todo mundo ganha — e o OSLO-RAFT perde a comparação

FlowScape (teste), melhoria sobre zero-flow (erro global do zero: 3,402°):

| Método | Global | Ativos>0,5° | Polos | Equador |
| --- | ---: | ---: | ---: | ---: |
| PanoFlow | **+92,6** | **+93,0** | **+95,3** | **+85,5** |
| RAFT-large | +74,4 | +74,6 | +54,7 | +80,5 |
| OSLO-RAFT | +66,0 | +67,0 | +61,0 | +59,7 |

A PanoFlow vence em **toda** região e **todo** limiar — erro global 4,6× menor, polar 8,4×
menor — inclusive em uniformidade. Consequência direta e assumida no texto: **"OSLO-RAFT é
bom em movimento amplo" não sobrevive como afirmação inter-laboratório.** O ganho de
uniformidade da Seção 5.2 vale contra uma arquitetura de perspectiva aplicada ao raster; não
vale contra o estado da arte de fluxo 360°.

Diagnóstico da distância: a grade de estimação do OSLO é r4 (~3,66° entre nós) e a mediana
do dataset é 2,47° — **sub-nó mesmo em "movimento amplo"** — contra ~0,35°/px da PanoFlow.
Mas veja a Seção 6.4 antes de aceitar essa explicação.

### 6.2 No regime sub-pixel, **ninguém** bate o preditor trivial

FLOW360 (teste), melhoria sobre zero-flow (erro global do zero: 0,4314°; mediana 0,132°):

| Método | Global | Ativos>0,25° | Ativos>0,5° | Polos |
| --- | ---: | ---: | ---: | ---: |
| SLOF (raft) | −221,0 | −59,7 | −25,9 | −133,3 |
| SLOF (raft ajustada) | −10,4 | −6,2 | −5,3 | −12,8 |
| SLOF (rotação simples) | −9,5 | −2,3 | −3,2 | −18,9 |
| SLOF (rotação dupla) | −0,6 | −0,1 | +0,0 | −1,2 |
| PanoFlow | −11,3 | −5,8 | −4,8 | −13,8 |
| RAFT-large | −16,0 | −8,0 | −7,6 | −23,9 |
| **OSLO-RAFT** | −15,1 | −4,9 | −4,0 | −26,9 |

Nenhuma linha positiva no global. Nenhuma positiva em ativos>0,25°. A única perto de zero
(rotação dupla) é um preditor que devolve praticamente o campo nulo — **herda** a linha de
base em vez de superá-la. **Três arquiteturas, três laboratórios, incluindo o método da casa
do próprio FLOW360, e o nosso.** O OSLO-RAFT está dentro do conjunto: não é exceção ao limite
que ele mede.

Isso importa porque a literatura reporta reduções de EPE contra o melhor anterior (PanoFlow:
27,3% sobre o estado da arte no FLOW360) e **nunca** contra um preditor trivial. A lição já
foi aprendida em campo vizinho: em predição de vídeo, "copie o último quadro" é uma linha de
base notoriamente forte, e aquela comunidade passou a reportá-la. Em fluxo óptico — e menos
ainda em 360° — isso nunca atravessou.

### 6.3 O contraste de regime — a evidência mais forte do trabalho

Mesmos pesos, mesmos modelos. **Só o dataset muda:**

| Modelo | FlowScape (amplo) | FLOW360 (sub-pixel) | oscilação |
| --- | ---: | ---: | ---: |
| PanoFlow | +92,6 | −11,3 | ~104 pontos |
| RAFT-large | +74,4 | −16,0 | ~90 pontos |
| OSLO-RAFT | +66,0 | −15,1 | ~81 pontos |

Todas cruzam o zero. Isso mata de vez a leitura "seus métodos são fracos": os mesmos pesos
produzem +66 a +93% no outro regime. **A falha pertence ao regime.**

E a objeção óbvia — "os datasets diferem em magnitude" — é respondida controlando a
magnitude em vez de agregá-la, comparando **faixas iguais de deslocamento** nos dois
conjuntos: na faixa 0,5°–1°, PanoFlow marca +84,2% no FlowScape e −6,4% no FLOW360;
RAFT-large, +78,4 contra −8,0. Na faixa 2°–4°, PanoFlow marca +95,2 contra −7,3. **As
inversões de 80 a 105 pontos sobrevivem ao controle de magnitude** — o que muda não é o
quanto o nó se moveu, é a vizinhança dele.

### 6.4 A grade de estimação **não** é o gargalo (e isso muda para onde vai o esforço)

Suspeita natural: o erro global do OSLO é culpa dos 3.072 nós de estimação. Dá para testar
sem treinar nada. A sonda entrega à grade de estimação a **resposta perfeita** (amostra a
verdade diretamente nos nós de estimação) e reconstrói r6 pelo transporte do próprio modelo.
O que sobra é imposto **só** pela resolução. Três reconstruções delimitam o piso, e a
terceira — a melhor combinação convexa possível — é limite inferior real, porque nenhum
upsampler dessa família pode superá-la.

| | oráculo r4 | erro real do OSLO | fração |
| --- | ---: | ---: | ---: |
| FlowScape (1.386 pares) | 0,062° | 1,158° | **~5%** |
| FLOW360 (2.567 pares) | 0,025° | 0,497° | **~5%** |

**Os outros 95% são do modelo.** Refinar de r4 para r5 move o oráculo em 0,019° e 0,003° —
irrelevante. E o número que fecha o argumento: o piso em r4 fica **abaixo** do erro da
PanoFlow (0,251° no FlowScape). A grade suportaria acurácia 4× melhor do que a alcançada,
portanto **não é ela que separa os dois métodos**. O que resta como hipótese é capacidade +
receita de treino (lote 2, e a compensação por acumulação de gradiente falhou) — e isso
ainda não foi isolado, então a distância para a PanoFlow **não deve** ser atribuída à
representação esférica.

(Ressalva regional: na costura o piso do oráculo é 11–12× o global e chega a 28% do erro do
OSLO. Aquela coluna deve ser lida com cautela; a causa provável é descontinuidade de embrulho
na verdade de referência de origem, não diagnosticada.)

### 6.5 O modo de falha dominante, em um número

No FLOW360, a faixa abaixo de 0,0625° contém **42,8% dos nós**, cujo deslocamento verdadeiro
é 0,0103° — **0,029 pixel ERP**. O erro do OSLO-RAFT nesses mesmos nós é 0,151° — **0,43
pixel**.

**Em quase metade da esfera, a cena se move três centésimos de pixel e o modelo afirma quinze
vezes isso.** Isso é falha de **calibração**, não de estimação: o modelo ganha onde há
movimento e perde na maioria estática. E há evidência controlada do mecanismo: no experimento
de rotação, nós próximos ao eixo têm deslocamento verdadeiro constante, e mesmo assim o erro
cresce 2,4× conforme o resto da esfera acelera. **O movimento vaza dos vizinhos pela agregação
recorrente da GRU.**

Isso motiva a proposta registrada (`docs/plans/DECISION_GATE.md`, ainda não executada): uma
**porta estático/movimento** por nó, treinada como classificação binária em `|GT| > 0,25°`,
multiplicando o campo previsto. Com a porta fechada, a saída é **exatamente** o campo nulo —
o modelo passa a **conter** a linha de base que precisa superar, e só pode perder para ela
decidindo errado. A aritmética prevista: global vai de −16,8% para ≈ +3%, o que seria a
primeira inversão de sinal do regime.

O enquadramento honesto disso está no próprio documento: o mecanismo **não é novo** —
codecs de vídeo fazem exatamente isso desde os anos 1990 (modo SKIP), e o fluxo óptico
clássico abstinha por limiar de confiança até a era profunda descartar a prática. O que é
novo é o **diagnóstico**: *compressão precifica a abstenção (em bits), predição seletiva
precifica (em cobertura), e a função de perda de fluxo óptico não precifica nada* — ela só
cobra por discordar da verdade, nunca recompensa emitir nada. Uma rede densa **não consegue
aprender a se abster porque nenhum termo da sua perda paga por isso.**

### 6.6 O controle que derrubou a premissa do próprio trabalho

A premissa fundadora era: *área igual compra precisão polar*. O teste correto é um par
casado — a **mesma arquitetura**, mesmos 1.558.768 parâmetros, mesma receita, mesma semente,
mesmo número de passos, rodando sobre uma grade **equiangular** (latitude/longitude) com a
mesma contagem de nós por nível e o mesmo aninhamento em quatro filhos. **Só a posição dos
nós muda.**

Em escala controlada (2 mil passos, só Replica360), a grade equiangular **vence em todas as
regiões dos dois conjuntos**, com faixas de duas sementes que não se sobrepõem (30% de
diferença entre o pior equiangular e o melhor HEALPix). Havia um confundidor plausível — os
dados são armazenados em ERP, e a grade equiangular reamostra os pixels quase um a um,
enquanto a HEALPix sempre interpola. O controle (rotacionar a validação para destruir esse
alinhamento) mostrou efeito **oposto ao previsto**: não custou nada à perna equiangular e
custou 11,6% à HEALPix.

Em escala de produção (20 mil passos, mistura completa), a ordenação **não replica**:
ativos>0,5° dá **+4,2 na HEALPix contra +0,3 na equiangular**, global empata em −21,2, e a
equiangular mantém vantagem nos polos por 20 pontos. Replicação parcial com **inversão de
sinal na métrica principal**.

O que cai é a **atribuição**, não a medição:

- A Tabela da Seção 5.2 **permanece válida** — a arquitetura esférica continua superando a
  rede sobre raster nos polos, replicado em dois datasets.
- O que **não** se sustenta é atribuir esse ganho ao posicionamento de área igual, que em
  escala controlada é mensuravelmente a escolha pior. A grade equiangular também é um grafo
  sobre a esfera, com perda geodésica, fluxo em plano tangente e vizinhança com embrulho em
  longitude — não é uma CNN sobre ERP.
- O candidato medido para o ganho polar é outro: a **suavização aprendida do upsampler**
  (Seção 2.4f).

Duas ressalvas registradas junto: a perna equiangular parte de um modelo pré-treinado em
HEALPix (desvantagem declarada **antes** da execução), e as duas grades não são pontuadas
contra o mesmo campo de referência (o zero-flow em ativos vale 0,856° na HEALPix e 0,894° na
equiangular, porque cada grade interpola a verdade com estêncil diferente — viés que
provavelmente favorece a HEALPix, na mesma direção do resultado).

Nas duas escalas, o critério de leitura foi declarado **antes** da execução, e nas duas o
ramo que disparou foi o desfavorável.

### 6.7 Dois bugs de métrica que mudariam conclusões

Vale registrar porque ilustra o quanto o instrumento importa quando os efeitos são pequenos:

1. **Piso do `arccos` em float32** (0,028°). 42,8% dos nós do FLOW360 se movem 0,0103°,
   abaixo do piso — então a linha de base de fluxo nulo era medida em 0,2105° em vez de
   0,2017°, **4,2% inflada**. Os erros dos modelos (0,28°–0,46°) estão 10–16× acima do piso e
   não se mexem. Efeito líquido **assimétrico: a métrica antiga favorecia todo método medido
   contra a linha de base.** Corrigido para haversine; o modelo final piora 4,4 pontos no
   global, e os ativos praticamente não se movem (os limiares de atividade já excluíam a
   faixa onde o piso atuava — por isso o resultado positivo da Seção 5.5 sobrevive intacto).
2. **Média por nó vs. média por área** (Seção 3.3). Só coincidem em grade de área igual;
   sem a correção, a comparação HEALPix × equiangular da Seção 6.6 seria inválida.

---

## 7. Resumo executivo: o que sobrevive, o que caiu, o que falta

**Sobrevive e é carregador de resultado:**

1. **O protocolo de avaliação esférico** — erro geodésico por região, por limiar de atividade
   e por faixa de deslocamento, sempre contra zero-flow. É o instrumento que torna todo o
   resto mensurável, e a ausência dele na literatura é o que permitiu o resultado da
   Seção 6.2 passar despercebido.
2. **A caracterização de regime** — a oscilação de 80 a 104 pontos com pesos idênticos,
   sobrevivendo ao controle de magnitude. Evidência mais forte do trabalho.
3. **A isolação causal** — a estrutura do campo domina a aparência por ~4×, e a magnitude
   *piora* o campo real em vez de ajudar. Esse nível de diagnóstico não existe na crítica
   vizinha (predição de vídeo).
4. **A medida de calibração estática** — 15× de super-afirmação em 43% da esfera, com o
   mecanismo (vazamento de vizinhos pela recorrência) demonstrado em experimento controlado.
5. **A comparação arquitetural replicada** — geometria esférica compra precisão polar e
   uniformidade *relativamente a uma arquitetura de perspectiva sobre raster*, replicado em
   dois datasets independentes (2,3× e 2,4× mais plano).
6. **A demonstração de poder de resolução sub-pixel** — 0,04° de resíduo (≈0,12 px ERP) sob
   constância de brilho exata, com um modelo de 1,5M de parâmetros.
7. **A sonda de piso de grade** — exclui a resolução de estimação como causa (95% do erro é
   do modelo).

**Caiu, e está escrito como caiu:**

- *"Amostragem de área igual compra precisão polar"* — derrubado pelo controle casado
  (Seção 6.6). A precisão polar é real e replicada; a **atribuição** à área igual não é.
- *"OSLO-RAFT é bom em movimento amplo"* — derrubado pela PanoFlow no FlowScape (Seção 6.1).
  Vitórias ficam escopadas a baselines internos; sem alegação de estado da arte.
- *"O muro é a robustez a mudança de aparência"* — corrigido pela decomposição (Seção 5.4):
  o muro é a **estrutura do campo**, com aparência como fator secundário (~1/3, no máximo).
- *"O muro é resolução sub-nó"* (diagnóstico do Ato I) — refutado pelo Ato II e pela sonda de
  piso.

**Falta (com critério declarado a priori):**

1. **PriOr-Flow** (estado da arte atual, ICCV 2025) sob o mesmo protocolo — ataca a distorção
   polar por outro mecanismo, então é a comparação mais informativa disponível.
2. **Capacidade e receita de treino** — o esforço que a Seção 6.4 liberou. Lote genuinamente
   maior (o atual é 2) e revisão do cronograma de taxa de aprendizado.
3. **Escala de dados sintéticos** com objetos independentes, que quebram o viés de campo
   coerente de todos os datasets existentes.
4. **A porta estático/movimento** (Seção 6.5), com falseador pré-registrado: precisa bater o
   classificador trivial "tudo é estático" em F1 antes de qualquer outra coisa — e há
   precedente interno de falha exatamente dessa classe (a confiança da cabeça diferencial saiu
   anti-correlacionada).
5. **Reproduzir a Tabela 1 do SLOF com uma linha `zero` a mais** — na métrica deles, na
   resolução deles, nos *buckets* deles. Hoje a afirmação "ninguém bate zero" está escopada ao
   nosso protocolo, e um revisor com os dados e uma tarde pode testá-la no dele. Ou vira a
   tabela mais persuasiva da tese, ou manda escopar a afirmação antes de publicá-la.

**A pergunta da tese, reformulada pelo que foi medido:** não é mais *"construímos um
estimador de fluxo 360° melhor?"*, e sim **"quando estimar fluxo em vídeo 360° é melhor do
que não estimar, e como se constrói um estimador que sabe a diferença?"**
