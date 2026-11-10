# Horizontes: outros formatos de trabalho com HEALPix para fluxo óptico

Busca de 2026-11-09, feita depois de fechar a caça à paridade com o RAFT (`docs/plans/RAFT_PARITY.md` §11.25–11.26). Complementa `Formulações de fluxo para HEALPix.md` (propostas P1–P5) e não repete o que está lá. Cerca de 35 buscas e 12 páginas lidas. "Não encontramos" significa só isso: a busca foi curta por tema. Antes de escrever qualquer afirmação de novidade, passar por `docs/plans/LITERATURE_SCOPE.md`.

> **Revisão da ordem (2026-11-09, depois da conversa).** A ordem das direções abaixo foi revista. Faces com pesos planares fica só como controle, porque a paridade já mostrou que o planar empata. Dados de clima em HEALPix é um caso de uso, não uma mudança de método. A única mudança de método é a hierarquia como representação. A direção promovida é o fluxo como movimento de um codec de vídeo na esfera. O teste de decisão dela é o probe `p10` (`experiments/proposals/p10_prediction_probe.py`, `docs/plans/RAFT_PARITY.md` §11.27), que está na fila.

## O que a evidência diz sobre cada propriedade do HEALPix

| Propriedade | O que a evidência diz | Fonte |
|---|---|---|
| Área igual | Efeito pequeno na acurácia. No projeto, o gêmeo equiangular empatou ou venceu. No codec HEALPix, ~4 %. No SO3UFormer (icosfera), o termo de quadratura na atenção vale +1,4 mIoU; em HEALPix esse termo é constante, então vem de graça. | A1 do projeto; arXiv 2610.32561; arXiv 2602.22867 |
| Hierarquia NESTED | É a alavanca nos trabalhos que a usam como representação: ~45 % no codec; Field-Space Attention com 15,5 M de parâmetros vence um ViT de 54,8 M em super-resolução do ERA5. O OSLO só a usa para pooling. | arXiv 2610.32561; arXiv 2512.20350 |
| 12 faces quase quadradas | Permitem Conv2d comum com padding entre faces (DLWP-HPX, cBottle, Aircast-Mars). Há artefatos documentados nas costuras das faces polares. Não encontramos uso em visão. | arXiv 2311.06253; 2505.06474; 2607.19370; 2407.11666 |
| Sem costura e sem polo singular | Os pixels de overshoot do FlowScape são um artefato do raster ERP. Em espaço de direções eles não existem. | P1/P6 do projeto |
| Anéis de iso-latitude | Transformada harmônica esférica rápida e diferenciável em GPU (cuHPX, 2025). | arXiv 2510.01785 |
| Dados nativos | Modelos de clima de escala de km já gravam a saída direto em HEALPix NESTED hierárquico (ICON/nextGEMS, Zarr por nível de zoom). | easy.gems.dkrz.de; EGU24-17007 |

Leitura: as propriedades que rendem são a hierarquia, o formato em faces e o fato de o dado já nascer em HEALPix. A área igual sozinha não rende.

## Direções, em ordem de recomendação

### 1. Faces HEALPix com pesos planares pré-treinados (continuação do P4)

- **Formato:** as 12 faces como imagens nside×nside com padding dos vizinhos; rede planar comum por cima.
- **Por que:** é o único formato HEALPix que herda pré-treino planar (FlyingThings), e a diferença de orçamento para PanoFlow/PriOr é de 60–75×. O P4 do projeto, sem nenhum treino, deu RAFT-small em faces empatado no in-raster (−3,5 %) e melhor no equador e no viewport (−26 %).
- **Evidência externa:** o padding em faces é padrão em clima (DLWP-HPX, cBottle, Aircast-Mars 2026). Duas buscas dirigidas não acharam uso em visão nem com pesos 2D pré-treinados.
- **Risco:** continua sendo RAFT. A contribuição seria a representação. O RAFT-large não repetiu o ganho no P4 (+13 % in-raster). Costuras das faces polares têm artefatos documentados (arXiv 2407.11666).
- **Primeiro experimento:** fine-tune do RAFT-small em faces com a receita pareada (2 seeds), mais a cabeça de banda do P6 para o overshoot. Leitura: in-raster e viewport contra o RAFT-small em ERP.

### 2. Dado nativo em HEALPix: ventos a partir de campos de modelos de clima

- **Formato:** não há reamostragem. Entrada, alvo e métrica vivem em HEALPix.
- **Por que:** inverte a vantagem de casa. O WindFlow (KDD 2022) treina RAFT em pares de umidade do GEOS-5 Nature Run para recuperar vento, em recortes lat-lon de uma grade global 2881×5760, e trabalhos de 2026 ainda usam esse mesmo RAFT (arXiv 2609.03100). O ICON/nextGEMS grava `ua`, `va`, `hus` e água de nuvem em HEALPix NESTED, com níveis de zoom prontos e passo de até 30 min. Aqui o concorrente planar é quem teria de reamostrar, os polos importam, o protocolo ERP com overshoot some, e o alvo denso é exato e abundante.
- **Evidência de lacuna:** a busca exata por "HEALPix" + "optical flow" só devolve o repositório do projeto. Nenhum trabalho de vento por fluxo óptico encontrado discute projeção ou polos.
- **Riscos:** (a) vento não é o movimento aparente do traçador; o WindFlow aceita isso e treina supervisionado. (b) Regime sub-nó: 20 m/s em 30 min são 36 km, ou seja ~0,7 px no zoom 7, ~2,8 px no zoom 9 e ~5,7 px no zoom 10 (conta minha, pixel ≈ 58,6°/2^zoom). Só os zooms finos escapam do regime em que o OSLO falha, e eles têm 3–12 M de pixels. (c) Não verifiquei política de acesso nem quais variáveis 3D existem em 30 min. (d) Muda o domínio da dissertação.
- **Primeiro experimento (horas, sem treino):** baixar um dia de `hus` e `ua/va` num nível, medir a distribuição do deslocamento em nós por zoom e a linha de base zero. Só segue se o deslocamento mediano passar de 1 nó em algum zoom treinável na 3090.

### 3. A hierarquia como representação do fluxo (campo grosso + resíduos NESTED)

- **Formato:** o fluxo é um campo no zoom grosso mais resíduos por nível, e a rede atualiza os resíduos (como o Field-Space Attention faz com temperatura e o codec faz com latentes).
- **Por que:** é a propriedade com mais evidência externa e a única que o OSLO não usa para estimar. Resíduo zero é "estático", então o gate do P7 vira esparsidade do resíduo em vez de uma cabeça extra.
- **O que já sabemos:** est r5 sozinho não ajudou; o P2 (cascata clássica) ficou inconclusivo porque o matcher clássico mal vence zero; o MS-RAFT mostra que fino sem inicialização piora e fino por cascata melhora. A versão treinada nunca foi rodada.
- **Risco:** 2–4× o custo de treino, evidência só de outras tarefas, e o gate já se mostrou genérico (ajuda mais o RAFT).
- **Primeiro experimento:** braço A1 da Proposta 2 antiga, mas com saída residual por nível e L1 nos resíduos; mesmo critério (fechar metade do gap, acima do espalhamento entre seeds).

### 4. Destilação sem rótulos para um aluno HEALPix

- **Formato:** professores planares em recortes perspectiva sem distorção; pseudo-rótulos levados à esfera e supervisionados por recorte nos nós.
- **Evidência:** o MTPano (2026) faz isso para segmentação, profundidade e normais com 140 k panoramas e 32 recortes por imagem, e supervisiona por recorte porque costurar gera artefatos. O TAPVid-360 (NeurIPS 2025) gera rótulos direcionais rodando CoTracker3 em vistas perspectiva de vídeos 360. O 360-1M tem um milhão de vídeos 360.
- **Por que HEALPix:** recortes de qualquer orientação caem numa grade única com densidade uniforme, sem caso especial polar.
- **Risco:** não serve para o FlowScape. O RAFT congelado em faces fica em 8–10 px in-raster lá, pior que o OSLO treinado (5,1–5,5 px), então o professor é pior que o rótulo. Serve para vídeo real, onde o gargalo medido foi dado.
- **Primeiro experimento (sem treino):** qualidade do pseudo-rótulo no flow360 corrigido: RAFT-large em recortes perspectiva reprojetado nos nós contra o GT.

### 5. Avaliar num consumidor: interpolação de quadros e codec de vídeo na esfera

- **Evidência:** o SVI360 (ECCV 2026) faz interpolação 360 com duas vistas ortogonais e reporta fluxo no FlowScape (EPE 12,36, SEPE 15,22, treinado sem supervisão de fluxo). Não compara com métodos de fluxo, não traz linha de base trivial e não separa por latitude. O codec neural 360 do ICCV 2025 adapta codecs planares (~8 % de economia) e não modela movimento na esfera. O codec HEALPix INR é só de imagem. Não encontramos codec de vídeo aprendido com compensação de movimento em HEALPix.
- **Risco:** no viewport o RAFT já está ~15 % à frente, então a expectativa honesta é que ele também vença aqui.
- **Primeiro experimento (sem treino):** interpolar o quadro do meio por warping com o fluxo de cada modelo e medir WS-PSNR e PSNR de viewport.

### 6. Mudar a saída para direções (rastreamento 360)

O TAPVid-360 define a tarefa como prever a direção do ponto na esfera, inclusive fora do campo de visão. O baseline prevê uma rotação por ponto, perde dentro do quadro para o CoTracker3 (0,406 vs 0,559) e fora do quadro acerta pouco (δ 0,116). Os autores pedem distribuições direcionais. É a tarefa em que a saída esférica é obrigatória, mas exige um rastreador multi-quadro inteiro. Fica como horizonte, não como próximo passo.

### 7. Lado espectral (cuHPX)

Rotação global é o tripleto toroidal ℓ=1 e sai por quadratura exata em HEALPix. Serve para estabilização de vídeo 360 real. No FlowScape não ajuda: o P1 mostrou que o modelo rígido explica só 58 % dos pixels abaixo de 1 px.

## Fatos novos para o registro de literatura

- SVI360 (ECCV 2026): mais um trabalho com números de fluxo 360 sem linha de base trivial e sem estratificação polar.
- SO3UFormer (2026): a maior parte da robustez a rotação vem de remover a codificação absoluta de latitude (25,3 → 64,7 mIoU sob SO(3)); quadratura e viés de gauge somam menos. Coerente com a conclusão do projeto de que a alavanca é a orientação, não a amostragem.
- Reconstrução 3D feed-forward para panoramas apareceu em 2026 (PanoVGGT, Wid3R, RIGOR, 360DVO). Fluxo induzido por profundidade + pose passa a ser um professor possível, com a ressalva do P1.

## Fontes

- HEAL-SWIN: https://arxiv.org/abs/2307.07313
- HEALFormer: https://arxiv.org/html/2603.25471v1
- cuHPX: https://arxiv.org/pdf/2510.01785
- Field-Space Attention: https://arxiv.org/html/2512.20350v1
- Attention on the Sphere: https://arxiv.org/abs/2505.11157
- SO3UFormer: https://arxiv.org/html/2602.22867
- DLWP-HPX: https://arxiv.org/html/2311.06253v2
- cBottle: https://arxiv.org/pdf/2505.06474
- Aircast-Mars: https://arxiv.org/pdf/2607.19370
- Neural Compression of Atmospheric States: https://arxiv.org/pdf/2407.11666
- WindFlow (KDD 2022): https://dl.acm.org/doi/10.1145/3534678.3539345 e https://github.com/tjvandal/windflow
- Ventos 3D por estéreo com RAFT: https://arxiv.org/html/2609.03100
- HEALPix hierárquico nextGEMS: https://easy.gems.dkrz.de/Processing/healpix/healpix_starter.html
- ICON em Zarr HEALPix (EGU24): https://meetingorganizer.copernicus.org/EGU24/EGU24-17007.html
- PARADIS (advecção semi-Lagrangiana neural): https://arxiv.org/abs/2601.21151
- Deriva de gelo marinho com fluxo óptico profundo: https://arxiv.org/abs/2510.26653v1
- SVI360: https://arxiv.org/abs/2607.11710
- 360VFI: https://arxiv.org/html/2407.14066v3
- TAPVid-360: https://arxiv.org/abs/2511.21946
- MTPano: https://arxiv.org/html/2602.05330
- Depth Any Panoramas: https://arxiv.org/abs/2512.16913
- 360-1M: https://arxiv.org/html/2412.07770v1
- PanoVGGT: https://arxiv.org/abs/2603.17571
- 360DVO: https://arxiv.org/html/2601.02309
- Beyond Perspective (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Regensky_Beyond_Perspective_Neural_360-Degree_Video_Compression_ICCV_2025_paper.html
- Codec HEALPix INR: https://arxiv.org/html/2610.32561
- PriOr-Flow: https://arxiv.org/html/2506.23897
- 360° Optical Flow using Tangent Images: https://arxiv.org/abs/2112.14331
