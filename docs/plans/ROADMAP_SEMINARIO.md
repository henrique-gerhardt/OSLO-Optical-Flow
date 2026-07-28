# Roadmap — seminário de andamento

Plano de execução dividido pelo critério: **só bloqueia a escrita do artigo aquilo
que, se ignorado, faz o texto conter uma afirmação errada.** O resto é proposta —
e proposta pré-registrada vale mais num seminário de andamento do que resultado
apressado.

- **Balde A** (§2) — executar antes de escrever. 4 itens.
- **Balde B** (§3) — propor no artigo, com critério de sucesso declarado a priori.
- **Balde C** (§4) — já concluído; entra como resultado, não como plano.

Estado do código: os três alavancas que o Balde A precisa (`--geodesic-metric`,
`--motion-bands-deg`, `--real-resample-flow-scale`) estão implementadas e
validadas localmente em Docker (§5). O restante é execução na box.

---

## 1. Pré-requisito — build na box

O `Dockerfile.oslo_raft` faz `COPY . /workspace/oslo`, então código novo exige
commit + push + rebuild antes de qualquer run:

```
cd ~/Developer/OSLO-Optical-Flow
git pull
OSLO_GIT_SHA=$(git rev-parse HEAD) \
docker compose -f docker-compose.oslo_raft.yml build
```

Prefixo comum de todos os runs abaixo (shards read-only, outputs graváveis):

```
SHARDS_HOST=../sfprep/shards OUTPUT_DIR=./outputs \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
```

Config do OSLO (idêntica à que produziu todas as linhas publicadas):

```
--grid healpix --retina --retina-resolution 7 --resolution 6 \
--estimation-resolution 4 --device cuda --amp \
--pyramid-cache /outputs/pyramid_cache
```

Bandas de movimento usadas em todo o Balde A (cobrem flow360 p50 ≈ 0,13° até
flowscape p90 ≈ 25–33°):

```
--motion-bands-deg 0,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,inf
```

Confirme os caminhos de checkpoint antes de rodar — os nomes abaixo são os de
`/outputs` conforme a campanha P1:

```
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft ls -1 /outputs
```

---

## 2. Balde A — bloqueantes

**Sequencie A1 primeiro.** Se a decisão for trocar a métrica, todos os números
mudam e A2–A4 teriam que ser refeitos.

### A1 — piso numérico da métrica geodésica

**Pergunta.** Quanto das nossas conclusões depende do piso de 0,028° do `arccos`
em float32 (§8b de `UNIVERSALITY_TABLE.md`)?

**Instrumento.** `--geodesic-metric {acos,haversine}`. `acos` reproduz todo
número existente; `haversine` (`2·asin(|a−b|/2)`) é matematicamente idêntico em
[0, π] e exato no zero. É um *switch*, não uma substituição, justamente para que
o efeito do piso seja **medido** em vez de assumido desprezível.

**A1.1 — prova do conserto** (o identity check tem que ir de 0,0280° para 0):

```
SHARDS_HOST=../sfprep/shards OUTPUT_DIR=./outputs \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python run_grid_floor_probe.py \
    --shards /data/shards --sources flow360:test --max-pairs 20 \
    --resolution 6 --estimation-resolutions 6 \
    --geodesic-metric haversine \
    --pyramid-cache /outputs/pyramid_cache \
    --output-dir /outputs/a1_identity_haversine --device cuda
```

Critério: `global_geo_deg` ≈ 0 (era 0,0280 com `acos`).

**A1.2 — impacto na linha headline** (mesma linha, duas métricas):

```
for M in acos haversine; do
SHARDS_HOST=../sfprep/shards OUTPUT_DIR=./outputs \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python run_oslo_raft.py \
    --grid healpix --retina --retina-resolution 7 --resolution 6 \
    --estimation-resolution 4 --device cuda --amp \
    --pyramid-cache /outputs/pyramid_cache \
    --eval-only --init-checkpoint /outputs/P1proper_ema6k/oslo_raft_ema.pt \
    --val-sources flow360:test \
    --geodesic-metric $M \
    --motion-bands-deg 0,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,inf \
    --output-dir /outputs/a1_oslo_test_$M
done
```

Critério: `active_*_improvement_pct` deve mudar **< 1 ponto** (previsão medida em
§5: viés < 0,1% em ≥ 0,25°). Se mudar mais que isso, a tabela de universalidade
inteira precisa ser reemitida em `haversine`.

**A1.3 — a alegação contaminada.** A afirmação B′ "0,046° = 0,13 px ERP" precisa
ser re-medida antes de ir para a tese. Localize o checkpoint B′ (`ls /outputs`) e
rode a perna *resampled* nas duas métricas:

```
for M in acos haversine; do
SHARDS_HOST=../sfprep/shards OUTPUT_DIR=./outputs \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python run_oslo_raft.py \
    --grid healpix --retina --retina-resolution 7 --resolution 6 \
    --estimation-resolution 4 --device cuda --amp \
    --pyramid-cache /outputs/pyramid_cache \
    --eval-only --init-checkpoint /outputs/<CKPT_B_LINHA>/oslo_raft.pt \
    --val-sources flow360:val --val-synth-rot-prob 1.0 \
    --synth-rot-min-deg 0.1 --synth-rot-max-deg 2 \
    --geodesic-metric $M \
    --output-dir /outputs/a1_bprime_$M
done
```

### DECISÃO DA MÉTRICA — FECHADA 2026-07-28: `haversine`

A1.2 rodou (OSLO EMA em flow360:test, 2567 pares, mesma config, só a métrica muda):

| | `acos` (publicado) | `haversine` | Δ |
| --- | --- | --- | --- |
| act₀.₂₅ | −4,9 | −4,92 | **−0,02** |
| act₀.₅ | −4,0 | −4,04 | **−0,04** |
| act₁.₀ | −5,7 | −5,70 | **0,00** |
| global | −14,0 | −15,12 | −1,12 |
| polos | −24,8 | −26,89 | −2,09 |
| zero global | 0,4368° | 0,4314° | −1,2% |

Critério pré-registrado ("actives mudam < 1 ponto") **atendido com folga de 25×**
⇒ a conclusão de universalidade é invariante à métrica. Mas global e polos
deslocam 1–2 pontos, e o baseline zero estava inflado em 1,2%.

**Recomendação anterior revertida.** O argumento "manter `acos` por consistência
com as linhas publicadas" não se sustenta: **nós rodamos todas as 11 linhas**, não
há número externo com que ser consistente. Sob `acos` global e polos estão errados
sem contrapartida; `haversine` é exato e não tem trade-off. ⇒ **toda a tabela sai
em `haversine`**, e o piso do harness (~2e-6°) entra na seção de método como
verificação, não como ressalva.

**Consequência de planejamento: A1 e A4 fundem-se.** Como toda linha precisa ser
re-rodada de qualquer forma, rode `--geodesic-metric haversine` **e**
`--motion-bands-deg` na mesma passada. Ver §2.A4.

### A2 — o gap val→test é composição ou falha de generalização?

**Pergunta.** OSLO faz act₀.₅ **+4,5 em val** e **−4,0 em test**. Isso é (a) o
test ter mais massa nas faixas de deslocamento onde *todo mundo* perde, ou (b) o
modelo genuinamente não generalizar para o pool novo?

**Instrumento.** As bandas do A4, aplicadas aos dois splits com o **mesmo
checkpoint**. Se a curva de melhoria *por banda* for igual em val e test, o gap
agregado é **composição** (o test tem 34,6% de actives contra 24,1% do val, e
zero global 2× maior) — pool shift, não falha de generalização. Se as curvas
divergirem banda a banda, é falha de generalização e precisa ser reportada como
limitação.

Isto não exige checkpoint novo nem código novo além do A4: são dois runs.

```
for S in flow360:val flow360:test; do
SHARDS_HOST=../sfprep/shards OUTPUT_DIR=./outputs \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python run_oslo_raft.py \
    --grid healpix --retina --retina-resolution 7 --resolution 6 \
    --estimation-resolution 4 --device cuda --amp \
    --pyramid-cache /outputs/pyramid_cache \
    --eval-only --init-checkpoint /outputs/P1proper_ema6k/oslo_raft_ema.pt \
    --val-sources $S \
    --motion-bands-deg 0,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,inf \
    --output-dir /outputs/a2_$(echo $S | tr ':' '_')
done
```

**Nota de escopo declarada.** O checkpoint final EMA foi selecionado por
*schedule* (último passo), não por pico de val — então para esse número
específico não há pressão de seleção. A pressão de seleção que resta é de
**campanha** (quais estágios e hiperparâmetros foram mantidos), e isso deve ser
declarado como limitação, não medido por este experimento.

### A3 — magnitude × estrutura: baixar o FPS resolveria?

**Pergunta.** "Regime" empacota duas coisas: **magnitude** do deslocamento
(escalar, controlada por FPS) e **estrutura** do campo (esparso/bimodal vs denso/
coerente — não controlada por FPS). O P0d mediu que a estrutura domina 4:1 *na
mesma magnitude sub-pixel* (troca de campo −153 pts vs troca de aparência −36).
Falta a célula vazia do quadrado: **estrutura real com magnitude grande**.

**Instrumento.** `--real-resample-flow-scale k` multiplica o campo GT real antes
da reamostragem. GT e frame 2 derivam do mesmo campo escalado, então a constância
fotométrica é exata em qualquer `k` (validado em §5). Isso varre a magnitude com
a estrutura **fixa** — algo que subamostragem temporal não consegue fazer de
forma controlada, e sem precisar compor GT sobre k frames (que acumula erro e
quebra em oclusão).

Perna 1 — **estrutura real**, magnitude varrida:

```
for K in 1 2 5 10 20; do
SHARDS_HOST=../sfprep/shards OUTPUT_DIR=./outputs \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python run_oslo_raft.py \
    --grid healpix --retina --retina-resolution 7 --resolution 6 \
    --estimation-resolution 4 --device cuda --amp \
    --pyramid-cache /outputs/pyramid_cache \
    --eval-only --init-checkpoint /outputs/P1proper_ema6k/oslo_raft_ema.pt \
    --val-sources flow360:val \
    --val-real-resample-prob 1.0 --real-resample-flow-scale $K \
    --motion-bands-deg 0,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,inf \
    --output-dir /outputs/a3_real_k$K
done
```

Perna 2 — **estrutura coerente** (rotação), mesmas magnitudes, como controle:

```
for D in 0.13 0.26 0.65 1.3 2.6; do
SHARDS_HOST=../sfprep/shards OUTPUT_DIR=./outputs \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python run_oslo_raft.py \
    --grid healpix --retina --retina-resolution 7 --resolution 6 \
    --estimation-resolution 4 --device cuda --amp \
    --pyramid-cache /outputs/pyramid_cache \
    --eval-only --init-checkpoint /outputs/P1proper_ema6k/oslo_raft_ema.pt \
    --val-sources flow360:val \
    --val-synth-rot-prob 1.0 --synth-rot-min-deg $D --synth-rot-max-deg $D \
    --motion-bands-deg 0,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,inf \
    --output-dir /outputs/a3_rot_d$D
done
```

**Leitura.** Plote as duas pernas contra o `target_geo_deg_p50` **medido** (não
contra `k` ou `d` nominais — a rotação tem deslocamento dependente da latitude).

| resultado | conclusão |
| --- | --- |
| curvas **convergem** em magnitude alta | estrutura só importa no sub-pixel ⇒ baixar o baseline temporal é recomendação prática legítima |
| curvas **permanecem separadas** | estrutura é eixo independente da magnitude ⇒ a crítica ao regime vale em cheio |

**Ressalvas a declarar no artigo.** (a) Escalar um campo por `k` não é o que `k`
frames reais produzem — trajetória não-linear e oclusão nova não aparecem; é uma
idealização **favorável**, o que torna o resultado forte se ainda assim falhar.
(b) `bilinear_sample_erp` faz wrap em longitude mas **clampeia** em latitude, de
modo que em `k` alto o frame 2 perto dos polos fica distorcido — trate as colunas
`poles_*` como suspeitas em `k` alto e leia o equador como o sinal limpo.

### A4 — curva de resposta ao deslocamento

**Pergunta.** Em qual deslocamento cada método passa a bater o baseline zero? Esse
número é a fronteira de regime que hoje afirmamos de forma categórica, e que
nenhum paper da área reporta.

**Instrumento.** `--motion-bands-deg` produz bandas **disjuntas** `[lo, hi)`. As
colunas `active_X` existentes são caudas *cumulativas* e por isso misturam regimes
(`active_0.25` num conjunto de grande movimento é dominado pelos pares de 70 px,
não pelos de 0,25°). Bandas são o instrumento correto para localizar o cruzamento.

**Não re-digite os flags.** Cada linha da tabela tem um protocolo próprio (SLOF
roda a `--iters 64 --infer-size 320x640`, a publicada deles; PanoFlow roda em
resolução nativa com CFE; RAFT-large em nativa) e misturá-los silenciosamente
quebra a comparabilidade sem gerar nenhum erro. Use `rerun_from_json.py`, que
relê os `args` gravados no JSON de cada run, aplica os overrides e re-emite o
comando pelo parser real do script que o produziu:

```
# 1) confira o que vai rodar (não executa nada)
SHARDS_HOST=../sfprep/shards OUTPUT_DIR=./outputs \
docker compose -f docker-compose.oslo_raft.yml run --rm oslo-raft \
  python rerun_from_json.py \
    /outputs/universality_slof_raft_test \
    /outputs/universality_slof_raftfinetune_test \
    /outputs/universality_slof_singlerotation_test \
    /outputs/universality_slof_switchrotation_test \
    /outputs/universality_slof_doublerotation_test \
    /outputs/universality_panoflow_test \
    /outputs/<ZERO_flow360_test> /outputs/<RAFT_flow360_test> \
    /outputs/<ZERO_flowscape_test> /outputs/<RAFT_flowscape_test> \
    /outputs/<PANOFLOW_flowscape_test> /outputs/<OSLO_flowscape_test> \
    --set geodesic_metric=haversine \
    --set motion_bands_deg=0,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,inf \
    --output-suffix _hav

# 2) execute (mesma linha + --run)
... --output-suffix _hav --run
```

Os nomes acima seguem a convenção registrada em §4 de `UNIVERSALITY_TABLE.md`
(`universality_slof_<ck>_test`); confirme com
`docker compose ... run --rm oslo-raft ls -1 /outputs` e substitua os
`<PLACEHOLDER>`. Se algum `--set` não existir no script daquele run, a ferramenta
aborta o lote inteiro com erro claro em vez de rodar meia tabela.

Para o comando de arquivo (que sobrevive a mudanças futuras de default), passe
`--explicit`.

**A linha do OSLO em flow360:test já está feita** — é o próprio run do A1.2,
com `haversine` + bandas. Restam 12.
**Figura resultante.** Eixo x = `band_*_zero_geo_deg` (deslocamento GT médio da
banda). Eixo y = `band_*_improvement_pct`. Uma linha por método, pontos dos dois
datasets no mesmo eixo. O cruzamento de y = 0 é o número novo.

Se todos os métodos cruzarem aproximadamente no mesmo ponto, a universalidade
deixa de ser uma tabela de 11 linhas e vira **uma lei empírica** — é essa a
contribuição-título sugerida para o artigo.

---

## 3. Balde B — propor no artigo (pré-registrado, não executado)

### B1 — gate estático/móvel *(carro-chefe)*

Saída por nó = `(flow, logit)`; fluxo final = `σ(logit) · flow`.

Por que é forte, e não mais uma heurística:

1. **O modelo contém o baseline por construção.** Com σ≡0 recupera-se o zero-flow
   exatamente. Um modelo que nesta formulação perde para o zero está mal treinado,
   não mal especificado. Deixa-se de competir com o baseline e passa-se a
   generalizá-lo.
2. **Casa com a estrutura medida no P0d.** O campo real é "majoritariamente
   estático + paralaxe esparsa"; a decomposição explícita é a fatoração correta.
3. **Desacopla a tensão que travou o P1.** Hoje `--loss-motion-weight` melhora
   actives e destrói o global — o mesmo parâmetro puxando em direções opostas. O
   gate separa "onde há movimento" de "qual é o movimento".
4. **Custo baixo:** uma cabeça + termo de loss, warm-start de
   `P1proper_ema6k`, ~5–10k passos, sem dados novos.

**Critério de sucesso, declarado a priori:** global **e** act₀.₅ ambos ≥ 0 em
flow360:**test**. Seria o primeiro método, de qualquer laboratório, a bater o zero
nos dois eixos naquele split.

**Risco e mitigação declarados.** O gate pode colapsar em σ≈0, reproduzindo o zero
exatamente (+0,0%): empate, não vitória, e não prova nada. Reportar junto:
(a) estatística de σ — tem que ser **bimodal**, não constante; (b) AUC de σ como
detector de `|GT| ≥ 0,25°`; (c) a ablação de correlação tem que continuar
colapsando (0,28° → 54°), provando que o fluxo dentro do gate é casamento genuíno.
Se colapsar, isso também é resultado: a evidência de movimento não é extraível
nesse regime, e a caracterização fecha.

### B2 — capacidade e receita de treino

A sonda de piso de grade exonerou a resolução: OSLO está **2,2×** (replica) a
**4,9×** (flow360) *acima* do próprio teto arquitetural. A folga é real e não é
resolução.

| fator | OSLO hoje | PanoFlow |
| --- | --- | --- |
| parâmetros | 1,56 M | 5,63 M |
| iterações | 8 treino / 12 eval | 20 |
| batch | **2** | 6–12 típico |
| schedule | lr constante + anneal | OneCycle longo |

**Registrar que a acumulação de gradiente já falhou** (accum-8, EMA act₀.₅ em
queda monótona +5,2 → +0,1): o próximo ataque precisa ser batch *genuíno* —
mixed precision, checkpointing de ativação ou multi-GPU — não acumulação.

Ganho esperado: se o OSLO chegar a ~10 pts do PanoFlow no flowscape **preservando**
a vantagem polar, a alegação vira "arquitetura esférica nativa iguala o SOTA ERP
globalmente com 3,6× menos parâmetros e é mais uniforme". Risco: pode não
preservar a uniformidade ao ganhar global.

### B3 — ablação mix-menos-chairs360

Foi deferida. Não bloqueia (o dataset é contribuição como artefato
independentemente de ter ajudado o treino), mas é honestidade de atribuição para
a tese final: hoje não sabemos o que os 23,5k pares contribuíram.

### B4 — reformulação

- **Baseline temporal longo (t → t+k)** — só faz sentido *depois* do A3, que diz
  se magnitude é o eixo. Verificar antes se o flow360 permite compor GT sobre k
  frames sem erro proibitivo.
- **Segmentação de movimento em vez de fluxo denso** — se ninguém bate o zero no
  fluxo por-pixel, talvez a saída correta nesse regime seja a máscara de movers.
  O gate do B1 já é meio caminho.

---

## 4. Balde C — já concluído (entra como resultado)

1. **Tabela de universalidade** — 11 linhas, uma harness, uma métrica, split
   limpo; incluindo *descobrir e reportar* o vazamento de split (nossa val ⊂
   treino do SLOF).
2. **Figura de contraste de regime** — 3 checkpoints, pesos idênticos, swings de
   103/89/80 pts. A evidência mais forte da tese.
3. **Quadrado de decomposição do P0** — a parede é o campo de movimento (−153 pts),
   não a aparência (−36), com inversão de sinal.
4. **P1a: o campo real é aprendível** — −72,5% → +42,7%.
5. **P1b: primeiros actives reais positivos**, com controle de ablação de
   correlação (0,28° → 54°).
6. **OSLO-RAFT-R**: vantagem polar + uniformidade vs arquitetura de perspectiva,
   **replicada em dois datasets independentes** (replica360 polos 2,88° vs 3,65°,
   2,3× mais plano; flowscape:test 3,147° vs 3,650°, 2,4×), com 3,4× menos
   parâmetros.
7. **chairs360**: gerador + 23,5k pares com verificação de warp.
8. **sfprep**: harness de dados com formato único e adaptadores para 5+ datasets.
9. **Sonda de piso de grade**: refutou a hipótese da resolução **por medição**,
   poupando uma campanha inteira.
10. **Piso numérico da métrica**: descoberto, quantificado, escopo declarado.
11. **Solução de variância por EMA**: σ 8,6 → 0,9 (9×).

Os itens 9 e 10 merecem destaque: são casos em que o trabalho **fechou uma porta
com medição em vez de argumento**.

---

## 5. Registro de validação local (Docker, CPU, 2026-07-28)

Rodado com o repo montado sobre a imagem `oslo-raft:cuda`
(`-v $PWD:/workspace/oslo -e PYTHONPATH=/workspace/oslo`), sem rebuild.

**A1 — piso confirmado e removido.** 200k pares de vetores unitários
**idênticos**: `acos` dá média 0,0281° / máx 0,0442°, batendo a previsão teórica
`sqrt(2·eps_f32) = 0,02798°`; `haversine` dá **exatamente 0**. Antipodal sob
`haversine` = 180,0000° (fórmula válida em toda a faixa). Modo inválido rejeitado.

**Contaminação medida por ângulo verdadeiro** (40k pares por linha, par
construído por rotação exata de `a` na direção de `t` ⊥ `a`, de modo que
`haversine` é ground truth por construção):

| ângulo real | leitura `acos` | viés | rms |
| --- | --- | --- | --- |
| 0,010° | 0,02809° | +180,87% | 181,10% |
| 0,028° | 0,03095° | +10,55% | 19,68% |
| 0,050° | 0,04818° | −3,63% | 11,53% |
| 0,100° | 0,10085° | +0,85% | 2,71% |
| **0,250°** | 0,25023° | **+0,09%** | **0,42%** |
| 0,500° | 0,50007° | +0,01% | 0,10% |
| 1,000° | 0,99995° | −0,01% | 0,03% |
| ≥ 3° | exato | 0,00% | 0,00% |

**Consequência.** No limiar dos actives (0,25°) o viés é +0,09% — os actives da
tabela de universalidade estão limpos, agora por medição e não por estimativa.
Isto também **corrige a heurística "61% de piso"** registrada em §8b de
`UNIVERSALITY_TABLE.md`: aquele número vinha da razão linear 0,028/0,046, que não
é como o erro se compõe. A contaminação real depende da *distribuição* de erros
(uma distribuição com massa perto de zero é inflada muito mais que o teste de
ângulo fixo sugere — vetores idênticos leem 0,028° de um valor verdadeiro 0). Por
isso A1.3 mede em vez de estimar; nenhum número deve ser anunciado antes disso.

**A4 — bandas.** `parse_bands` aceita `inf`, rejeita menos de duas arestas e
arestas não-crescentes. As bandas **particionam** exatamente o conjunto válido
(soma das contagens = total). Com erro constante de 0,30° injetado, a melhoria
por banda troca de sinal exatamente onde o zero cruza 0,30° (−140,4% / +19,97% /
+60,00% / +80,00%), confirmando o cálculo. Demonstrada a diferença que motiva a
mudança: a cauda cumulativa `active_0.25` tem zero médio 1,1250° contra 0,3748°
da banda `[0,25; 0,5)` — a cauda está contaminada por movimento grande, a banda
não. `accumulate_maps` (streaming) bate com `summarize_maps` (one-shot) em todas
as bandas.

**A3 — escala do campo.** `ShardFlowDataset` aceita `real_resample_flow_scale`
(default 1,0, ou seja, runs existentes ficam bit-idênticos). A escala é
exatamente linear através do amostrador ERP em k = 1, 5, 20 — GT e frame 2 saem
do mesmo tensor escalado, então a constância fotométrica é preservada em qualquer
escala.

**CLI.** As 7 linhas de comando exatas do §2 foram parseadas pelos `parse_args()`
reais dos três scripts. Defaults inalterados (sem bandas, `acos`, escala 1,0) ⇒
runs existentes reproduzem.

### 5.1 Smoke end-to-end em dados reais (flow360:test, 40 pares, CPU)

Rodado com `--predictor zero` nas duas métricas. Dois achados que não eram
previsíveis do teste sintético:

**(a) O piso está confinado à banda mais baixa** — e é lá que mora quase metade
da esfera:

| banda | ocupação | zero `acos` | zero `haversine` | inflação |
| --- | --- | --- | --- | --- |
| [0; 0,0625) | **44,5%** | 0,0334° | 0,0208° | **+60,6%** |
| [0,0625; 0,125) | 17,8% | 0,0884° | 0,0898° | −1,6% |
| [0,125; 0,25) | 14,3% | 0,1777° | 0,1785° | −0,4% |
| [0,25; 0,5) | 12,9% | 0,3500° | 0,3502° | −0,06% |
| [0,5; 1) | 8,1% | 0,6875° | 0,6875° | 0,00% |
| [1; 2) | 1,4% | 1,2989° | 1,2989° | 0,00% |
| [2; 4) | 0,6% | 2,7763° | 2,7764° | 0,00% |
| [4; 8) | 0,28% | 5,5593° | 5,5595° | 0,00% |
| [8; 16) | 0,13% | 10,3340° | 10,3340° | 0,00% |
| [16; 32) | 0,10% | 22,9529° | 22,9529° | 0,00% |
| [32; ∞) | 0,01% | 71,9626° | 71,9626° | 0,00% |

Confirma por medição em dados reais o que §5 mostrou em ângulos controlados: tudo
**acima de 0,0625° está limpo**, e o global (0,2482 → 0,2537, +2,2%) é inflado
só porque 44,5% dos nós estão na banda contaminada. As arestas de banda escolhidas
também ficam validadas: distribuem a massa de forma útil e o flow360:test de fato
alcança a cauda ≥ 32° (0,01% dos nós).

**(b) Sob `acos`, o preditor zero "vence" a si mesmo.** O `--predictor zero` é,
por construção, idêntico ao baseline — a melhoria tem que ser exatamente 0. Sob
`haversine` é: `global_improvement_pct = 0,0000`. Sob `acos` é **+0,12% global**
(actives +0,0102 / +0,0036 / +0,0006).

Implicação direta para a tabela de universalidade: a linha **SLOF doublerotation
(+0,03 global)** está **dentro** desse artefato numérico — seu sinal positivo no
global não é distinguível do ruído da própria métrica. Já o act₀.₅ dela (+0,12)
está ~30× acima do artefato (+0,0036), logo é real, ainda que desprezível. A
frase correta na tese é: *o único row não-negativo é um preditor-de-zero trivial,
e mesmo seu positivo global é artefato de métrica.* Confirmar em escala completa
no A1.2 (este smoke é de 40 pares; o efeito é viés, não ruído de média zero, então
não deve encolher com mais pares — mas isso precisa ser verificado, não assumido).

Scripts: `scratchpad/validate_roadmap.py` → `ROADMAP_LEVERS_VALIDATION_PASSED`;
`scratchpad/validate_cli.py` → `CLI_VALIDATION_PASSED`.

---

## 5.2 A1.1 EXECUTADO na box (2026-07-28) — e o que ele revelou

**Resultado direto.** Identity check (`--estimation-resolutions 6`), flow360:test,
20 pares:

| | `acos` | `haversine` | ganho |
| --- | --- | --- | --- |
| identity global | 0,0280° | 0,00066° | 42× |
| identity actives (≥ 0,25°) | 0,0280° (travado) | 0,0000021° | ~13.000× |
| como % do zero (0,1814°) | 15,4% | 0,36% | — |

O piso do `arccos` **na métrica** acabou. Prova: sob `acos` os actives não podiam
ler abaixo de 0,028° por construção; agora leem 2,1e-6°.

**As bandas localizaram o resíduo restante em uma única faixa.** `pwc`:

| banda | ocupação | erro |
| --- | --- | --- |
| **[0; 0,0625)** | **50,3%** | **0,0013025°** |
| todas as demais (0,0625 → ∞) | 49,7% | ~0,0000021° |

`0,5029 × 0,0013025 = 0,000655` = exatamente o global 0,000656 ⇒ a banda mais
baixa é **100%** do resíduo. Sinal decisivo: essa banda tem o *menor*
`tangent_epe_rad` de todas (2,5e-11) e o *maior* erro geodésico — ou seja, o fluxo
tangente está exato e a perda está na **conversão**.

### Causa-raiz: `logmap` usava o mesmo `arccos`, dentro da construção do GT

```python
# geometry.py:207-208 (ANTES)
dot = (base * endpoints).sum(-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)  # eps=1e-8
theta = torch.acos(dot)
tangent_3d = endpoints - dot * base     # cancelamento catastrófico p/ theta pequeno
```

Dois defeitos: (a) `acos` tem a mesma resolução de `sqrt(2·eps_f32)`; (b) o clamp em
`1 − 1e-8` impunha um **piso duro de `acos(1 − 1e-8) = 0,0081°` em qualquer
magnitude de fluxo GT**. Além disso `endpoints − dot·base` é diferença de vetores
quase iguais O(1) — perde ~4 dígitos em float32.

Corrigido com a forma de corda mais a projeção a partir de `delta = e − p`
(identidade: `delta − (delta·p)p = sin θ · u`, exato). Medido:

| ângulo real | erro antigo | erro novo | ganho |
| --- | --- | --- | --- |
| 0,001° | 6,81e-4° | 1,74e-6° | 392× |
| 0,010° | 6,85e-3° | 1,24e-6° | 5.544× |
| 0,023° | 7,06e-3° | 1,80e-6° | 3.929× |
| 0,0625° → 90° | ~2e-6° | ~2e-6° | 1× (sem regressão) |

O erro novo é **plano** em 0,001–90° (~1e-6°), assinatura de estar no piso de
representação do float32 e não num piso algorítmico.

### Achado colateral: o preditor `zero` nunca foi exatamente zero

`--predictor zero` gera fluxo ERP nulo que passa por `erp_flow_to_tangent` →
`logmap`. Com o código antigo, `logmap(p, p)` retornava **0,0081°** numa direção
numericamente arbitrária, em vez de 0. Verificado após o conserto:
`logmap(p, p) → max |flow| = 0,000e+00`.

**O que isso NÃO muda.** O baseline zero de toda alegação publicada não vem do
preditor: `compute_maps` monta `zero_endpoint = points.expand` diretamente
(`metrics.py:90`), sem passar por `logmap`. E a loss de treino usa
`sequence_geodesic_loss(preds, batch["endpoint"], …)` — endpoint exato. Escopo
verificado por varredura: dentro de `spherical_flow/`, o único consumidor do
`flow` derivado é `metrics.py:89` (`tangent_epe_rad`, métrica secundária).

⇒ **Nenhum número headline e nenhum treino foi afetado.** O que estava
contaminado: `tangent_epe_rad`, o alvo de reconstrução da probe, e a linha de
controle `--predictor zero` (que é 0% por definição e só serviu de sanity check).

### Piso de medição do harness — valor a declarar na tese

Não é 0,028° (métrica antiga) nem 0,0007° (contaminado pelo `logmap` antigo):
é **~2e-6°**, cinco ordens de grandeza abaixo do sinal do flow360. A métrica
deixou de ser fator limitante em qualquer regime.

---

## 5.3 A1.2 — a primeira curva de resposta ao deslocamento (OSLO, flow360:test)

O run do A1.2 já traz bandas, então é também a primeira linha do A4. OSLO EMA
final, 2567 pares, `haversine`:

| banda | ocupação | zero | OSLO | melhoria |
| --- | --- | --- | --- | --- |
| [0; 0,0625) | 32,8% | 0,0167° | 0,1117° | **−570,3%** |
| [0,0625; 0,125) | 15,8% | 0,0926° | 0,1366° | −47,5% |
| [0,125; 0,25) | 16,9% | 0,1798° | 0,2297° | −27,8% |
| [0,25; 0,5) | 16,6% | 0,3569° | 0,3915° | −9,7% |
| **[0,5; 1,0)** | **12,2%** | 0,7026° | 0,6989° | **+0,53%** |
| [1; 2) | 3,5% | 1,2783° | 1,3737° | −7,5% |
| [2; 4) | 1,0% | 2,8380° | 3,0836° | −8,7% |
| [4; 8) | 0,7% | 5,6302° | 6,0986° | −8,3% |
| [8; 16) | 0,3% | 10,666° | 11,597° | −8,7% |
| [16; 32) | 0,11% | 21,953° | 22,471° | −2,4% |
| [32; ∞) | 0,11% | 59,460° | 60,057° | −1,0% |

**O cruzamento não é um limiar, é uma janela.** O OSLO só supera o zero em
`[0,5; 1,0°)`, e volta a perder acima disso. Isso é mais informativo do que a
hipótese original (limiar monotônico) e a tabela agregada escondia por completo.

Duas leituras que a curva entrega:

1. **O colapso abaixo de 0,25° é quantificado pela primeira vez** — −570% na banda
   mais baixa, que é 33% da esfera. É exatamente a calibração estática, e é o alvo
   declarado do gate do B1: a métrica pela qual o gate deve ser julgado passa a ser
   *esta banda*, não só o global agregado.
2. **O platô de −8% entre 1° e 16° é plano demais para ser falha de resolução.**
   Erro de grade cresceria com o deslocamento; erro relativo constante aponta viés
   sistemático de escala. Registrado como observação, **não medido** — não
   interpretar sem uma medição dedicada.

Se as demais linhas mostrarem janelas em posições diferentes, a figura fica mais
forte que o "todos cruzam no mesmo ponto" que eu havia antecipado.

---

## 6. Esqueleto do artigo

| seção | conteúdo | balde |
| --- | --- | --- |
| Problema | fluxo óptico 360° é avaliado num regime; o benchmark de vídeo real está em outro | — |
| Resultados obtidos | tabela de universalidade, contraste de regime, decomposição P0, OSLO | C |
| Rigor metodológico | vazamento de split, piso da métrica, exoneração da grade | A1 + C9/C10 |
| Caracterização | curva de resposta ao deslocamento, magnitude × estrutura | A3, A4 |
| Limitações | gap val→test decomposto | A2 |
| Trabalho proposto | gate, capacidade, ablações — com critérios a priori | B |
| Cronograma | ordenado por custo | — |

**Frase-escopo a usar (a versão ampla é indefensável).** Não escreva "aplicações
reais estão em outro regime" — vídeo 360 de drone ou veículo a 30 fps tem
deslocamento substancial. Escreva:

> O único benchmark 360° de vídeo real com GT (FLOW360) instancia um regime onde
> o baseline trivial é imbatível, e nenhum trabalho publicado reporta esse
> baseline.

O problema metodológico não é qual regime é "o real" — é que **o número publicado
não informa de que lado do cruzamento você está**. É exatamente isso que a curva
do A4 mede.

---

## 7. Ordem e esforço

| ordem | bloco | esforço | compra |
| --- | --- | --- | --- |
| 1 | A1 | ~2–3 h de GPU | blindagem: decide a métrica antes de gastar o resto |
| 2 | A4 + A2 | ~4–6 h de GPU (11 evals) | a figura-título e a decomposição do gap |
| 3 | A3 | ~2 h de GPU (10 evals) | preempta a objeção "não é só FPS?" |
| 4 | B1 | ~1 semana | única chance de um positivo inequívoco — **propor, não executar** |

A1 + A4 + A2 + A3 já entregam uma boa dissertação: rigor metodológico mais uma
contribuição mensurável que o campo não tem. B1 é o que separa "boa" de "forte", e
o custo é baixo o bastante para valer a aposta mesmo com risco de colapso do gate.
B2 só se sobrar tempo — melhora um número, não muda a natureza da contribuição.
