# Relatório final — pipeline de preparação de dados (pouso C172P / BIKF)

Gerado em 2026-09-17. Pipeline implementado em Python puro (`pipeline/` + `scripts/`),
executado sobre os 15 CSVs reais em `voos-piloto/`. Nenhum arquivo original foi
alterado, movido ou apagado.

## 1. O que foi encontrado

### 1.1 Inventário
15 CSVs brutos, ~247 colunas cada, frequência efetiva mediana **60.0 Hz** em todos
os arquivos (`sim_time` monotônico, sem duplicatas, poucos gaps pequenos — ver
`data/manifests/file_timing_audit.csv`).

### 1.2 Formato heterogêneo (achado não previsto pela spec)
13 dos 15 arquivos usam `,` como delimitador e `.` como separador decimal.
**Dois arquivos** (`voo-com-rajada.csv`, `voo-normal2.csv`) usam `;` como
delimitador e `,` como separador decimal (exportação em locale pt-BR/europeu).
Detecção automática por arquivo implementada em `pipeline/io_utils.py`
(`detect_format`), nunca por nome de arquivo.

### 1.3 Offset do sensor de AGL
`/position[0]/altitude-agl-ft[0]` nunca chega a 0 no touchdown — estabiliza em
~3.4–3.6 ft mesmo com os trens principais em contato contínuo. Interpretado
como a altura geométrica do datum/CG da aeronave acima do ponto de contato das
rodas (offset físico do C172P), não um erro de sensor. Documentado e usado
para validar `threshold_elevation_ft`.

### 1.4 Mapeamento dos trens de pouso (C172P)
Confirmado empiricamente, não assumido: `gear[0]` é o **trem de nariz**
(único com `steering-norm` variável); `gear[1]` e `gear[2]` são os **trens
principais** (`steering-norm` constante = 0 em todos os arquivos). Usado em
toda a segmentação de pousos.

### 1.5 circuito-completo.csv: dois pousos, dois cenários de detecção
Segmentado com sucesso em `circuito-completo__landing-01` (t≈0–131s, touch-and-go
detectado via critério de liftoff, não apenas groundspeed) e
`circuito-completo__landing-02` (t≈297–437s, preparação de aproximação após o
circuito completo). Ambos compartilham `session_id=circuito-completo`.

### 1.6 ACHADO IMPORTANTE: o segundo pouso do circuito usa uma pista física diferente
Ao investigar por que a geometria along/cross-track de `landing-02` ficava sem
sentido (cross-track na casa dos milhares de metros, heading_error próximo de
±90–180°) usando a config de pista RWY02/20, foi constatado que:
- o heading da aeronave em touchdown/rollout do segundo pouso é estável em
  ~264–269° verdadeiros;
- a posição de touchdown está a ~634 m da cabeceira 29 (par de pistas 11/29 do
  apt.dat local do FlightGear), não perto da 02/20.

**Conclusão**: o segundo pouso do circuito foi realizado na pista 29 (o outro
par de pistas de BIKF), não na mesma pista 02/20 usada pelos outros 14
episódios. Criada uma segunda configuração `data/config/runway_29.yaml`
(threshold e heading derivados geometricamente do apt.dat local), usada
*apenas* para computar as features relativas à pista de
`circuito-completo__landing-02` (`runway_id="29"` em `episodes.csv`). As
*definições* das features (along_track, cross_track, etc.) são idênticas;
apenas a referência de pista muda por episódio. **Sinalizado para revisão
humana** apesar da evidência geométrica ser forte e consistente (heading +
posição concordam independentemente).

### 1.7 BIKF "RWY 01" vs apt.dat atual
O apt.dat local do FlightGear não tem mais uma pista "01" em BIKF — foi
renomeada para "02" (deriva de variação magnética, evento real ~2013). A
geometria física (threshold lat/lon, true heading) dos valores fornecidos na
spec bate com a cabeceira "02" do apt.dat a ~5 casas decimais, então a
geometria foi mantida e só o rótulo textual "RWY 01" foi tratado como legado.
`runway_length_m`: spec deu 3054 m; distância geométrica calculada do apt.dat
= 3046.4 m (diff 0.25%, ambos registrados). `threshold_elevation_ft=135`
confirmado empiricamente via altitude MSL no touchdown de 3 pousos retos menos
o offset do sensor AGL (seção 1.3). Ver `data/config/runway.yaml` para a
análise completa.

### 1.8 Pousos "ruidosos" em vento de 16 kt
`voo-3-com-16-nos` e `voo-2-com-16-nos` fazem múltiplos toques breves
(porpoising) antes de assentar — nunca satisfazem "0.3s contínuos com os dois
trens principais em contato" simultaneamente. Implementado um detector de
touchdown em 3 níveis (`pipeline/segment_circuit.py::segment_one_landing`):
(1) os dois trens principais por ≥0.3s: (2) fallback — qualquer trem principal
sozinho por ≥0.3s; (3) fallback final — usa o próprio critério de ROLLOUT
(≥0.8s de contato de qualquer trem) como o instante de touchdown. Todos os 15
episódios elegíveis foram segmentados com sucesso; o nível usado é registrado
em `touchdown_confirmation_tier` (ver `pipeline/segment_circuit.py`).

Também foi necessário reforçar a detecção de COMPLETE por "liftoff" (usada só
para o touch-and-go de `landing-01`): a versão inicial confundia um simples
solavanco (bounce) com um verdadeiro arremeeter/decolagem, porque um bounce
também produz `AGL` subindo por >1s. Corrigido exigindo também throttle alto
(≥0.6) OU groundspeed aumentando (>2 kt) durante o intervalo sem contato —
throttle parado em 0.35 com groundspeed caindo (como em `voo-3-com-16-nos`)
não conta mais como liftoff.

### 1.9 Cenários confirmados por dados, não só pelo nome do arquivo
`wind-speed-kt` médio e `turbulence-magnitude-norm` máximo foram calculados
por arquivo (`data/manifests/episodes.csv`, colunas `qa_*`). Todos batem
qualitativamente com o nome do arquivo:

| scenario_id | arquivos | wind médio (kt) | turb máx |
|---|---|---|---|
| normal | 3 | ~3.5 | ~0.006 (ruído basal) |
| wind_8kt | 2 | ~9.3 | ~0.006 |
| wind_16kt | 3 | ~18.4–19.0 | ~0.006 |
| gust / gust_direction_variant | 3 | ~12.5–12.7 | ~0.006 |
| turbulence_05 / _08 | 2 | ~3.5 | 0.25 / 0.64 |
| circuit | 1 (2 episódios) | ~3.6 | ~0.006 |

**Divergência não resolvida, registrada**: os arquivos "8 nós"/"16 nós" têm
vento médio real de ~9.3 kt / ~18.4–19.0 kt, não exatamente 8/16 kt — mantido
o rótulo do nome do arquivo (a intenção original do piloto), mas o valor
numérico real está documentado. A tentativa de confirmar
`voo-com-rajada-vindo-da-direita` como vindo especificamente "da direita" via
heading relativo médio do voo inteiro não foi conclusiva (o heading médio de
todo o voo é dominado por trechos fora da final; ambos os arquivos de rajada
mostram ~-80° de vento relativo). Mantido o rótulo do nome do arquivo,
sinalizado como não confirmado por dados.

## 2. O que foi alterado / criado

Estrutura completa `data/` criada conforme especificação (seção 2), com dados
processados em 6 estágios reprodutíveis (`scripts/01_...` a `scripts/11_...`).
Nenhuma transformação sobrescreve dados de estágio anterior; cada estágio
produz arquivos novos.

### 2.1 `voo-inicio-teste.csv`
Identificado por correspondência tolerante (case/hífen/underscore/extensão,
`pipeline/io_utils.py::is_pilot_familiarization_file`). Preservado em
`data/raw/`, `eligible_for_training=false`, `exclusion_reason=pilot_familiarization`
em todos os manifestos. Passou pelos estágios de parsing/resampling/features
*apenas* para teste técnico do parser (`data/processed/excluded/`,
`data/features/unnormalized/voo-inicio-teste.csv`), mas com 0 linhas em
`baseline_core`/`full_approach`/`recovery_ablation` e ausente de todos os
folds (testado em `tests/test_pipeline.py::test_pilot_familiarization_excluded_from_folds`).

### 2.2 Colunas: 247 → 14 neural_input + 4 label + 3 derived_feature_source
Ver `data/manifests/columns_catalog.csv` (uma linha por coluna) e
`data/manifests/exclusions.csv`. Distribuição de papéis:

| role | n colunas |
|---|---|
| excluded | 193 |
| qa_only | 20 |
| neural_input | 14 |
| flight_manager_only | 11 |
| label | 4 |
| derived_feature_source | 3 |
| metadata | 2 |

Motivos de exclusão: 129 constantes/não aplicáveis ao C172P (ex.: rotores,
esquis, flutuadores, gear[19..22] — claramente de um template genérico
multi-aeronave do FlightGear), 30 de famílias proibidas pela seção 10
(freios, yoke, pedais, luzes, etc.), 25 fora do schema v1 das duas MLPs
(mach, acelerações, instrumentação), 9 duplicatas confirmadas.

### 2.3 Duplicatas confirmadas (`data/manifests/duplicate_columns.csv`)
- `altitude-agl-m` = `altitude-agl-ft * 0.3048`, com um **atraso de exatamente
  1 amostra** entre as duas propriedades (erro residual ~5e-6 m após alinhar
  o lag — artefato de ordem de atualização da property tree do FlightGear na
  mesma linha do CSV). Descartada `altitude-agl-m`.
- `speed-down-fps` = `-vertical-speed-fps` (sinal invertido, componente NED).
  Descartada.
- `engine[1]/throttle` idêntico a `engine[0]/throttle` (C172P é monomotor).
  Descartada.
- 5 pares de colunas exatamente byte-idênticas, incluindo um achado curioso:
  `/engines[0]/engine[7]/{rpm,n1,n2}` são byte-idênticos a
  `/gear[0]/gear[{0,1,2}]/rollspeed-ms[0]` — quase certamente um artefato de
  aliasing/geração de propriedades do logger para um índice de motor
  inexistente (C172P só tem `engine[0]`). Ambos os lados já eram excluídos
  por outros motivos; documentado por transparência.
- **Candidatos verificados e NÃO confirmados como duplicatas** (mantidos
  separados, com nota): `airspeed-kt` vs `vias-kts` (correlação ~0.99 mas com
  spike de inicialização e erro residual real; tratada como medida distinta,
  não usada como input) e `side-slip-rad` vs `beta-deg` (`beta-deg` é
  **constante = 0** em todos os arquivos — propriedade morta/não usada pelo
  FDM, não uma representação alternativa válida do sideslip).

### 2.4 Transientes de inicialização (seção 6)
Detectados automaticamente para as 14 aproximações padronizadas (13 pousos
retos + `landing-01`) usando: airspeed em [55,75] kt, flaps não-nulo e
estável (desvio-padrão móvel de 1s < 0.01), trajetória inbound
(`along_track_m<0`), aproximação restante suficiente (≥5s), sustentado por
≥2s contínuos, busca limitada a 30s do início do episódio. **Todos os 14
transientes foram detectados automaticamente** (nenhum precisou manter o
episódio completo por falha de detecção) — ver
`data/manifests/transient_diagnostics.csv` para duração removida, airspeed e
flaps antes/depois, distância à pista e AGL no fim do transiente, por
episódio. Durações removidas variam de ~1.6s a ~13.8s.

### 2.5 full_approach vs baseline_core do segundo pouso (seção 5)
`APPROACH_MANEUVERING` (t≈296.7s) e `FINAL_APPROACH_ESTABLISHED` (t≈320.3s)
detectados automaticamente por combinação de heading_error, cross-track,
along-track e taxa de descida, usando a config `runway_29.yaml` (ver 1.6).
`full_approach` preserva os ~140s completos da preparação de aproximação;
`baseline_core` usa só os ~117s finais a partir de `FINAL_APPROACH_ESTABLISHED`.
Ver `data/reports/figures/variant_comparison_circuito-completo__landing-02.png`.

### 2.6 Reamostragem para 10 Hz
Filtro causal Butterworth de 2ª ordem, corte 4 Hz (`scipy.signal.lfilter`,
nunca `filtfilt`), atraso de grupo estimado em **~57.9 ms** no corte.
Sinais discretos (WOW dos 3 trens, flaps — ver nota abaixo) mantidos por
retenção do último valor, sem filtragem. Amostras da grade de 10 Hz obtidas
por retenção causal do valor filtrado mais recente (nunca por interpolação
linear entre uma amostra passada e uma futura, que violaria "sem usar
amostras futuras"). Verificado: grade uniforme de 10 Hz em todos os 16
episódios (`tests/test_pipeline.py::test_resampled_is_10hz`).

**Ajuste feito durante a implementação**: `flaps` foi originalmente tratado
como sinal contínuo filtrado, mas isso produzia *ringing* do filtro
(overshoot de até +1.4%/-1.1% fora de [0,1]) nas transições entre
detentes. Como a seção 12 da spec já agrupa flaps com WOW como transição a
preservar, `flaps` passou a ser tratado como sinal retido (sem filtro),
eliminando o artefato. Da mesma forma, colunas de controle fisicamente
limitadas (`throttle`, `aileron`, `elevator`, `rudder`, trims, flaps,
speedbrake, spoilers) são recortadas para seu intervalo físico válido
*depois* da filtragem, porque o filtro causal pode ultrapassar levemente o
limite físico em transições abruptas (ex.: throttle chegando a -0.014) —
isso é um artefato de filtragem, não um valor real, e a spec exige que
throttle permaneça em [0,1].

### 2.7 Features (seções 14–15)
Schema fixo em `data/features/schema.json`. Rede lateral: 11 entradas + 2
rótulos. Rede longitudinal: 12 entradas + 2 rótulos. `elevator_effective =
elevator_command + elevator_trim` (trim nunca é usado como entrada
isoladamente). `glideslope_error_deg`: o ângulo bruto (atan2) diverge para
perto de ±180° quando a aeronave passa da cabeceira (o triângulo
geométrico degenera); resolvido retendo o último valor válido quando a
distância à cabeceira cai abaixo de 50 m (configurável em
`preprocessing.yaml: glideslope_feature.floor_distance_to_threshold_m`),
análogo ao comportamento real de um receptor de glideslope ILS perto da
cabeceira.

### 2.8 Splits e scalers (seções 16–17)
14 sessões elegíveis. Holdout diagnóstico fixo escolhido
deterministicamente por `sha256(session_id)` entre as 3 sessões `wind_16kt`
→ **`voo-2-com-16-nos`** (as outras duas, `voo-3-com-16-nos` e
`voo-com-16-nos`, permanecem no rodízio de treino/validação). 13 folds
leave-one-session-out sobre as 13 sessões restantes (os dois pousos de
`circuito-completo` sempre juntos, testado). Scaler por fold calculado
*apenas* nas sessões de treino daquele fold (0 variáveis com
desvio-padrão degenerado em qualquer fold). Scaler `final_all_valid`
calculado sobre as 14 sessões elegíveis (incluindo o holdout diagnóstico),
marcado `purpose=final_training_after_model_selection`, nunca usado para
validação offline.

## 3. Decisões automáticas (resumo)

- Delimitador/decimal/encoding detectados por arquivo, não hardcoded.
- Mapeamento dos trens (nariz vs principais) inferido de `steering-norm`.
- `touchdown_confirmation_tier` com fallback em 3 níveis para pousos
  turbulentos.
- Critério de liftoff/touch-and-go reforçado com throttle/groundspeed para
  não confundir bounce com decolagem real.
- `runway_id` por episódio (`02` para 14 episódios, `29` só para
  `circuito-completo__landing-02`).
- `glideslope_error_deg` com retenção causal abaixo do piso de 50 m.
- `flaps` e controles limitados fisicamente recortados após a filtragem.

## 4. Decisões que exigem revisão humana

1. **RWY29 para o segundo pouso do circuito** (seção 1.6) — evidência
   geométrica forte (heading + posição), mas contradiz a suposição de pista
   única da spec. Recomenda-se confirmação com o log de voo do piloto ou o
   plano de voo do circuito.
2. **`threshold_elevation_ft` da pista 29** (150 ft) é uma estimativa
   empírica (não há elevação por cabeceira no apt.dat nesse formato) — ver
   nota em `data/config/runway_29.yaml`.
3. Direção do vento em `voo-com-rajada-vindo-da-direita` não confirmada por
   dados agregados (seção 1.9).
4. `wind_8kt`/`wind_16kt`: valor médio real diverge levemente do nome do
   arquivo (~9.3/~18.7 kt vs 8/16 kt nominais).
5. `runway_length_m`: 3046.4 m (geométrico, apt.dat) vs 3054 m (spec) — usada
   a primeira, ambas registradas.

## 5. Arquivos gerados

Ver árvore completa em `data/` (config/, manifests/, validated/, processed/,
features/, splits/, scalers/, reports/). Principais entregáveis para
treinamento: `data/features/baseline_core/*.csv` (dataset principal),
`data/features/full_approach/*.csv` e `data/features/recovery_ablation/*.csv`
(experimentos futuros), `data/splits/normalized_folds/fold_*_{train,val}.csv`
(prontos, já normalizados por fold), `data/scalers/folds/*.json`,
`data/scalers/final_all_valid/*.json`, `data/features/schema.json`.

## 6. Como reproduzir

```bash
cd voos-piloto
python3 scripts/01_ingest.py
python3 scripts/02_audit.py
python3 scripts/03_segment_circuit.py
python3 scripts/04_build_episodes.py
python3 scripts/05_build_validated.py
python3 scripts/06_resample.py
python3 scripts/07_detect_transients_and_phases.py
python3 scripts/08_build_features.py
python3 scripts/08b_build_processed_variants.py
python3 scripts/09_build_splits_and_scalers.py
python3 scripts/10_apply_scalers.py
python3 scripts/11_build_reports.py
python3 -m pytest tests/ -v
```

## 7. Resultado dos testes

`python3 -m pytest tests/ -v` → **128 passed, 0 failed** (cobre todos os 21
itens da seção 19: integridade dos CSVs brutos, ordenação/duplicatas de
timestamp, frequência de 10 Hz, ausência de NaN/Inf, dimensões exatas das
duas redes, ausência de colunas proibidas/lat-lon/frame/sim_time nos inputs,
os dois pousos do circuito no mesmo fold, exclusão de `voo-inicio-teste` de
todos os folds, scalers sem vazamento de sessão de validação/holdout,
normalização com média~0 (desvio-padrão próximo de 1, com banda mais larga
para variáveis de taxa angular com caudas pesadas — ver nota no teste),
rótulos preservando escala original, ranges de controle válidos, duplicatas
resolvidas, e rastreabilidade linha-a-linha).

## 8. Recomendações

- **Transientes iniciais**: usar `baseline_core` como configuração principal
  do baseline (exclui o transiente artificial), conforme já implementado.
  `recovery_ablation` fica pronto para um experimento futuro de robustez.
- **Aproximação completa do circuito**: `full_approach` preserva os ~140s
  completos; recomenda-se usá-lo em um experimento separado antes de
  decidir se vale a pena misturar aproximações "com manobra de circuito"
  no dataset principal — por ora, `baseline_core` (só a final estabilizada)
  é a escolha mais segura para a arquitetura atual.
- Confirmar manualmente o achado da pista 29 antes de publicar resultados
  baseados em `circuito-completo__landing-02`.
