# Player-Team Fit Model — Design Doc

## Motivation

Inspired by a conversation with Chris Mast (Lean Basketball Analytics,
former Hawks/Hornets analytics), who described an approach of asking "what
opportunities isn't a player being given, and how would he perform in a
different context" rather than just describing what already happened. His
critique of AI analytics tools generally: "so what — evidence, not just
answers."

## Core design constraint (non-negotiable)

This model must NEVER predict a specific outcome for a specific
hypothetical player-team pairing. That would be indistinguishable from the
trade-value/"who should we draft" speculation this project already
explicitly refuses elsewhere (see `query_router.py`'s `out_of_scope`
handling). Every output must be either (a) a structural, computable fact
about a real roster today, or (b) a comp to real players who already exist
in a similar real situation. No LLM-generated or model-invented outcome
numbers, ever — same rule as every other function in this project.

## Architecture, layer by layer

### Layer 1: Player archetype feature vector

Seven feature axes, each grounded in a real, already-pulled data source:

| Axis | Signal | Data source |
|---|---|---|
| Shot-creation style | Self-creation vs. assisted share | `playtype_offense_*.csv` (Isolation/PRBallHandler/Postup vs. Spotup/Cut/Handoff/OffScreen POSS split), via `compute_offense.py`'s `playtype_offense` |
| Playmaking role | `AST_PCT`, `AST_TO`, `AST_RATIO`, `DRIVE_AST_PCT`, `DRIVE_PASSES_PCT` | `usage_context_2025_26.csv` (`AST_PCT`, `AST_TO`, `AST_RATIO`) + `drives_2025_26.csv` (`DRIVE_AST_PCT`, `DRIVE_PASSES_PCT`), via `compute_offense.py`'s `drive_efficiency` |
| Finishing / rim pressure | Drive volume and efficiency | `drives_2025_26.csv` (`DRIVES`, `DRIVE_PTS`, `DRIVE_FG_PCT`) |
| Off-ball movement profile | Spotup vs. Cut/OffScreen mix | `playtype_offense_spotup_2025_26.csv`, `playtype_offense_cut_2025_26.csv`, `playtype_offense_offscreen_2025_26.csv` (POSS share across categories) |
| Defensive role | Defensive playtype mix, shot suppression by zone, hustle stats | `playtype_defense_*.csv` (POSS/PPP by category), `shot_defense_{overall,3pt,2pt,rim}_2025_26.csv` (`PCT_PLUSMINUS`/`PLUSMINUS`), `hustle_stats_2025_26.csv` (deflections, contests, boxouts) |
| Efficiency-under-volume | `TS_PCT`/`EFG_PCT` relative to `USG_PCT` | `usage_context_2025_26.csv` (`TS_PCT`, `EFG_PCT`, `USG_PCT`) |
| Pace / tempo fit | `PACE`, transition volume/efficiency | `usage_context_2025_26.csv` (`PACE`, `E_PACE`) + `playtype_offense_transition_2025_26.csv` (`POSS`, `PPP`) |

All seven axes are computable today from CSVs already pulled and
schema-validated as part of the standard `refresh_data.sh` cycle — no new
data pull is required to begin Layer 1 implementation.

### Layer 2: Style embedding via SVD

Factor the player × play-type matrix (players as rows, play-type
POSS-share/PPP as columns, offense and defense both included) to derive
latent style dimensions directly from the data's own variance structure,
rather than relying solely on the seven hand-picked axes above. The
hand-picked axes in Layer 1 give the model interpretable, named inputs;
the SVD embedding gives it dimensions the data actually supports, which
may not align cleanly with any single hand-picked axis. Both are retained
and used together downstream, not one in place of the other.

### Layer 3: Similarity via Mahalanobis distance

Real-player comps are computed using Mahalanobis distance, not naive
Euclidean or cosine distance, specifically because the feature axes above
are correlated (e.g. high usage and high assist rate tend to move
together; drive volume and rim-pressure metrics overlap). Mahalanobis
distance corrects for that covariance structure so a comp isn't
double-weighted on two features that are really measuring one underlying
thing. This is the only mechanism by which this model ever names another
player — always an existing player in the dataset, never a synthetic or
invented one.

Comps reflect playing style only, not position or physical size — there
is no position/height/weight feature anywhere in this model, so two
players can be styled similarly while being completely different builds
(e.g. Chet Holmgren appearing in SGA's comp list despite the size
difference).

### Layer 4: Team roster fingerprint + orthogonal-projection gap score

A team's current roster is represented as a subspace of style-space
(spanned by its rostered players' style vectors, from Layers 1–2). A
candidate player's fit is scored as the magnitude of the component of
their style vector that is *orthogonal* to that subspace — i.e. how much
of what they do is not already covered by someone on the roster. This is
a structural gap measurement, not a performance prediction: it answers
"does this roster already have a player who moves like this," not "how
would this player perform here."

### Layer 5: Gradient-based sensitivity + gradient ascent to ideal archetype

**Sensitivity**: for a given team fingerprint, compute which feature
direction most increases the orthogonal-gap fit score — i.e. which single
axis of style the roster is most lacking.

**Ascent**: starting from the sensitivity direction, run gradient ascent
in style-space to find the team's theoretical "ideal" archetype point —
the synthetic style vector that would maximize fit. This synthetic point
is an intermediate computation only. It is immediately snapped to the
nearest real player(s) via Layer 3's Mahalanobis comp mechanism, and only
those real-player comps are ever surfaced as output. The synthetic optimum
itself is never shown as an answer, logged as an answer, or exposed
through any API response — consistent with the core design constraint
above.

## Determining archetype count (explicitly not asserted upfront)

The number of real distinct archetypes will be determined by explained
variance from PCA and/or elbow method or silhouette score on k-means
applied to the Layer 1/2 feature space — **not** decided in advance.

The following eight archetypes were hypothesized during this planning
session as plausible candidates worth testing for, based on domain
intuition about how NBA role-players are commonly described:

1. Movement shooter
2. Primary shot creator
3. Offensive hub
4. Rim-running finisher
5. 3-and-D wing
6. High-activity defensive specialist
7. Two-way engine
8. Traditional post scorer

These are **unvalidated hypotheses to test against real data**, not
asserted facts. The actual archetype count and composition that emerges
from PCA/clustering on real player data may differ — in count, in
boundaries, or in which of these eight (if any) survive as a clean
cluster. No downstream logic should assume these eight are correct until
that validation step has run.

### Real result (k-means on the 305-player feature matrix)

The validation step above has now run (`feature_vector.py`, full output in
`KMEANS_CLUSTERING_OUTPUT.txt`). The eight hypotheses above are kept for
historical record; they are not what the data actually supports.

- **No strongly-separated cluster structure at any k.** A k-means sweep
  from k=3 through k=12 on the same standardized, median-imputed feature
  matrix used for PCA found silhouette scores ranging from 0.096 (k=3,
  the best of the sweep) down to 0.062 — well below the ~0.25 threshold
  generally considered evidence of real cluster structure. Player style
  in this feature space is closer to continuous than discretely
  clustered.
- **The best-available k=3 split produced three coarse groups, not eight
  clean archetypes:** a "bigs" cluster (51 players — rim protection, box
  outs, efficient finishing), a "high-usage/high-AST ball-handler"
  cluster (108 players, over a third of the qualified population,
  including both Jokić and SGA despite their different offensive
  styles), and a weak, low-signal "everyone else" cluster (146 players,
  minimal distinguishing features).
- **The PC3 blend observed in the earlier PCA run holds under full
  clustering too.** PCA's PC3 had mixed "primary shot creator" and
  "offensive hub" together (both Jokić and SGA scored high on it despite
  different styles); direct verification confirms Jokić and SGA land in
  the same k-means cluster even when clustering on the full feature
  space rather than just the top principal components. This isn't a
  PC3-specific artifact — it's a genuine property of this feature set.
- **Design implication:** the 8-archetype hypothesis is not supported as
  a set of clean, nameable player types in this feature space. This does
  **not** block the rest of the model — Layers 2-5 (SVD embedding,
  Mahalanobis comps, orthogonal-projection gap scoring, gradient
  sensitivity/ascent) all operate on continuous style vectors directly
  and were never dependent on discrete archetype labels. The
  8-archetype language should be treated as retired illustrative
  framing, not a functional component of the model going forward.

## What this replaces / what it doesn't do

This model does **not** predict how a specific player would perform on a
specific team. It does not output a projected stat line, a fit score
framed as a performance forecast, or any number attributed to a
hypothetical player-team pairing that doesn't already exist.

What it does surface:
- Structural roster gaps — which style dimensions a team's current roster
  does not cover (Layer 4).
- Real comps — existing players whose style vectors are close to a given
  reference point, corrected for feature correlation (Layer 3).
- Which single feature direction most affects a team's fit gap (Layer 5,
  sensitivity only — never the synthetic ascent endpoint itself).

Anything resembling "Player X would average Y on Team Z" is out of scope
for this model, permanently, by the core design constraint above — not a
current limitation to be relaxed later.

## Implementation status

Layer 1 (feature vectors, `feature_vector.py`), the PCA/k-means archetype
validation above, and Layer 3 (Mahalanobis comps) are built. Layer 4
(roster fingerprint + gap score) is built but **not validated as
trustworthy** — see below. Layer 5 (gradient sensitivity/ascent) is not
yet started.

### Layer 4 status: metric found unreliable, root cause identified, fix not yet built

`roster_fit.py`'s original full-28-feature orthogonal-projection gap
score failed its own basketball-sense check: CLE (two elite shot-blocking
bigs, Mobley + Allen) scored *worse* Donovan Clingan coverage than CHI
(no true center at all) — the opposite of basketball intuition
(`TEAM_FIT_LAYER4_OUTPUT.txt`).

`layer4_subsets.py` investigated by (a) validating each candidate feature
against real `NET_RATING` before trusting it, (b) restricting the gap
score to 6 named capability subsets instead of all 28 features at once,
and (c) replacing the orthogonal-projection method with
nearest-rostered-player distance, since subset dimensionality (2-7
features) is smaller than a typical roster's qualified-player count,
which makes the SVD-span method degenerate (every roster trivially spans
a low-dimensional subset, forcing gap → 0 for any player against any
team). None of these fixes resolved the CLE/CHI failure on their own:
removing the single most outlier-distorting feature
(`CONTESTED_SHOTS_PER36`) didn't fix it, and averaging over the roster's
3 closest players instead of just the closest didn't either. A
reference-sensitivity test found the ranking *did* flip correctly for
Chet Holmgren (a less statistically extreme rim protector) — but a
second, independently-picked median-range reference (Onyeka Okongwu)
failed to replicate that pass, in a different way: his near-population-
average z-scored vector has small magnitude by construction, which
mechanically compresses `pct_covered` toward 0 for every team regardless
of actual style similarity.

**Follow-up diagnostic (`layer4_middleband.py`, run 2026-09-19):** tested
whether a genuine "middle band" of reference-player vector magnitude
exists where `pct_covered` behaves sensibly, by screening 270 qualified
players' `RIM_PROTECTION_TRIMMED` magnitude and testing 7 points spanning
the full range (1.46 to 5.00) against the same CLE > CHI basketball-sense
check:

| Player | Magnitude | CLE vs. CHI |
|---|---|---|
| Onyeka Okongwu | 1.46 | FAIL |
| Nique Clifford | 2.00 | FAIL |
| John Konchar | 2.75 | **PASS** |
| Anthony Gill | 3.42 | FAIL |
| Chet Holmgren | 3.99 | **PASS** |
| Jusuf Nurkić | 4.16 | FAIL |
| Donovan Clingan | 5.00 | FAIL |

Only 2 of 7 points pass, and they don't cluster: 2.75 and 3.99 are
separated by two failing points (3.42, 4.16) in between. **This rules out
the "middle band" hypothesis** — magnitude alone does not explain which
reference players produce a sensible ranking and which don't. The
Holmgren pass was not evidence of a trustworthy magnitude range; it was
close to a coin flip.

**Conclusion:** `pct_covered`, as currently defined (nearest-rostered-
player Euclidean distance in z-scored feature space, expressed as a
percentage of the reference player's own vector magnitude), is not a
reliable metric at any point tested so far. The problem is in the metric
itself, not in reference-player selection or feature-subset choice. Next
step is a genuinely different metric — candidates worth testing:
a percentile-rank-based coverage score instead of a Euclidean-distance
ratio, or reporting raw untransformed distance without normalizing by
the reference player's own magnitude (which is what ties the score to
how extreme that specific player happens to be, the common thread across
every failure mode found so far).

Percentile-rank was chosen as the direction to test first, over raw
unnormalized distance, because it is bounded and comparable by
construction across features of different raw variance (the specific
property the z-scored magnitude-normalization failure traced to), and
because it matches this project's existing trust-vocabulary —
`signature_play_type` already ranks players by `PERCENTILE`, and every
Layer 4 diagnostic so far already explains its own findings in
percentile terms because that is the unit a scout can act on directly
("this roster's closest comp sits 35 percentile points below the query
player"), unlike an opaque z-scored ratio.

### Percentile-space redesign: real improvement, not a full fix

`layer4_percentile.py` reruns the identical 7-point reference-player
spectrum from the middle-band test, converting each
`RIM_PROTECTION_TRIMMED` feature to a percentile rank (0-100 across the
305 qualified players, via the same `(series < val).mean() * 100`
formula already used elsewhere in this codebase for percentile
printouts) before computing nearest-rostered-player coverage, instead of
using z-scored distance normalized by the reference player's own
magnitude.

Two of the seven spectrum players (John Konchar, Jusuf Nurkić) could not
be tested on this subset at all: both have a genuine missing value in
one of the 4 `RIM_PROTECTION_TRIMMED` features (insufficient possessions
in that specific defensive play-type category, per Layer 1's
never-zero-fill-a-missing-category policy), so no percentile or
z-scored vector can be computed for them — this is a real data-coverage
limit of this specific 4-feature subset, not a bug in the percentile
transform.

Of the 5 players with complete data:

| Player | z-scored verdict | Percentile-space verdict |
|---|---|---|
| Onyeka Okongwu | FAIL | **PASS** |
| Nique Clifford | FAIL | FAIL |
| Anthony Gill | FAIL | **PASS** |
| Chet Holmgren | PASS | PASS |
| Donovan Clingan | FAIL | FAIL |

3 of 5 testable points pass in percentile space, versus 2 of 7 under the
z-scored metric — a real improvement, and notably it fixed exactly the
failure mode the metric was built to fix (Okongwu, the near-median
outlier-collapse case, now passes). But it is **not a full fix**: Nique
Clifford and Donovan Clingan still produce the wrong-direction ranking.
Percentile rank removes the magnitude-normalization defect but does not
by itself resolve every case — Clingan in particular remains an extreme
outlier in percentile terms too (99.7th-percentile-equivalent on his
most extreme axis), so an outlier reference can still distort
nearest-neighbor distance even on a bounded 0-100 scale.

**Status at the 5-point spectrum: promising, not yet production-
trustworthy.** The 3-of-5 result alone is not real verification — 5
points, several hand-picked by a magnitude-screening heuristic, is not a
statistical sample, and "CLE should beat CHI" is a basketball-intuition
judgment call, not a checkable external fact. This was called out
directly and correctly: before trusting this metric further, it needed
to be checked against real, independently-computed data at league scale,
not more hand-picked pairwise comparisons.

### League-wide verification against real, independent data (2026-09-19)

`layer4_team_groundtruth.py` builds `TEAM_RIM_DEFENSE_REAL`: a
volume-weighted mean of real `PLUSMINUS` (rim shot defense, from
`data/shot_defense_rim_2025_26.csv`) across each team's qualified
rostered players — a genuine, independently-computed "how well does this
team's actual personnel suppress rim shots" number, joined on the same
current-roster `TEAM_ABBREVIATION` every other Layer 4 module already
uses (not the raw CSV's `PLAYER_LAST_TEAM_ABBREVIATION`, which can be
stale for a traded player — Donovan Clingan is on POR in this season's
data, not CLE/CHI as used in the earlier illustrative diagnostics).

This was correlated against the Layer-4 percentile-space coverage score
for two reference players, across all 30 NBA teams (not one hand-picked
pair):

| Reference player | Pearson r (n=30) | Robustness check (excl. own team, n=29) |
|---|---|---|
| Chet Holmgren | **−0.4490** | −0.3274 |
| Donovan Clingan | **−0.4027** | −0.4117 |

Both correlations are in the expected direction (good real rim defense →
higher coverage of an elite rim-protector reference) and moderate in
strength — well above `layer4_subsets.py`'s own
`WEAK_CORRELATION_THRESHOLD = 0.05` bar — and the effect **survives
excluding each reference player's own roster** (the one data point that
scores a trivial 100% by construction), ruling out the concern that the
correlation is just an artifact of that mechanical case.

**Caveat, stated directly, not hidden:** this check is not fully
independent. `TEAM_RIM_DEFENSE_REAL` and one of `RIM_PROTECTION_TRIMMED`'s
4 input features (`SHOT_SUPPRESSION_LESS_THAN_6FT`) are both built from
the same underlying `PLUSMINUS` values in
`data/shot_defense_rim_2025_26.csv` — at the player level for the
coverage metric's own inputs, aggregated to team level for the
ground-truth stat here. So this validates that the nearest-neighbor/
percentile-transform *mechanism* correctly propagates a real signal from
player-level data up to a sensible team-level result — it is not a fully
external check against an unrelated outcome variable (that would require
something like team defensive `NET_RATING`, not yet run).

**Status, honestly: this is real, citable evidence that the
percentile-space coverage metric works as intended for the
rim_protection subset specifically** — a moderate, correctly-signed,
robustness-checked correlation across the full league, not 5 anecdotes.
It is not proof the whole Layer 4 approach is sound: this checks one
capability subset against one partially-overlapping outcome proxy. The
honest next steps, if this is pursued further: (1) run the same
league-wide correlation check for the other 5 capability subsets
(perimeter_defense, primary_shot_creation, playmaking, movement_scoring,
pace_transition), each against its own appropriate ground-truth proxy,
and (2) find a genuinely external outcome variable (e.g. team defensive
`NET_RATING`) to check against, since the rim-protection check above
shares one input feature with its own ground-truth stat.
