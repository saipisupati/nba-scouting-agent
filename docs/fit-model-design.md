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

Not yet built. Data pipeline prerequisite (`usage_context_2025_26.csv`) is
already pulled and schema-validated as of 2026-08-21.

Next step: implement Layer 1 (feature vector construction) and run PCA
against real player data to test whether the hypothesized archetypes
actually emerge.
