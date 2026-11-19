[![make.fun competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/meigidimas/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/meigidimas/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity — week-one quantile forecaster

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every Grand Theft Auto VI trophy on PlayStation 5, the
share of Exophase-tracked players who will hold it at 2026-11-26 00:00
UTC, a week after launch, as five quantiles (5%, 25%, 50%, 75%, 95%) per
trophy. Scored by pinball loss on the logit scale; a prize needs a loss
strictly below the best house entry.

## How it works

`main.py` (Python 3.13, standard library only, no network) forecasts in
logit space, because the score does. Two forces move every rate in
opposite directions — players earn trophies as they play, while Exophase
tracks more players over launch week, which dilutes every rate — so the
median blends two views of the same history, both fitted on log time
(`ln(1 + hours)` since the release day began):

1. **Logit trend (3/4 of the blend):** a per-trophy least-squares line
   of logit(rate) on log time over the last 24 hours of reads, its
   slope shrunk toward the cross-trophy median slope, so a trophy with
   three flat reads never gets a wild slope of its own.
2. **Count model (1/4 of the blend):** log earned-count and log
   players-tracked each get a line on log time over the full history,
   and the rate forecast is the ratio of the two extrapolated counts.
   This is the view that knows dilution. It uses the full history
   because the last day's growth overstates the week when the player
   base is near its fastest growth.

The blend is pulled slightly back toward the last observed rate
(shrinkage that grows with how much distinct history the trophy has).
The spread combines the house random-walk term (0.5 logit per root-day)
with the fit's residual spread, the slope's uncertainty over the
extrapolation gap, and a penalty when the history is stale.

**Frozen history gets special handling.** On launch night Exophase did
not refresh its counts for hours, so a whole history can be one frozen
page — and the kit documents what the first refresh did: tracked
players jumped 638 → 3,868 while the trophies earned outside the one
everybody-held trophy barely moved (305 → 408), so every other trophy's
rate fell to roughly a fifth. When a trophy's history shows no real
movement, the median is shifted down by about that scale of dilution
(less for easy early trophies, more for the hardest ones) and the spread
is widened. The 100% first-mission trophy is the exception: it stays
near 100% as new players arrive.

Fallbacks (persistence with the house spread, then a wide low prior)
keep the output valid on empty, single-read, or hole-ridden histories.
The program is deterministic: the same case always produces the same
bytes. About 0.07 s per case.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one
`<id>.out` forecast per case.

## Local validation

No local grader ships with the kit (the server grades every entry), so
the model was validated two ways, both with the exact pinball loss from
the rules.

**Practice reconstruction.** The practice answers are sealed, but the
kit states the refresh they straddle (players 638 → 3,868, trophies
earned 943 → 4,276, every rate but the 100% one down), which pins the
average non-trivial trophy to about 22% of its frozen rate. Under five
scenarios built on that arithmetic (uniform and tiered drops, integer
earned-counts), this entry scores 0.17–0.24 against the house baseline's
1.10–1.78. These are simulations, not the sealed answers — but the
baseline's simulated loss (1.34–1.78) brackets the 1.51 the rules state
for it, which is a good sign the scenarios are in the right neighbourhood.

**Synthetic final-like worlds.** Three simulated 2-day histories with a
4.5-day horizon: steady power-law growth, logistic saturation with
refresh noise, and growth that stalls after the deadline.

| world | house baseline | this entry |
| --- | --- | --- |
| steady growth | 0.266 | **0.245** |
| saturating + noisy | 0.533 | **0.329** |
| growth stalls | **0.310** | 0.358 |

The entry beats persistence when there is real signal to follow and
loses to it only in the stall world, where no trend-following forecast
can win. It has not been tested against the house curve baseline or the
house model forecast locally — those are server-side — so beating the
prize bar is not guaranteed.

## License

MIT — see [LICENSE](LICENSE).
