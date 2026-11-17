# Forecast GTA VI's launch-day Twitch viewers hour by hour

Entry for the make.fun bounty `gta6-twitch-ccu-forecast-01` ($250): a program
that reads one case (`/input/<id>.in`) and writes, for each of hours 2–24 of
GTA VI's release day (UTC), five quantiles (5/25/50/75/95%) of the Twitch
concurrent viewers in the "Grand Theft Auto VI" category, given the first
hour's print.

## How it works

See the header comment in `main.ts`. In short: for every hour h = 2..24 the
program fits, across the past launches shipped inside the case
("comparables"), a ridge-shrunk straight line in log space,

```
log(viewers_h / viewers_1) = a_h + b_h * log(viewers_1)
```

so the forecast shape adapts to the size of the first hour's print — a
launch that was already huge in hour 1 stays fairly flat, one that started
small climbs steeply as the world unlocks. The spread comes from the scatter
of rolling-midnight-like past launches around that line. If a case carries
too few usable comparables it falls back to the house persistence baseline.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22+, standard library only, no network, deterministic.

## Local practice grade

On the kit's four practice launches (past big launches, SullyGnome hourly
data), scored against the recorded prints:

| | loss | typical miss | 90% coverage |
|---|---|---|---|
| this entry | **0.161** | 26.1% | 0.902 |
| house persistence baseline | 0.387 | 82.7% | 0.783 |

Per-case loss: p00 0.146, p01 0.144, p02 0.121, p03 0.232.
