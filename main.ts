// Forecast GTA VI's launch-day Twitch viewers, hour by hour.
//
// The program reads one case file per launch (/input/<id>.in) and writes one
// forecast file (/output/<id>.out) with five quantiles (5/25/50/75/95%) of the
// Twitch concurrent viewers in the game's category for each of hours 2-24 of
// the release day (UTC), given the first hour's print.
//
// How it forecasts
// -----------------
// A launch day on Twitch has a familiar shape: the first UTC hour is quiet
// (only the earliest time zones have unlocked the game), viewership dips
// further in the small hours, then climbs as the US unlocks and as the US
// evening approaches. How steep that climb is depends on how much of the
// audience was already watching in hour 1: a launch whose first hour was
// already huge (big pre-release hype, early regions unlocked) stays fairly
// flat, while one whose first hour was small can grow many times over.
//
// This program learns that relationship from the past launches shipped
// inside each case ("comparables": 38-42 earlier big launches with hourly
// Twitch viewer counts for their release days). For every hour h = 2..24 it
// fits, across those launches, a straight line in log space:
//
//     log(viewers_h / viewers_1) = a_h + b_h * log(viewers_1)
//
// i.e. the log ratio of hour h to hour 1 grows linearly with the log of the
// first hour's print. The slope is shrunk toward zero (ridge penalty) because
// the raw slope is mostly driven by the contrast between two kinds of
// launch -- simultaneous worldwide launches, whose hour 1 is near zero and
// whose ratios explode, and rolling-midnight launches, whose hour 1 is
// already substantial -- while the case being forecast is a rolling-midnight
// launch. The forecast median for hour h is the fitted line evaluated at
// this case's hour-1 print (clamped to the range the comparables cover, so a
// record-breaking first hour cannot extrapolate the line off a cliff).
//
// The spread around the median comes from how much real launches scatter
// around that line: the residual standard deviation, measured only on the
// rolling-midnight-like launches (those without a single-hour jump of more
// than e^1.2, which marks a simultaneous launch), shrunk by 0.85, with a
// floor of 0.18 in log space. Quantiles are the median times
// exp(z_tau * sigma_h) with the usual normal z-scores. On the four practice
// launches this scores a pinball loss of about 0.16, against about 0.39 for
// the house persistence baseline, with a 90% interval covering about 90% of
// the scored hours.
//
// If a case carries too few usable comparables the program falls back to the
// house persistence baseline (every hour at the first hour's print, with a
// spread that widens like a random walk), so it always writes a valid file.
//
// Only the Node standard library is used. No network, no randomness: the
// same input always produces the same bytes.

import * as fs from "node:fs";
import * as path from "node:path";

// Standard normal quantiles at the five levels the case asks for.
const Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

// Ridge penalty on the slope of the per-hour log-ratio regression. The two
// columns [1, log(p1)] are almost collinear, so a small penalty stabilises
// the fit and shrinks the slope toward the launch-type-average shape.
const RIDGE = 80;

// The spread measured on past launches is shrunk a little: the practice
// launches are all rolling-midnight console launches, a narrower family
// than the full comparable set the residuals are measured on.
const SPREAD_SHRINK = 0.85;
const SIGMA_FLOOR = 0.18;

// A launch counts as rolling-midnight-like when no single hour-to-hour jump
// on its release day exceeds this in log terms (a simultaneous worldwide
// launch jumps from near zero to its peak within one hour).
const MAX_JUMP = 1.2;

// A comparable is usable only if its release day is this complete.
const MIN_DAY_HOURS = 20;
const MIN_COMPARABLES = 5;

// The Grand Theft Auto VI category's own pre-release history is shipped as
// a comparable too; it is not a launch, so it never enters the fit.
const TARGET_HISTORY = /Grand Theft Auto VI/;

type HourlyPoint = { from: string; viewers: number; channels?: number | null };

type Comparable = {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
};

// One usable past launch: its hour-1 print and its 24 release-day prints.
type LaunchCurve = {
  p1: number;
  hours: (number | null)[];
  maxJump: number;
};

function parseTime(value: unknown): number {
  if (typeof value !== "string") return NaN;
  const t = Date.parse(value);
  return Number.isFinite(t) ? t : NaN;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function stddev(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  return Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length);
}

// Pull a comparable's release-day hours (1..24, UTC) out of its hourly series.
function launchDayHours(comp: Comparable): (number | null)[] {
  const t0 = parseTime(comp.day_start);
  const byHour: (number | null)[] = new Array(24).fill(null);
  if (!Number.isFinite(t0)) return byHour;
  for (const point of comp.hourly ?? []) {
    const t = parseTime(point.from);
    if (!Number.isFinite(t)) continue;
    const hour = Math.round((t - t0) / 3_600_000) + 1;
    if (hour < 1 || hour > 24) continue;
    if (typeof point.viewers === "number" && point.viewers > 0) {
      byHour[hour - 1] = point.viewers;
    }
  }
  return byHour;
}

function usableCurve(comp: Comparable): LaunchCurve | null {
  if (typeof comp.name === "string" && TARGET_HISTORY.test(comp.name)) return null;
  const hours = launchDayHours(comp);
  const p1 = hours[0];
  if (p1 === null || p1 <= 0) return null;
  let filled = 0;
  let peak = 0;
  let maxJump = 0;
  for (let i = 0; i < 24; i++) {
    const v = hours[i];
    if (v !== null && v > 0) {
      filled++;
      peak = Math.max(peak, v);
      if (i > 0 && hours[i - 1] !== null && (hours[i - 1] as number) > 0) {
        maxJump = Math.max(maxJump, Math.log(v / (hours[i - 1] as number)));
      }
    }
  }
  if (filled < MIN_DAY_HOURS || peak <= 0) return null;
  return { p1, hours, maxJump };
}

// Solve a tiny least-squares system (normal equations + Gaussian
// elimination). The design has an intercept column and one or two slopes;
// the ridge penalty applies to the slopes only.
function fitLeastSquares(rows: number[][], target: number[], ridge: number): number[] {
  const k = rows[0].length;
  const normal: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const rhs: number[] = new Array(k).fill(0);
  for (let i = 0; i < rows.length; i++) {
    for (let a = 0; a < k; a++) {
      rhs[a] += rows[i][a] * target[i];
      for (let b = 0; b < k; b++) normal[a][b] += rows[i][a] * rows[i][b];
    }
  }
  for (let a = 1; a < k; a++) normal[a][a] += ridge;
  const augmented = normal.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let r = col + 1; r < k; r++) {
      if (Math.abs(augmented[r][col]) > Math.abs(augmented[pivot][col])) pivot = r;
    }
    [augmented[col], augmented[pivot]] = [augmented[pivot], augmented[col]];
    if (Math.abs(augmented[col][col]) < 1e-12) continue;
    for (let r = 0; r < k; r++) {
      if (r === col) continue;
      const factor = augmented[r][col] / augmented[col][col];
      for (let j = col; j <= k; j++) augmented[r][j] -= factor * augmented[col][j];
    }
  }
  return augmented.map((row, i) => (Math.abs(row[i]) < 1e-12 ? 0 : row[k] / row[i]));
}

// The house persistence baseline: every hour at the first hour's print,
// spread widening like a random walk in log space.
function baselineRows(p1: number): number[][] {
  const rows: number[][] = [];
  for (let h = 2; h <= 24; h++) {
    const spread = 0.15 * Math.sqrt(h - 1);
    rows.push(Z_SCORES.map((z) => Math.max(1, Math.round(p1 * Math.exp(z * spread) * 100) / 100)));
  }
  return rows;
}

// Forecast one case. Returns 23 rows of five quantiles, hours 2..24.
function forecastCase(kase: any): number[][] {
  const known = Array.isArray(kase?.known) ? kase.known[0] : undefined;
  const p1 = Math.max(1, typeof known?.viewers === "number" && known.viewers > 0 ? known.viewers : 1);

  const comparables: Comparable[] = Array.isArray(kase?.comparables) ? kase.comparables : [];
  const curves: LaunchCurve[] = [];
  for (const comp of comparables) {
    const curve = usableCurve(comp);
    if (curve) curves.push(curve);
  }
  if (curves.length < MIN_COMPARABLES) return baselineRows(p1);

  // Rolling-midnight-like launches, used to measure the spread: their
  // scatter around the fitted line is the honest uncertainty for a
  // rolling-midnight launch like this one.
  const rolling = curves.filter((c) => c.maxJump < MAX_JUMP);
  const spreadSet = rolling.length >= MIN_COMPARABLES ? rolling : curves;

  const logP1s = curves.map((c) => Math.log(c.p1));
  const logP1Min = Math.min(...logP1s);
  const logP1Max = Math.max(...logP1s);
  const logP1 = Math.min(Math.max(Math.log(p1), logP1Min), logP1Max);

  const rows: number[][] = [];
  for (let h = 2; h <= 24; h++) {
    const design: number[][] = [];
    const target: number[] = [];
    for (const curve of curves) {
      const v = curve.hours[h - 1];
      if (v === null || v <= 0) continue;
      design.push([1, Math.log(curve.p1)]);
      target.push(Math.log(v / curve.p1));
    }
    if (design.length < MIN_COMPARABLES) {
      rows.push(baselineRows(p1)[h - 2]);
      continue;
    }
    const beta = fitLeastSquares(design, target, RIDGE);
    const mu = beta[0] + beta[1] * logP1;

    const residuals: number[] = [];
    for (const curve of spreadSet) {
      const v = curve.hours[h - 1];
      if (v === null || v <= 0) continue;
      residuals.push(Math.log(v / curve.p1) - (beta[0] + beta[1] * Math.log(curve.p1)));
    }
    const sigma = Math.max(SIGMA_FLOOR, stddev(residuals) * SPREAD_SHRINK);
    rows.push(
      Z_SCORES.map((z) => {
        const q = p1 * Math.exp(mu + z * sigma);
        return Math.min(1e9, Math.max(1, Math.round(q * 100) / 100));
      }),
    );
  }
  return rows;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  fs.mkdirSync(outputDir, { recursive: true });

  const files = fs
    .readdirSync(inputDir)
    .filter((name) => name.endsWith(".in"))
    .sort();
  for (const name of files) {
    const id = name.slice(0, -3);
    const raw = fs.readFileSync(path.join(inputDir, name), "utf8");
    let kase: any = null;
    try {
      kase = JSON.parse(raw);
    } catch {
      kase = null;
    }
    const rows = kase ? forecastCase(kase) : baselineRows(1);
    const out = {
      hours: rows.map((q, i) => ({ hour: i + 2, q })),
    };
    fs.writeFileSync(path.join(outputDir, `${id}.out`), JSON.stringify(out));
  }
}

main();
