import { createHash, randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import { dirname, join } from "node:path";
import { calculateCost } from "@earendil-works/pi-ai";
import type { Api, Message, Model, Usage } from "@earendil-works/pi-ai";
import { getAgentDir, withFileMutationQueue } from "@earendil-works/pi-coding-agent";
import type { RoutingPolicy } from "./router-config.ts";
import type { EffectiveControl, TaskAssessment, TaskFamily } from "./routing-types.ts";

const VERSION = 1;
const MAX_PROFILES = 256;
const MAX_CALIBRATIONS = 1024;
const MAX_LEDGER = 8192;
const MAX_SAMPLES = 10000;
const MAX_FILE_BYTES = 2_000_000;
const FAMILIES: readonly TaskFamily[] = ["mechanical", "coding", "reasoning", "research", "review", "unknown"];
const historyData: unique symbol = Symbol("routing-history");
// Explicit mutable model aliases have no durable release identity without resolved metadata.
const unresolvedAliasEpoch = randomUUID();

interface Profile {
  n: number;
  input: number;
  output: number;
  cacheWriteFraction: number;
  longWriteFraction: number;
  latencyMs: number;
  lastTimestamp: number;
  lastCacheReadFraction: number;
  updated: number;
}
interface Calibration {
  n: number;
  successes: number;
  predictionSum: number;
  updated: number;
}
interface TaskProfile {
  n: number;
  successes: number;
  family: number;
  user: number;
  contract: number;
  updated: number;
}
interface Data {
  version: number;
  sequence: number;
  profiles: Record<string, Profile>;
  calibrations: Record<string, Calibration>;
  tasks: Record<string, TaskProfile>;
  ledger: Record<string, number>;
}
/** Opaque, read-only snapshot; persisted records contain hashed identities and numeric aggregates only. */
export interface RoutingHistory {
  readonly [historyData]: { data: Data; loaded: boolean };
}
export interface UsageObservation {
  id: string;
  modelKey: string;
  controlKey: string;
  family: TaskFamily;
  usage: Usage;
  durationMs: number;
  timestamp: number;
}
export interface TaskOutcome {
  id: string;
  taskId: string;
  modelKey: string;
  controlKey: string;
  family: TaskFamily;
  classifierVersion: string;
  prediction: number;
  success: boolean;
  source: "user" | "contract";
}

export function emptyHistory(): RoutingHistory {
  return { [historyData]: { data: { version: VERSION, sequence: 0, profiles: {}, calibrations: {}, tasks: {}, ledger: {} }, loaded: false } };
}
function hash(parts: unknown): string {
  return createHash("sha256").update(JSON.stringify(parts)).digest("hex");
}
function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, canonical(item)]));
  return value;
}
/** Includes endpoint and effective model configuration, not the UI label or credentials. */
export function modelKey(model: Model<Api>, resolvedName?: string): string {
  const identity = resolvedName ?? model.id;
  return hash(canonical({ provider: model.provider, id: model.id, api: model.api, endpoint: model.baseUrl,
    resolvedName: identity, unresolvedAliasEpoch: stableVersion(identity) ? undefined : unresolvedAliasEpoch,
    reasoning: model.reasoning, thinkingLevelMap: model.thinkingLevelMap,
    compat: model.compat, contextWindow: model.contextWindow, maxTokens: model.maxTokens,
    cost: model.cost, promptCache: model.promptCache, samplingParams: model.samplingParams }));
}
function profileKey(model: string, control: string, family: TaskFamily): string {
  return hash([model, control, family]);
}
function calibrationKey(model: string, control: string, family: TaskFamily, version: string, prediction: number): string {
  return hash([model, control, family, version, Math.min(19, Math.floor(prediction * 20))]);
}
function stableVersion(version: string): boolean {
  return Boolean(version.trim()) && !/(?:^|[^a-z])(latest|current|auto)(?:$|[^a-z])/i.test(version);
}
function numeric(value: unknown, max = Number.MAX_SAFE_INTEGER): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= max;
}
function object(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
function validate(data: unknown): Data {
  if (!object(data) || data.version !== VERSION || !Number.isSafeInteger(data.sequence) || !numeric(data.sequence)
    || Object.keys(data).sort().join(",") !== "calibrations,ledger,profiles,sequence,tasks,version") throw new Error("unsupported schema or invalid header");
  for (const [name, limit] of [["profiles", MAX_PROFILES], ["calibrations", MAX_CALIBRATIONS], ["tasks", MAX_PROFILES], ["ledger", MAX_LEDGER]] as const) {
    const entries = data[name];
    if (!object(entries) || Object.keys(entries).length > limit) throw new Error(`invalid or oversized ${name}`);
    for (const [key, entry] of Object.entries(entries)) {
      if (!/^[a-f0-9]{64}$/.test(key)) throw new Error(`invalid ${name} identity`);
      if (name === "ledger") {
        if (!Number.isSafeInteger(entry) || !numeric(entry) || entry > data.sequence) throw new Error("invalid deduplication ledger");
        continue;
      }
      if (!object(entry) || Object.values(entry).some((value) => !numeric(value))) throw new Error(`invalid numeric ${name} aggregate`);
      const fields = name === "profiles"
        ? ["n", "input", "output", "cacheWriteFraction", "longWriteFraction", "latencyMs", "lastTimestamp", "lastCacheReadFraction", "updated"]
        : name === "tasks" ? ["n", "successes", "family", "user", "contract", "updated"] : ["n", "successes", "predictionSum", "updated"];
      if (Object.keys(entry).sort().join(",") !== fields.sort().join(",") || !Number.isInteger(entry.n)
        || !numeric(entry.n, MAX_SAMPLES) || entry.n === 0 || !Number.isSafeInteger(entry.updated)
        || !numeric(entry.updated) || entry.updated > data.sequence) throw new Error(`invalid ${name} sample count`);
      if (name === "profiles") {
        if (!numeric(entry.cacheWriteFraction, 1) || !numeric(entry.longWriteFraction, 1) || !numeric(entry.lastCacheReadFraction, 1)
          || !numeric(entry.input, 100_000_000) || !numeric(entry.output, 100_000_000) || !numeric(entry.latencyMs, 86_400_000)) throw new Error("invalid performance measurements");
      } else if (name === "tasks") {
        if (!numeric(entry.successes, entry.n) || !Number.isInteger(entry.successes) || !Number.isInteger(entry.family)
          || !numeric(entry.family, FAMILIES.length - 1) || !numeric(entry.user, entry.n) || !numeric(entry.contract, entry.n)
          || !Number.isInteger(entry.user) || !Number.isInteger(entry.contract) || entry.user + entry.contract !== entry.n) throw new Error("invalid task-family measurements");
      } else if (!numeric(entry.successes, entry.n) || !Number.isInteger(entry.successes)
        || !numeric(entry.predictionSum, entry.n)) throw new Error("invalid user-quality measurements");
    }
  }
  return data as unknown as Data;
}

export async function loadHistory(): Promise<RoutingHistory> {
  const path = join(getAgentDir(), "jev-llm-rtr-history.json");
  let content: string;
  try {
    const stat = await fs.stat(path);
    if (stat.size > MAX_FILE_BYTES) throw new Error("history exceeds its bounded size limit");
    content = await fs.readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return emptyHistory();
    throw new Error(`Could not read Jev routing history at ${path}: ${error instanceof Error ? error.message : String(error)}`);
  }
  try {
    return { [historyData]: { data: validate(JSON.parse(content)), loaded: true } };
  } catch (error) {
    throw new Error(`Invalid Jev routing history at ${path}: ${error instanceof Error ? error.message : String(error)}. The existing file was not reset.`);
  }
}
function trim<T extends { updated: number }>(entries: Record<string, T>, maximum: number): void {
  const keys = Object.keys(entries);
  if (keys.length <= maximum) return;
  keys.sort((a, b) => entries[a].updated - entries[b].updated);
  for (const key of keys.slice(0, keys.length - maximum)) delete entries[key];
}
async function mutate(update: (data: Data) => boolean): Promise<void> {
  const path = join(getAgentDir(), "jev-llm-rtr-history.json");
  await withFileMutationQueue(path, async () => {
    const data = (await loadHistory())[historyData].data;
    if (!update(data)) return;
    trim(data.profiles, MAX_PROFILES);
    trim(data.calibrations, MAX_CALIBRATIONS);
    trim(data.tasks, MAX_PROFILES);
    const ledgerKeys = Object.keys(data.ledger);
    if (ledgerKeys.length > MAX_LEDGER) {
      ledgerKeys.sort((a, b) => data.ledger[a] - data.ledger[b]);
      for (const key of ledgerKeys.slice(0, ledgerKeys.length - MAX_LEDGER)) delete data.ledger[key];
    }
    await fs.mkdir(dirname(path), { recursive: true });
    const temporary = `${path}.${randomUUID()}.tmp`;
    try {
      await fs.writeFile(temporary, JSON.stringify(data), { flag: "wx", mode: 0o600 });
      const file = await fs.open(temporary, "r");
      try { await file.sync(); } finally { await file.close(); }
      await fs.rename(temporary, path);
    } catch (error) {
      throw new Error(`Could not save Jev routing history at ${path}: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      await fs.unlink(temporary).catch((error: NodeJS.ErrnoException) => { if (error.code !== "ENOENT") throw error; });
    }
  });
}
function validIdentity(input: { id: string; modelKey: string; controlKey: string; family: TaskFamily }): void {
  if (![input.id, input.modelKey, input.controlKey].every((value) => typeof value === "string" && value.length > 0 && value.length <= 4096)
    || !FAMILIES.includes(input.family)) throw new Error("Invalid Jev routing observation identity");
}
function mean(previous: number, value: number, n: number): number {
  return previous + (value - previous) / n;
}
export async function recordUsage(input: UsageObservation): Promise<void> {
  validIdentity(input);
  const usage = input.usage;
  if (![usage.input, usage.output, usage.cacheRead, usage.cacheWrite].every((value) => numeric(value, 100_000_000))
    || !numeric(usage.input + usage.cacheRead + usage.cacheWrite, 100_000_000)
    || !numeric(input.durationMs, 86_400_000) || !numeric(input.timestamp)
    || (usage.reasoning !== undefined && !numeric(usage.reasoning, usage.output))
    || (usage.cacheWrite1h !== undefined && !numeric(usage.cacheWrite1h, usage.cacheWrite))) throw new Error("Invalid Jev routing usage measurements");
  await mutate((data) => {
    const id = hash(["usage", input.id]);
    if (data.ledger[id] !== undefined) return false;
    const sequence = ++data.sequence;
    data.ledger[id] = sequence;
    const key = profileKey(input.modelKey, input.controlKey, input.family);
    const previous = data.profiles[key];
    const n = Math.min(MAX_SAMPLES, (previous?.n ?? 0) + 1);
    const inputTotal = usage.input + usage.cacheRead + usage.cacheWrite;
    const writeFraction = inputTotal ? usage.cacheWrite / inputTotal : 0;
    const longFraction = usage.cacheWrite ? (usage.cacheWrite1h ?? 0) / usage.cacheWrite : 0;
    const newest = !previous || input.timestamp >= previous.lastTimestamp;
    data.profiles[key] = {
      n, input: mean(previous?.input ?? 0, inputTotal, n), output: mean(previous?.output ?? 0, usage.output, n),
      cacheWriteFraction: mean(previous?.cacheWriteFraction ?? 0, writeFraction, n),
      longWriteFraction: mean(previous?.longWriteFraction ?? 0, longFraction, n),
      latencyMs: mean(previous?.latencyMs ?? 0, input.durationMs, n),
      lastTimestamp: newest ? input.timestamp : previous?.lastTimestamp ?? 0,
      lastCacheReadFraction: newest ? (inputTotal ? usage.cacheRead / inputTotal : 0) : previous?.lastCacheReadFraction ?? 0,
      updated: sequence,
    };
    return true;
  });
}
export async function recordOutcome(input: TaskOutcome): Promise<void> {
  validIdentity(input);
  if (typeof input.taskId !== "string" || !input.taskId || input.taskId.length > 4096 || typeof input.classifierVersion !== "string"
    || input.classifierVersion.length > 4096 || !numeric(input.prediction, 1) || typeof input.success !== "boolean"
    || !["user", "contract"].includes(input.source)) throw new Error("Invalid Jev routing task outcome");
  await mutate((data) => {
    const id = hash(["physical-outcome", input.id, input.modelKey, input.controlKey]);
    const task = hash(["physical-task-pair", input.taskId, input.modelKey, input.controlKey]);
    const qualityId = hash(["user-quality-outcome", input.id, input.modelKey, input.controlKey]);
    const qualityTask = hash(["user-quality-task-pair", input.taskId, input.modelKey, input.controlKey]);
    const newPhysical = data.ledger[id] === undefined && data.ledger[task] === undefined;
    const newQuality = input.source === "user" && data.ledger[qualityId] === undefined && data.ledger[qualityTask] === undefined;
    if (!newPhysical && !newQuality) return false;
    const sequence = ++data.sequence;
    if (newPhysical) {
      // Physical reliability keeps its first observation, whether a check result or user acceptance.
      data.ledger[id] = sequence;
      data.ledger[task] = sequence;
      const key = profileKey(input.modelKey, input.controlKey, input.family);
      const previous = data.tasks[key];
      let n = previous?.n ?? 0;
      let successes = previous?.successes ?? 0;
      let user = previous?.user ?? 0;
      let contract = previous?.contract ?? 0;
      if (n === MAX_SAMPLES) {
        user = Math.floor(user / 2); contract = Math.floor(contract / 2);
        n = user + contract;
        successes = Math.min(n, Math.floor(successes / 2));
      }
      data.tasks[key] = { n: n + 1, successes: successes + Number(input.success), family: FAMILIES.indexOf(input.family),
        user: user + Number(input.source === "user"), contract: contract + Number(input.source === "contract"), updated: sequence };
    }
    if (newQuality) {
      // User quality labels have an independent first-label ledger: a check cannot consume this slot.
      data.ledger[qualityId] = sequence;
      data.ledger[qualityTask] = sequence;
      if (stableVersion(input.classifierVersion)) {
        const key = calibrationKey(input.modelKey, input.controlKey, input.family, input.classifierVersion, input.prediction);
        const previous = data.calibrations[key];
        let n = previous?.n ?? 0;
        let successes = previous?.successes ?? 0;
        let predictionSum = previous?.predictionSum ?? 0;
        if (n === MAX_SAMPLES) {
          n = Math.floor(n / 2);
          successes = Math.min(n, Math.floor(successes / 2));
          predictionSum /= 2;
        }
        data.calibrations[key] = { n: n + 1, successes: successes + Number(input.success),
          predictionSum: predictionSum + input.prediction, updated: sequence };
      }
    }
    return true;
  });
}
/** One-sided 95% Wilson bound: a conservative estimate, never a success guarantee. */
function lowerBound(successes: number, n: number): number {
  if (successes === 0) return 0;
  const z = 1.645;
  const rate = successes / n;
  const denominator = 1 + z * z / n;
  return Math.max(0, (rate + z * z / (2 * n) - z * Math.sqrt((rate * (1 - rate) + z * z / (4 * n)) / n)) / denominator);
}
export function calibratedProbability(history: RoutingHistory, model: string, control: string, family: TaskFamily, classifierVersion: string, prediction: number): { probability: number; samples: number } {
  if (!numeric(prediction, 1)) throw new Error("Invalid Jev classifier probability");
  const record = stableVersion(classifierVersion) ? history[historyData].data.calibrations[calibrationKey(model, control, family, classifierVersion, prediction)] : undefined;
  if (!record) return { probability: prediction, samples: 0 };
  return { probability: Math.min(prediction, lowerBound(record.successes, record.n)), samples: record.n };
}

export function estimatePair(history: RoutingHistory, model: Model<Api>, control: EffectiveControl, assessment: TaskAssessment, inputTokens: number, messages: readonly Message[], policy: RoutingPolicy, resolvedName?: string): { costUsd: number; latencyMs: number; expectedOutputTokens: number; expectedTurns: number; score: number } {
  const key = modelKey(model, resolvedName);
  const profile = policy.historyEnabled ? history[historyData].data.profiles[profileKey(key, control.key, assessment.family)] : undefined;
  // Unobserved output is an effort-dependent allowance, not the context-fit reserve or measured reasoning.
  const initialOutput = Math.min(model.maxTokens, Math.max(128, assessment.outputTokens) * (1 + Math.max(0, control.effortRank) * 0.5));
  const expectedOutputTokens = Math.min(model.maxTokens, profile ? profile.output : initialOutput);
  const turnsByFamily: Record<TaskFamily, number> = { mechanical: 1, coding: 3, reasoning: 2, research: 3, review: 2, unknown: 3 };
  const baseTurns = assessment.phase === "review" ? 1 : assessment.boundedExecution ? Math.min(2, turnsByFamily[assessment.family]) : turnsByFamily[assessment.family];
  const tasks = policy.historyEnabled ? history[historyData].data.tasks[profileKey(key, control.key, assessment.family)] : undefined;
  // Labeled failures support an inferred retry allowance, not measured task turns or independent-trial guarantees.
  const recoveryMultiplier = tasks && tasks.n >= policy.minimumCalibrationSamples
    ? Math.min(4, 1 / Math.max(0.25, tasks.successes / tasks.n)) : 1;
  const expectedTurns = baseTurns * recoveryMultiplier;
  const tokens = Math.max(0, inputTokens);
  let previous: Message | undefined;
  for (let index = messages.length - 1; index >= 0; index--) {
    if (messages[index].role === "assistant") { previous = messages[index]; break; }
  }
  const ttlMs = (model.promptCache?.short ?? 0) * 1000;
  const age = previous ? Date.now() - previous.timestamp : Infinity;
  const affinity = previous?.role === "assistant" && previous.provider === model.provider && previous.model === model.id && previous.api === model.api
    && profile && previous.timestamp === profile.lastTimestamp && previous.usage.cacheRead > 0 && ttlMs > 0 && age >= 0 && age <= ttlMs;
  // Discount only a bounded fraction of a witnessed cache hit; never predict a guaranteed full hit.
  const cacheRead = affinity ? tokens * Math.min(0.8, profile.lastCacheReadFraction * 0.75) : 0;
  const cacheWrite = Math.min(tokens - cacheRead, tokens * (profile?.cacheWriteFraction ?? 0));
  const usage: Usage = { input: tokens - cacheRead - cacheWrite, output: expectedOutputTokens, cacheRead, cacheWrite,
    cacheWrite1h: cacheWrite * (profile?.longWriteFraction ?? 0), totalTokens: tokens + expectedOutputTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  // calculateCost honors context-price tiers and the special long-cache-write price.
  let costUsd = calculateCost(model, usage).total;
  // Subsequent turns incur cumulative context growth at each applicable catalog price tier.
  const originalInput = usage.input;
  for (let turn = 1; turn < expectedTurns; turn++) {
    usage.input = originalInput + turn * expectedOutputTokens;
    usage.totalTokens = tokens + (turn + 1) * expectedOutputTokens;
    costUsd += calculateCost(model, usage).total * Math.min(1, expectedTurns - turn);
  }
  const perTurnLatency = profile ? profile.latencyMs : 1000 + expectedOutputTokens * 50;
  const latencyMs = perTurnLatency * expectedTurns;
  const score = costUsd + latencyMs / 1000 * policy.latencyUsdPerSecond;
  return { costUsd: Number.isFinite(costUsd) && costUsd >= 0 ? costUsd : Infinity, latencyMs, expectedOutputTokens, expectedTurns,
    score: Number.isFinite(score) && score >= 0 ? score : Infinity };
}

export function historyReport(history: RoutingHistory): string {
  const { data, loaded } = history[historyData];
  const profiles = Object.values(data.profiles);
  const calibrations = Object.values(data.calibrations);
  const taskProfiles = Object.values(data.tasks);
  const physicalOutcomes = taskProfiles.reduce((sum, profile) => sum + profile.n, 0);
  const usageSamples = profiles.reduce((sum, profile) => sum + profile.n, 0);
  const outcomeSamples = calibrations.reduce((sum, calibration) => sum + calibration.n, 0);
  const physicalUserOutcomes = taskProfiles.reduce((sum, profile) => sum + profile.user, 0);
  const physicalCheckOutcomes = taskProfiles.reduce((sum, profile) => sum + profile.contract, 0);
  const averageOutput = usageSamples ? profiles.reduce((sum, profile) => sum + profile.output * profile.n, 0) / usageSamples : 0;
  const averageLatency = usageSamples ? profiles.reduce((sum, profile) => sum + profile.latencyMs * profile.n, 0) / usageSamples : 0;
  const averageInput = usageSamples ? profiles.reduce((sum, profile) => sum + profile.input * profile.n, 0) / usageSamples : 0;
  const averagePrediction = outcomeSamples ? calibrations.reduce((sum, calibration) => sum + calibration.predictionSum, 0) / outcomeSamples : 0;
  const successRate = outcomeSamples ? calibrations.reduce((sum, calibration) => sum + calibration.successes, 0) / outcomeSamples : 0;
  const specialists = FAMILIES.flatMap((family, index) => {
    const tasks = taskProfiles.filter((task) => task.family === index);
    const n = tasks.reduce((sum, task) => sum + task.n, 0);
    const successes = tasks.reduce((sum, task) => sum + task.successes, 0);
    return n ? [`${family}: ${n} labeled pair outcomes, ${successes} passing checks or user-accepted outcomes across ${tasks.length} pair profiles`] : [];
  });
  return [loaded ? "Jev routing history: loaded schema v1." : "Jev routing history: no persisted observations loaded; using conservative estimates.",
    `${profiles.length} physical-model/native-control/task-family profiles; ${usageSamples} retained observed calls.`,
    usageSamples ? `Observed mean input ${Math.round(averageInput)}, output ${Math.round(averageOutput)} tokens (reasoning included), latency ${Math.round(averageLatency)} ms per call.` : "Output, latency, and task turns are unmeasured estimates.",
    `${taskProfiles.length} physical task-family profiles; ${physicalOutcomes} retained task/pair outcomes (${physicalCheckOutcomes} check, ${physicalUserOutcomes} user; first physical outcome per pair).`,
    `${calibrations.length} classifier-version/probability bins; ${outcomeSamples} retained stable user-quality labels; check outcomes are excluded.`,
    ...(outcomeSamples ? [`Retained user task-acceptance labels: mean classifier prediction ${(averagePrediction * 100).toFixed(1)}%, observed success ${(successRate * 100).toFixed(1)}%; labels are not an independent benchmark.`] : []),
    ...specialists,
    "Quality calibration uses explicit user task-acceptance labels and conservative lower bounds, not check results or a success guarantee; unseen or mutable classifier aliases use raw predictions.",
    "Task turns are conservative estimates; sufficient labeled failures add inferred recovery overhead, not observed completion costs. Cache affinity requires a recent witnessed same-model hit.",
    `History is bounded to ${MAX_PROFILES} performance/task profiles, ${MAX_CALIBRATIONS} calibration bins, and ${MAX_LEDGER} deduplication identities; old entries may expire.`].join("\n");
}
