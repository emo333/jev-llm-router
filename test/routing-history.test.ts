import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, test } from "node:test";
import type { Api, AssistantMessage, Model, Usage } from "@earendil-works/pi-ai";
import { routingPolicy } from "../src/router-config.ts";
import { calibratedProbability, emptyHistory, estimatePair, historyReport, loadHistory, modelKey, recordOutcome, recordUsage } from "../src/routing-history.ts";
import type { TaskOutcome, UsageObservation } from "../src/routing-history.ts";
import type { EffectiveControl, TaskAssessment } from "../src/routing-types.ts";

const originalAgentDir = process.env.PI_CODING_AGENT_DIR;
let agentDir: string;
let path: string;
before(async () => {
  agentDir = await fs.mkdtemp(join(tmpdir(), "jev-history-test-"));
  path = join(agentDir, "jev-llm-rtr-history.json");
  process.env.PI_CODING_AGENT_DIR = agentDir;
});
beforeEach(async () => { await fs.rm(path, { force: true }); });
after(async () => {
  if (originalAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = originalAgentDir;
  await fs.rm(agentDir, { recursive: true, force: true });
});

function model(overrides: Partial<Model<Api>> = {}): Model<Api> {
  return { id: "physical-v1", provider: "test", name: "Display", api: "openai-completions", baseUrl: "https://endpoint.invalid/v1",
    reasoning: true, input: ["text"], contextWindow: 100_000, maxTokens: 32_000,
    promptCache: { short: 300 }, cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1.25 }, ...overrides };
}
const low: EffectiveControl = { level: "low", native: "low", key: "effort:low", effortRank: 2, outputReserve: 8192 };
const high: EffectiveControl = { level: "high", native: "high", key: "effort:high", effortRank: 4, outputReserve: 16384 };
const assessment: TaskAssessment = { family: "coding", risk: "low", verifiable: true, boundedExecution: false, phase: "execution", outputTokens: 1000 };
const policy = routingPolicy({ thinkingLevelCaps: {} });
function usage(overrides: Partial<Usage> = {}): Usage {
  return { input: 1000, output: 400, cacheRead: 0, cacheWrite: 0, totalTokens: 1400,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }, ...overrides };
}
function observation(overrides: Partial<UsageObservation> = {}): UsageObservation {
  return { id: "call-1", modelKey: modelKey(model()), controlKey: low.key, family: "coding", usage: usage(), durationMs: 2500, timestamp: Date.now(), ...overrides };
}
function outcome(overrides: Partial<TaskOutcome> = {}): TaskOutcome {
  return { id: "label-1", taskId: "task-1", modelKey: modelKey(model()), controlKey: low.key, family: "coding",
    classifierVersion: "jev-2026-10-01", prediction: 0.99, success: true, source: "user", ...overrides };
}
function assistant(observed: UsageObservation, overrides: Partial<AssistantMessage> = {}): AssistantMessage {
  return { role: "assistant", content: [{ type: "text", text: "private answer" }], api: "openai-completions", provider: "test", model: "physical-v1",
    usage: observed.usage, stopReason: "stop", timestamp: observed.timestamp, ...overrides };
}

test("unseen model/control pairs retain raw probabilities and effort-dependent total-task estimates", async () => {
  const history = await loadHistory();
  assert.deepEqual(calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-2026-10-01", 0.99), { probability: 0.99, samples: 0 });
  const cheap = estimatePair(history, model(), low, assessment, 5000, [], policy);
  const expensive = estimatePair(history, model(), high, assessment, 5000, [], policy);
  assert.equal(cheap.expectedTurns, 3);
  assert.ok(expensive.expectedOutputTokens > cheap.expectedOutputTokens);
  assert.ok(expensive.costUsd > cheap.costUsd);
  assert.ok(cheap.costUsd > (5000 + cheap.expectedOutputTokens * 2) / 1_000_000);
  assert.match(historyReport(history), /unmeasured estimates/);
});

test("actual output and latency replace pair estimates without double-counting reasoning", async () => {
  await recordUsage(observation({ usage: usage({ output: 1000, reasoning: 600 }), durationMs: 9000 }));
  const history = await loadHistory();
  const pair = estimatePair(history, model(), low, { ...assessment, family: "coding", phase: "review" }, 1000, [], { ...policy, latencyUsdPerSecond: 0.01 });
  assert.equal(pair.expectedOutputTokens, 1000);
  assert.equal(pair.costUsd, 0.003);
  assert.equal(pair.latencyMs, 9000);
  assert.equal(pair.score, pair.costUsd + 0.09);
  const otherControl = estimatePair(history, model(), high, assessment, 1000, [], policy);
  const otherFamily = estimatePair(history, model(), low, { ...assessment, family: "research" }, 1000, [], policy);
  assert.notEqual(otherControl.expectedOutputTokens, 1000);
  assert.notEqual(otherFamily.expectedOutputTokens, 1000);
});

test("profile means merge observations and repeated usage identities are idempotent", async () => {
  await Promise.all([
    recordUsage(observation({ id: "a", usage: usage({ output: 100 }), durationMs: 1000 })),
    recordUsage(observation({ id: "b", usage: usage({ output: 300 }), durationMs: 3000 })),
    recordUsage(observation({ id: "a", usage: usage({ output: 9999 }), durationMs: 9999 })),
  ]);
  const pair = estimatePair(await loadHistory(), model(), low, { ...assessment, phase: "review" }, 1000, [], policy);
  assert.equal(pair.expectedOutputTokens, 200);
  assert.equal(pair.latencyMs, 2000);
});

test("cache affinity requires observed usage, the same fingerprint and previous response, and a known recent TTL", async () => {
  const observed = observation({ usage: usage({ input: 200, cacheRead: 800 }), timestamp: Date.now() });
  await recordUsage(observed);
  const history = await loadHistory();
  const task = { ...assessment, phase: "review" as const };
  const cold = estimatePair(history, model(), low, task, 1000, [], policy);
  const warm = estimatePair(history, model(), low, task, 1000, [assistant(observed)], policy);
  assert.ok(warm.costUsd < cold.costUsd);
  assert.ok(warm.costUsd > (1000 * 0.1 + 400 * 2) / 1_000_000, "a witnessed cache hit is not a guaranteed full future hit");
  for (const previous of [assistant(observed, { provider: "other" }), assistant(observed, { model: "physical-v2" }),
    assistant(observed, { api: "openai-responses" }), assistant(observed, { timestamp: observed.timestamp - 1 }),
    assistant(observed, { usage: usage() })]) {
    assert.equal(estimatePair(history, model(), low, task, 1000, [previous], policy).costUsd, cold.costUsd);
  }
  const unknownTTL = model({ promptCache: undefined });
  await recordUsage(observation({ id: "unknown-cache", modelKey: modelKey(unknownTTL), timestamp: observed.timestamp, usage: observed.usage }));
  const unknownHistory = await loadHistory();
  assert.equal(estimatePair(unknownHistory, unknownTTL, low, task, 1000, [assistant(observed)], policy).costUsd,
    estimatePair(unknownHistory, unknownTTL, low, task, 1000, [], policy).costUsd);
});

test("expired, future-dated, and other-endpoint cache observations cannot discount cost", async () => {
  for (const timestamp of [Date.now() - 301_000, Date.now() + 60_000]) {
    const observed = observation({ id: String(timestamp), timestamp, usage: usage({ input: 200, cacheRead: 800 }) });
    await recordUsage(observed);
    const history = await loadHistory();
    const task = { ...assessment, phase: "review" as const };
    assert.equal(estimatePair(history, model(), low, task, 1000, [assistant(observed)], policy).costUsd,
      estimatePair(history, model(), low, task, 1000, [], policy).costUsd);
  }
  const changed = model({ baseUrl: "https://different.invalid/v1" });
  const previous = assistant(observation());
  assert.deepEqual(estimatePair(await loadHistory(), changed, low, assessment, 1000, [previous], policy),
    estimatePair(emptyHistory(), changed, low, assessment, 1000, [previous], policy));
});

test("catalog tier pricing and long-cache-write semantics are retained", async () => {
  const tiered = model({ cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 1.25,
    tiers: [{ inputTokensAbove: 1000, input: 4, output: 8, cacheRead: 0.4, cacheWrite: 5 }] } });
  await recordUsage(observation({ modelKey: modelKey(tiered), usage: usage({ input: 0, output: 100, cacheWrite: 1000, cacheWrite1h: 1000 }) }));
  const pair = estimatePair(await loadHistory(), tiered, low, { ...assessment, phase: "review" }, 2000, [], policy);
  assert.equal(pair.expectedOutputTokens, 100);
  assert.equal(pair.costUsd, (2000 * 8 + 100 * 8) / 1_000_000);
});

test("physical fingerprints distinguish endpoint, configuration, resolved model, and release but ignore UI labels", () => {
  const key = modelKey(model());
  assert.equal(modelKey(model({ name: "Different label" })), key);
  for (const changed of [model({ id: "physical-v2" }), model({ baseUrl: "https://other.invalid/v1" }), model({ api: "openai-responses" }),
    model({ thinkingLevelMap: { high: "max" } }), model({ compat: { supportsReasoningEffort: false } })]) assert.notEqual(modelKey(changed), key);
  assert.notEqual(modelKey(model(), "actual-local-model-v1"), modelKey(model(), "actual-local-model-v2"));
  assert.equal(modelKey(model({ compat: { supportsStore: false, supportsDeveloperRole: true } })),
    modelKey(model({ compat: { supportsDeveloperRole: true, supportsStore: false } })));
});

test("outcome calibration isolates version, physical model, native control, task family, and prediction bin", async () => {
  await recordOutcome(outcome({ success: false }));
  const history = await loadHistory();
  const calibrated = calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-2026-10-01", 0.99);
  assert.equal(calibrated.samples, 1);
  assert.equal(calibrated.probability, 0);
  for (const [key, control, family, version, prediction] of [
    [modelKey(model({ id: "physical-v2" })), low.key, "coding", "jev-2026-10-01", 0.99],
    [modelKey(model()), high.key, "coding", "jev-2026-10-01", 0.99],
    [modelKey(model()), low.key, "review", "jev-2026-10-01", 0.99],
    [modelKey(model()), low.key, "coding", "jev-2026-10-02", 0.99],
    [modelKey(model()), low.key, "coding", "jev-2026-10-01", 0.8],
  ] as const) assert.deepEqual(calibratedProbability(history, key, control, family, version, prediction), { probability: prediction, samples: 0 });
});

test("user and contract labels for the same task cannot inflate quality samples, including concurrent writes", async () => {
  await Promise.all([recordOutcome(outcome()), recordOutcome(outcome({ id: "duplicate-contract", source: "contract" })),
    recordOutcome(outcome({ id: "same-task-different-version", classifierVersion: "jev-2026-10-02" }))]);
  const history = await loadHistory();
  assert.equal(calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-2026-10-01", 0.99).samples, 1);
  assert.equal(calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-2026-10-02", 0.99).samples, 0);
  await recordOutcome(outcome({ taskId: "other-task" }));
  assert.equal(calibratedProbability(await loadHistory(), modelKey(model()), low.key, "coding", "jev-2026-10-01", 0.99).samples, 1);
});

test("successful labels retain uncertainty and never improve the raw prediction", async () => {
  await Promise.all(Array.from({ length: 30 }, (_, index) => recordOutcome(outcome({ id: `label-${index}`, taskId: `task-${index}` }))));
  const calibrated = calibratedProbability(await loadHistory(), modelKey(model()), low.key, "coding", "jev-2026-10-01", 0.99);
  assert.equal(calibrated.samples, 30);
  assert.ok(calibrated.probability > 0.9 && calibrated.probability < 0.99);
  assert.ok(calibrated.probability < 1);
});

test("mutable classifier aliases do not acquire false version-stable calibration", async () => {
  for (const version of ["typesafe/jev-latest", "CURRENT", "auto", ""]) {
    await recordOutcome(outcome({ id: version || "empty", taskId: version || "empty", classifierVersion: version }));
    assert.deepEqual(calibratedProbability(await loadHistory(), modelKey(model()), low.key, "coding", version, 0.99), { probability: 0.99, samples: 0 });
  }
});

test("specialist failure profiles infer bounded recovery cost only with sufficient labels", async () => {
  const threshold = { ...policy, minimumCalibrationSamples: 4 };
  const baseline = estimatePair(emptyHistory(), model(), low, assessment, 1000, [], threshold);
  for (let index = 0; index < 3; index++) await recordOutcome(outcome({ id: `fail-${index}`, taskId: `fail-${index}`, success: false }));
  assert.equal(estimatePair(await loadHistory(), model(), low, assessment, 1000, [], threshold).expectedTurns, baseline.expectedTurns);
  await recordOutcome(outcome({ id: "success", taskId: "success" }));
  const history = await loadHistory();
  const pair = estimatePair(history, model(), low, assessment, 1000, [], threshold);
  assert.equal(pair.expectedTurns, 12);
  assert.ok(pair.costUsd > baseline.costUsd);
  assert.equal(estimatePair(history, model(), low, { ...assessment, family: "research" }, 1000, [], threshold).expectedTurns, 3);
  assert.match(historyReport(history), /coding: 4 labeled pair outcomes, 1 passing checks or user-accepted outcomes/);
});

test("disabling history leaves deterministic cold estimates and does not reuse observed profiles", async () => {
  await recordUsage(observation({ usage: usage({ output: 50 }), durationMs: 100 }));
  const disabled = { ...policy, historyEnabled: false };
  assert.deepEqual(estimatePair(await loadHistory(), model(), low, assessment, 1000, [], disabled),
    estimatePair(emptyHistory(), model(), low, assessment, 1000, [], disabled));
});

test("queued usage and outcomes preserve both streams without persisting text or secrets", async () => {
  await Promise.all([recordUsage(observation({ id: "SECRET-PROMPT-IN-ID" })), recordOutcome(outcome({ taskId: "SECRET-TASK-TEXT" }))]);
  const content = await fs.readFile(path, "utf8");
  assert.ok(!content.includes("SECRET"));
  assert.ok(!content.includes("endpoint.invalid"));
  const data = JSON.parse(content);
  assert.equal(Object.keys(data.profiles).length, 1);
  assert.equal(Object.keys(data.calibrations).length, 1);
  assert.equal(Object.keys(data.tasks).length, 1);
  for (const section of [data.profiles, data.calibrations, data.tasks]) {
    for (const [key, aggregate] of Object.entries(section)) {
      assert.match(key, /^[a-f0-9]{64}$/);
      assert.ok(Object.values(aggregate as Record<string, unknown>).every((value) => typeof value === "number"));
    }
  }
});

test("an interrupted atomic rename preserves committed history and a retry is not falsely deduplicated", async (t) => {
  await recordUsage(observation());
  const committed = await fs.readFile(path, "utf8");
  const rename = t.mock.method(fs, "rename", async () => { throw new Error("interrupted rename"); });
  await assert.rejects(recordUsage(observation({ id: "new-call", usage: usage({ output: 600 }) })), /Could not save.*interrupted rename/);
  assert.equal(await fs.readFile(path, "utf8"), committed);
  assert.deepEqual((await fs.readdir(agentDir)).filter((name) => name.endsWith(".tmp")), []);
  rename.mock.restore();
  await recordUsage(observation({ id: "new-call", usage: usage({ output: 600 }) }));
  assert.equal(estimatePair(await loadHistory(), model(), low, assessment, 1000, [], policy).expectedOutputTokens, 500);
});

test("corrupt or unsupported history is reported and cannot be overwritten as if empty", async () => {
  for (const content of ["{broken", JSON.stringify({ version: 99 }), JSON.stringify({ version: 1, sequence: 0, profiles: {}, calibrations: {}, tasks: {}, ledger: {}, prompt: "secret" })]) {
    await fs.writeFile(path, content);
    await assert.rejects(loadHistory(), /Invalid Jev routing history/);
    await assert.rejects(recordUsage(observation()), /Invalid Jev routing history/);
    assert.equal(await fs.readFile(path, "utf8"), content);
  }
});

test("non-numeric measurements and reasoning exceeding the inclusive output total are rejected", async () => {
  await assert.rejects(recordUsage(observation({ usage: usage({ reasoning: 401, output: 400 }) })), /Invalid.*usage measurements/);
  await assert.rejects(recordUsage(observation({ durationMs: NaN })), /Invalid.*usage measurements/);
  await assert.rejects(recordOutcome(outcome({ prediction: 1.01 })), /Invalid.*task outcome/);
});

test("deduplication ledger remains bounded while preserving recently committed identities", async () => {
  await recordUsage(observation());
  const data = JSON.parse(await fs.readFile(path, "utf8"));
  data.sequence = 8192;
  data.ledger = Object.fromEntries(Array.from({ length: 8192 }, (_, index) => [createHash("sha256").update(`old-${index}`).digest("hex"), index + 1]));
  await fs.writeFile(path, JSON.stringify(data));
  await recordUsage(observation({ id: "fresh" }));
  const saved = JSON.parse(await fs.readFile(path, "utf8"));
  assert.equal(Object.keys(saved.ledger).length, 8192);
  await recordUsage(observation({ id: "fresh", usage: usage({ output: 9999 }) }));
  assert.equal(estimatePair(await loadHistory(), model(), low, assessment, 1000, [], policy).expectedOutputTokens, 400);
});

test("performance profiles are bounded and evicted pairs revert honestly to estimates", async () => {
  await recordUsage(observation());
  const data = JSON.parse(await fs.readFile(path, "utf8"));
  const originalProfile = Object.values(data.profiles)[0];
  data.sequence = 256;
  data.profiles = Object.fromEntries(Array.from({ length: 256 }, (_, index) => [createHash("sha256").update(`old-pair-${index}`).digest("hex"), { ...(originalProfile as object), updated: index + 1 }]));
  await fs.writeFile(path, JSON.stringify(data));
  await recordUsage(observation({ id: "fresh-pair" }));
  const saved = JSON.parse(await fs.readFile(path, "utf8"));
  assert.equal(Object.keys(saved.profiles).length, 256);
  assert.equal(estimatePair(await loadHistory(), model(), low, assessment, 1000, [], policy).expectedOutputTokens, 400);
  assert.deepEqual(estimatePair(await loadHistory(), model({ id: "unseen" }), low, assessment, 1000, [], policy),
    estimatePair(emptyHistory(), model({ id: "unseen" }), low, assessment, 1000, [], policy));
});

test("mutable physical model aliases do not reuse calibration after a new identity epoch", async () => {
  const alias = model({ id: "physical-latest" });
  const key = modelKey(alias);
  assert.equal(modelKey(alias), key);
  await recordOutcome(outcome({ modelKey: key }));
  // Exercise a fresh module identity epoch; a static import would reuse the existing module and nonce.
  const nextModule = await import(new URL("../src/routing-history.ts?new-identity-epoch", import.meta.url).href);
  assert.notEqual(nextModule.modelKey(alias), key);
  assert.equal(nextModule.modelKey(model()), modelKey(model()));
  assert.equal(nextModule.modelKey(alias, "resolved-release-2026-10-01"), modelKey(alias, "resolved-release-2026-10-01"));
  assert.deepEqual(nextModule.calibratedProbability(await nextModule.loadHistory(), nextModule.modelKey(alias), low.key, "coding", "jev-2026-10-01", 0.99),
    { probability: 0.99, samples: 0 });
});

test("invalid numeric aggregates and oversized history are honestly rejected", async () => {
  await recordUsage(observation());
  const data = JSON.parse(await fs.readFile(path, "utf8"));
  const profile = Object.values(data.profiles)[0] as Record<string, unknown>;
  profile.output = "not a measurement";
  await fs.writeFile(path, JSON.stringify(data));
  await assert.rejects(loadHistory(), /Invalid Jev routing history.*numeric/);
  await fs.writeFile(path, " ".repeat(2_000_001));
  await assert.rejects(loadHistory(), /Could not read.*bounded size limit/);
});

test("calibration bins are bounded without treating an evicted bin as trustworthy data", async () => {
  await recordOutcome(outcome());
  const data = JSON.parse(await fs.readFile(path, "utf8"));
  const aggregate = Object.values(data.calibrations)[0];
  data.sequence = 1024;
  data.calibrations = Object.fromEntries(Array.from({ length: 1024 }, (_, index) => [createHash("sha256").update(`old-bin-${index}`).digest("hex"), { ...(aggregate as object), updated: index + 1 }]));
  await fs.writeFile(path, JSON.stringify(data));
  await recordOutcome(outcome({ id: "fresh-label", taskId: "fresh-task", classifierVersion: "jev-2026-10-02" }));
  const saved = JSON.parse(await fs.readFile(path, "utf8"));
  assert.equal(Object.keys(saved.calibrations).length, 1024);
  const history = await loadHistory();
  assert.equal(calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-2026-10-02", 0.99).samples, 1);
  assert.deepEqual(calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-2026-10-01", 0.99), { probability: 0.99, samples: 0 });
});

test("mutable classifier aliases still learn physical specialist outcomes without stable calibration", async () => {
  for (let index = 0; index < 4; index++) {
    await recordOutcome(outcome({ id: `mutable-${index}`, taskId: `mutable-${index}`, classifierVersion: "jev-latest", success: index === 3 }));
  }
  const history = await loadHistory();
  assert.deepEqual(calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-latest", 0.99), { probability: 0.99, samples: 0 });
  const pair = estimatePair(history, model(), low, assessment, 1000, [], { ...policy, minimumCalibrationSamples: 4 });
  assert.equal(pair.expectedTurns, 12);
  const data = JSON.parse(await fs.readFile(path, "utf8"));
  assert.equal(Object.keys(data.calibrations).length, 0);
  const physical = Object.values(data.tasks)[0] as { n: number; successes: number };
  assert.equal(physical.n, 4);
  assert.equal(physical.successes, 1);
  assert.ok(historyReport(history).includes("4 retained task/pair outcomes"));
  assert.match(historyReport(history), /0 retained stable user-quality labels/);
});

test("the same task learns separate cheap failure, stronger-control success, and recovery-model success", async () => {
  const cheap = outcome({ success: false, source: "contract" });
  const strongerControl = outcome({ controlKey: high.key });
  const strongerModel = outcome({ modelKey: modelKey(model({ id: "recovery-v2" })) });
  // Deliberately reuse the label ID: event deduplication is also scoped to the physical pair.
  await Promise.all([recordOutcome(cheap), recordOutcome(strongerControl), recordOutcome(strongerModel)]);
  const history = await loadHistory();
  assert.deepEqual(calibratedProbability(history, cheap.modelKey, low.key, "coding", cheap.classifierVersion, 0.99), { probability: 0.99, samples: 0 });
  assert.equal(calibratedProbability(history, cheap.modelKey, high.key, "coding", cheap.classifierVersion, 0.99).samples, 1);
  assert.equal(calibratedProbability(history, strongerModel.modelKey, low.key, "coding", cheap.classifierVersion, 0.99).samples, 1);
  const data = JSON.parse(await fs.readFile(path, "utf8"));
  const profiles = Object.values(data.tasks) as { n: number; successes: number }[];
  assert.equal(profiles.length, 3);
  assert.equal(profiles.reduce((sum, profile) => sum + profile.n, 0), 3);
  assert.equal(profiles.reduce((sum, profile) => sum + profile.successes, 0), 2);
});

test("contract-first physical observation leaves an independent slot for the first user-quality label", async () => {
  await recordOutcome(outcome({ success: false, source: "contract", classifierVersion: "jev-2026-10-02" }));
  await recordOutcome(outcome({ id: "later-user-acceptance", success: true, source: "user", classifierVersion: "jev-2026-10-02" }));
  await recordOutcome(outcome({ id: "duplicate-user", success: false, source: "user", classifierVersion: "jev-2026-10-02" }));
  const history = await loadHistory();
  const quality = calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-2026-10-02", 0.99);
  assert.equal(quality.samples, 1);
  assert.ok(quality.probability > 0 && quality.probability < 0.99);
  const data = JSON.parse(await fs.readFile(path, "utf8"));
  const physical = Object.values(data.tasks)[0] as { n: number; successes: number; user: number; contract: number };
  assert.equal(physical.n, 1);
  assert.equal(physical.successes, 0);
  assert.equal(physical.user, 0);
  assert.equal(physical.contract, 1);
  assert.ok(historyReport(history).includes("1 retained task/pair outcomes"));
  assert.ok(historyReport(history).includes("1 retained stable user-quality labels"));
});

test("generic passing check contracts never certify calibrated quality for a cheap phase downgrade", async () => {
  await Promise.all(Array.from({ length: 40 }, (_, index) => recordOutcome(outcome({
    id: `check-${index}`, taskId: `check-${index}`, source: "contract", prediction: 1, success: true,
  }))));
  const history = await loadHistory();
  const quality = calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-2026-10-01", 1);
  assert.deepEqual(quality, { probability: 1, samples: 0 });
  assert.equal(quality.samples >= policy.minimumCalibrationSamples && quality.probability >= policy.qualityThreshold, false);
  const data = JSON.parse(await fs.readFile(path, "utf8"));
  const physical = Object.values(data.tasks)[0] as { n: number; successes: number; contract: number };
  assert.equal(physical.n, 40);
  assert.equal(physical.successes, 40);
  assert.equal(physical.contract, 40);
  assert.equal(Object.keys(data.calibrations).length, 0);
  assert.ok(historyReport(history).includes("40 check, 0 user"));
});

test("contract failure can affect physical recovery cost without inventing a user-quality label", async () => {
  await recordOutcome(outcome({ source: "contract", success: false }));
  const history = await loadHistory();
  const quality = calibratedProbability(history, modelKey(model()), low.key, "coding", "jev-2026-10-01", 0.99);
  assert.deepEqual(quality, { probability: 0.99, samples: 0 });
  assert.equal(estimatePair(history, model(), low, assessment, 1000, [], { ...policy, minimumCalibrationSamples: 1 }).expectedTurns, 12);
});
