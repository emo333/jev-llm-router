import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { Api, Model } from "@earendil-works/pi-ai";
import { effectiveControls } from "../src/effective-controls.ts";
import { describeThinkingCap, loadRouterConfig, routingPolicy, thinkingLevelsForModel } from "../src/router-config.ts";

function model(overrides: Partial<Model<Api>> = {}): Model<Api> {
  return { id: "reasoner", name: "Reasoner", provider: "test", api: "openai-responses", baseUrl: "https://example.invalid", reasoning: true, input: ["text"], contextWindow: 200000, maxTokens: 64000, cost: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 }, ...overrides };
}

test("managed Anthropic effort is forced high even when Pi requests off", () => {
  const m = model({ api: "anthropic-messages", compat: { supportsMidConvoEffort: true, forceAdaptiveThinking: true } });
  assert.deepEqual(effectiveControls(m, "medium"), []);
  const controls = effectiveControls(m, "high");
  assert.equal(controls.length, 1);
  assert.equal(controls[0].level, "high");
  assert.equal(controls[0].native, "adaptive:high");
  assert.equal(controls[0].effortRank, 4);
  assert.match(describeThinkingCap(m, { thinkingLevelCaps: { "test/reasoner": "low" } }), /excluded/);
});

test("native effort aliases are deduplicated and cannot bypass caps", () => {
  const m = model({ thinkingLevelMap: { minimal: "low", low: "low", medium: "high", high: "high", xhigh: "high", max: "high" } });
  assert.deepEqual(thinkingLevelsForModel(m, { thinkingLevelCaps: {} }), ["off", "low", "high"]);
  assert.deepEqual(effectiveControls(m, "minimal").map((c) => c.level), ["off"]);
  assert.deepEqual(effectiveControls(m, "medium").map((c) => c.level), ["off", "low"]);
  const low = effectiveControls(m, "low").find((c) => c.level === "low")!;
  assert.equal(low.key, effectiveControls(m).find((c) => c.level === "low")!.key);
});

test("nonreasoning models expose only off", () => {
  assert.deepEqual(effectiveControls(model({ reasoning: false }), "off").map((c) => [c.level, c.effortRank]), [["off", 0]]);
});

test("adaptive Anthropic minimal and low share the native low effort", () => {
  const m = model({ api: "anthropic-messages", compat: { forceAdaptiveThinking: true } });
  assert.deepEqual(effectiveControls(m).map((c) => c.level), ["off", "low", "medium", "high"]);
  assert.deepEqual(effectiveControls(m, "minimal").map((c) => c.level), ["off"]);
});

test("budget-based Anthropic reserves answer room and collapses clipped budgets", () => {
  const m = model({ api: "anthropic-messages", maxTokens: 3072 });
  const controls = effectiveControls(m);
  assert.deepEqual(controls.map((c) => c.native), ["disabled", "budget:1024", "budget:2048"]);
  assert.equal(controls.at(-1)!.outputReserve, 3072);
  assert.equal(controls.at(-1)!.level, "low");
});

test("Google native budgets and discrete mappings are reflected", () => {
  const pro = model({ api: "google-generative-ai", id: "gemini-2.5-pro" });
  assert.equal(effectiveControls(pro).at(-1)!.native, "budget:32768");
  assert.equal(effectiveControls(pro).at(-1)!.outputReserve, 33792);
  const discrete = model({ api: "google-vertex", id: "gemini-3-pro-preview", thinkingLevelMap: { off: null, minimal: "low", low: "low", medium: "high", high: "high" } });
  assert.deepEqual(effectiveControls(discrete, "medium").map((c) => c.native), ["level:low"]);
  const dynamic = model({ api: "google-generative-ai", id: "older-gemini" });
  assert.deepEqual(effectiveControls(dynamic, "low").map((c) => c.native), ["budget:0"]);
});

test("completion switches cannot pretend to expose multiple effort levels", () => {
  const m = model({ api: "openai-completions", provider: "zai", baseUrl: "https://api.z.ai/api/paas/v4" });
  assert.deepEqual(effectiveControls(m).map((c) => c.level), ["off", "high"]);
  assert.deepEqual(effectiveControls(m, "low").map((c) => c.level), ["off"]);
  const capped = model({ api: "openai-completions", compat: { thinkingFormat: "qwen", supportsReasoningEffort: false, thinkingTokenBudgetField: "thinking_budget" }, maxTokens: 3072 });
  const enabled = effectiveControls(capped).filter((c) => c.effortRank > 0);
  assert.equal(enabled.length, 2);
  assert.equal(enabled.at(-1)!.outputReserve, 3072);
  assert.match(enabled.at(-1)!.native, /"budget":2048/);
});

test("chat-template controls resolve actual switch and budget variables", () => {
  const m = model({ api: "openai-completions", compat: { thinkingFormat: "chat-template", chatTemplateKwargs: { enable_thinking: { $var: "thinking.enabled" }, budget: { $var: "thinking.budget", omitWhenOff: true } } } });
  const off = effectiveControls(m, "off")[0];
  // Custom switch names still carry a true/false control on the wire.
  assert.ok(off);
  assert.match(effectiveControls(m).at(-1)!.native, /16384/);
});

test("Bedrock adaptive Claude uses native effort and unsupported providers expose no fake knobs", () => {
  const claude = model({ api: "bedrock-converse-stream", id: "us.anthropic.claude-opus-4-7-v1", name: "Claude Opus 4.7", thinkingLevelMap: { xhigh: "max", max: "max" } });
  assert.equal(effectiveControls(claude).find((c) => c.level === "xhigh")!.native, "adaptive:xhigh");
  const other = model({ api: "bedrock-converse-stream", name: "Nova", id: "amazon.nova" });
  assert.deepEqual(effectiveControls(other, "medium"), []);
  assert.equal(effectiveControls(other).length, 1);
});

test("Mistral prompt-mode reasoning is one real enabled option", () => {
  const m = model({ api: "mistral-conversations" });
  assert.deepEqual(effectiveControls(m).map((c) => c.level), ["off", "high"]);
  assert.deepEqual(effectiveControls(m, "low").map((c) => c.level), ["off"]);
});

test("routing policy rejects invalid thresholds, limits, types and obsolete settings", () => {
  for (const policy of [{ qualityThreshold: 1.1 }, { protectedThreshold: .5 }, { latencyUsdPerSecond: -1 }, { contextSafetyTokens: 1.5 }, { minimumCalibrationSamples: 0 }, { phaseRouting: true }, { historyEnabled: "yes" }, { unexpected: true }]) {
    assert.throws(() => routingPolicy({ thinkingLevelCaps: {}, policy } as Parameters<typeof routingPolicy>[0]), /Invalid/);
  }
});

test("configuration validates all features even when caps are omitted and preserves unrelated fields", async () => {
  const directory = await mkdtemp(join(tmpdir(), "jev-controls-"));
  const path = join(directory, "config.json");
  try {
    assert.deepEqual(await loadRouterConfig(path), { thinkingLevelCaps: {} });
    await writeFile(path, JSON.stringify({ unrelated: { keep: true }, classifier: { provider: "typesafe", id: "jev-latest" }, verification: { command: "npm test", timeoutMs: 10000, maxAttempts: 2 }, policy: { historyEnabled: false } }));
    const config = await loadRouterConfig(path);
    assert.deepEqual(config.unrelated, { keep: true });
    assert.equal(config.classifier!.id, "jev-latest");
    assert.equal(routingPolicy(config).historyEnabled, false);
    for (const invalid of [{ policy: { maxEscalations: -1 } }, { classifier: { provider: "", id: "jev" } }, { verification: { command: "", timeoutMs: 1, maxAttempts: 1 } }, { verification: { command: "true", timeoutMs: 0, maxAttempts: 1 } }, { thinkingLevelCaps: { "test/reasoner": "bogus" } }]) {
      await writeFile(path, JSON.stringify(invalid));
      await assert.rejects(loadRouterConfig(path), /Invalid/);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("late sampling overrides enforce the native effort rather than the requested Pi level", () => {
  for (const m of [
    model({ samplingParams: { reasoning: { effort: "high" } } }),
    model({ api: "azure-openai-responses", samplingParams: { reasoning: { effort: "high" } } }),
    model({ api: "openai-completions", samplingParams: { reasoning_effort: "high" } }),
  ]) {
    assert.deepEqual(effectiveControls(m, "medium"), []);
    const controls = effectiveControls(m, "high");
    assert.equal(controls.length, 1);
    assert.equal(controls[0].effortRank, 4);
    assert.equal(controls[0].level, "high");
    assert.equal(controls[0].native, "override:effort:high");
    assert.match(describeThinkingCap(m, { thinkingLevelCaps: { "test/reasoner": "low" } }), /excluded/);
  }
  const disabled = model({ samplingParams: { reasoning: { effort: "none" } } });
  assert.deepEqual(effectiveControls(disabled, "off").map((c) => c.effortRank), [0]);
});

test("ambiguous overridden native switches, templates, and budgets cannot claim capped eligibility", () => {
  for (const samplingParams of [
    { thinking: { type: "enabled" } },
    { chat_template_kwargs: { enable_thinking: true } },
    { thinking_budget: 32768 },
    { reasoning_effort: "high", enable_thinking: true },
    { reasoning_effort: null },
  ]) {
    const m = model({ api: "openai-completions", compat: { thinkingFormat: "qwen", supportsReasoningEffort: false }, samplingParams });
    assert.deepEqual(effectiveControls(m, "high"), []);
    assert.deepEqual(effectiveControls(m, "max"), []);
    assert.equal(effectiveControls(m).length, 1);
    assert.equal(effectiveControls(m)[0].level, "high");
    assert.equal(effectiveControls(m)[0].outputReserve, m.maxTokens);
  }
  const ignored = model({ api: "anthropic-messages", samplingParams: { reasoning_effort: "max" } });
  assert.ok(effectiveControls(ignored, "low").length > 0);
});

test("Google discrete models with off unsupported cannot be dispatched as off", () => {
  const pro = model({ api: "google-generative-ai", id: "gemini-3-pro-preview", thinkingLevelMap: { off: null, minimal: null, low: "low", medium: "high", high: "high" } });
  assert.deepEqual(effectiveControls(pro, "off"), []);
  assert.deepEqual(effectiveControls(pro, "minimal"), []);
  assert.deepEqual(effectiveControls(pro, "low").map((c) => [c.level, c.native]), [["low", "level:low"]]);
});

test("Copilot omitted off effort cannot masquerade as disabled reasoning", () => {
  const copilot = model({ provider: "github-copilot", thinkingLevelMap: { xhigh: "xhigh" } });
  assert.deepEqual(effectiveControls(copilot, "off"), []);
  assert.ok(!effectiveControls(copilot, "medium").some((c) => c.level === "off"));
  const inherited = effectiveControls(copilot).find((c) => c.native === "provider-default")!;
  assert.equal(inherited.effortRank, 5);
  assert.equal(inherited.outputReserve, copilot.maxTokens);
});
