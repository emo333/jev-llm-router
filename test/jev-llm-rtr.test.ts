import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, test } from "node:test";
import type { Api, ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierResult, Message, Model, ModelsClassifierOptions } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ExtensionVirtualModel, ModelRouteRequest, ScopedModel, SessionEntry } from "@earendil-works/pi-coding-agent";
import extension, { route } from "../src/jev-llm-rtr.ts";
import { execCommand } from "../node_modules/@earendil-works/pi-coding-agent/dist/core/exec.js";
import { calibratedProbability, loadHistory } from "../src/routing-history.ts";
import type { RouteState } from "../src/routing-types.ts";

const originalAgentDir = process.env.PI_CODING_AGENT_DIR;
let testAgentDir: string;
before(async () => {
  testAgentDir = await mkdtemp(join(tmpdir(), "jev-router-test-"));
  process.env.PI_CODING_AGENT_DIR = testAgentDir;
});
beforeEach(async () => {
  await rm(join(testAgentDir, "jev-llm-rtr.json"), { force: true });
  await rm(join(testAgentDir, "jev-llm-rtr-history.json"), { force: true });
});
after(async () => {
  if (originalAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = originalAgentDir;
  await rm(testAgentDir, { recursive: true, force: true });
});

async function writeRouterConfig(config: unknown): Promise<void> {
  await writeFile(join(testAgentDir, "jev-llm-rtr.json"), JSON.stringify(config));
}

function model(id: string, outputPrice = 1, overrides: Partial<Model<Api>> = {}): Model<Api> {
  return {
    id,
    name: id,
    provider: "test",
    api: "openai-responses",
    baseUrl: "https://example.invalid",
    reasoning: true,
    input: ["text"],
    contextWindow: 200_000,
    maxTokens: 32_000,
    cost: { input: 1, output: outputPrice, cacheRead: 0, cacheWrite: 0 },
    ...overrides,
  };
}
const fast = model("fast");
const strong = model("strong", 10);
const virtual = model("auto", 0, { provider: "jev", api: "pi-virtual" });
const jev: ClassifierModel<string> = {
  id: "jev-latest",
  name: "Jev",
  provider: "typesafe",
  api: "typesafe-system-one",
  type: "classifier",
  baseUrl: "https://example.invalid",
  input: ["text"],
  contextWindow: 100_000,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
function choice(probabilities: Record<string, number>, selected?: string): ClassifierAnswer {
  const best = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0];
  return { type: "choice", choice: selected ?? best[0], probabilities, confidence: best[1] };
}
function result(answers: Record<string, ClassifierAnswer>): ClassifierResult {
  const taskAnswers: Record<string, ClassifierAnswer> = {
    taskFamily: choice({ coding: 1 }), risk: choice({ low: 1 }), phase: choice({ execution: 1 }), outputLength: choice({ normal: 1 }),
    verifiable: { type: "bool", probability: 1 }, boundedExecution: { type: "bool", probability: 0 },
  };
  return { api: jev.api, provider: jev.provider, model: jev.id, answers: { ...taskAnswers, ...answers }, stopReason: "stop", timestamp: 0 };
}
type RouteRequest = Parameters<typeof route>[0];
function request(overrides: Partial<RouteRequest> = {}): RouteRequest {
  return {
    model: virtual,
    thinkingLevel: "off",
    reason: "user",
    messages: [{ role: "user", content: "Fix the parser and add regression tests.", timestamp: 0 }],
    ...overrides,
  };
}
function fixture(
  answers: Record<string, ClassifierAnswer> = {
    m0: choice({ low: 0.98, insufficient: 0.02 }),
    m1: choice({ medium: 0.98, insufficient: 0.02 }),
    strongest: choice({ m1: 0.9, m0: 0.1 }),
  },
) {
  const calls: { input: ClassifierContext; options?: ModelsClassifierOptions }[] = [];
  const state = {
    models: [fast, strong, virtual],
    scoped: [{ model: fast }, { model: strong }, { model: virtual }] as ScopedModel[],
    classifiers: [jev],
    response: result(answers),
    onClassify: () => {},
  };
  const ctx = {
    get scopedModels() {
      return state.scoped;
    },
    modelRegistry: {
      getAvailable: () => state.models,
      find: (provider: string, id: string) => state.models.find((model) => model.provider === provider && model.id === id),
      getAvailableOfType: async () => state.classifiers,
      classify: async (_model: unknown, input: ClassifierContext, options?: ModelsClassifierOptions) => {
        calls.push({ input, options });
        state.onClassify();
        return state.response;
      },
    },
  } as unknown as ExtensionContext;
  return { ctx, calls, state };
}

function statusFixture(ctx: ExtensionContext, mode: ExtensionContext["mode"] = "tui", selections: (string | undefined)[] = []) {
  let definition: ExtensionVirtualModel<NonNullable<RouteRequest["state"]>> | undefined;
  const handlers = new Map<string, ((event: { type: string; [key: string]: unknown }, ctx: ExtensionContext) => unknown)[]>();
  const commands = new Map<string, (args: string, ctx: ExtensionContext) => Promise<void>>();
  const statuses: (string | undefined)[] = [];
  const notifications: { message: string; type?: string }[] = [];
  const selectCalls: { title: string; options: string[] }[] = [];
  const branch: SessionEntry[] = [];
  Object.assign(ctx, {
    mode,
    hasUI: mode === "tui" || mode === "rpc",
    model: virtual,
    cwd: testAgentDir,
    sessionManager: { getBranch: () => branch },
    ui: {
      setStatus: (key: string, text: string | undefined) => {
        assert.equal(key, "jev-router");
        statuses.push(text);
      },
      notify: (message: string, type?: string) => notifications.push({ message, type }),
      select: async (title: string, options: string[]) => {
        selectCalls.push({ title, options });
        return selections.shift();
      },
    },
  });
  extension({
    on: (event: string, handler: (event: { type: string; [key: string]: unknown }, ctx: ExtensionContext) => unknown) => handlers.set(event, [...(handlers.get(event) ?? []), handler]),
    exec: (command: string, args: string[], options: { timeout?: number; cwd?: string }) => execCommand(command, args, options.cwd ?? testAgentDir, options),
    registerCommand: (name: string, options: { handler: (args: string, ctx: ExtensionContext) => Promise<void> }) => commands.set(name, options.handler),
    registerVirtualModel: (value: typeof definition) => {
      definition = value;
    },
  } as unknown as ExtensionAPI);
  return { definition: definition!, statuses, notifications, selectCalls, commands, branch, emit: async (type: string, data = {}) => {
    const results: unknown[] = [];
    for (const handler of handlers.get(type) ?? []) results.push(await handler({ type, ...data }, ctx));
    return results;
  } };
}

function storedRoute(id: string, thinkingLevel: string, modelId = "auto"): SessionEntry {
  return {
    type: "custom",
    id: "route-entry",
    parentId: null,
    timestamp: "2026-10-02T00:00:00Z",
    customType: "pi.virtual-model-state",
    data: { provider: "jev", modelId, state: { provider: "test", id, thinkingLevel } },
  };
}

test("/jev reports scoped model caps and defaults to each model maximum", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/fast": "low" } });
  const { ctx } = fixture();
  const { commands, notifications } = statusFixture(ctx, "json");
  await commands.get("jev")!("", ctx);
  assert.equal(notifications.length, 1);
  assert.match(notifications[0].message, /test\/fast: low \(configured cap\)/);
  assert.match(notifications[0].message, /test\/strong: high \(model maximum\)/);
  assert.match(notifications[0].message, /jev-llm-rtr\.json/);
  assert.doesNotMatch(notifications[0].message, /jev\/auto/);
});

test("/jev lets the user change a model cap and persists it", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/fast": "low" } });
  const { ctx } = fixture();
  const { commands, notifications, selectCalls } = statusFixture(ctx, "tui", [
    "test/fast — low (configured cap)",
    "medium (cap at medium)",
  ]);
  await commands.get("jev")!("", ctx);
  assert.equal(selectCalls.length, 2);
  assert.deepEqual(selectCalls[1].options, [
    "off (cap at off)",
    "minimal (cap at minimal)",
    "low (cap at low)",
    "medium (cap at medium)",
    "high (cap at high)",
    "Use model maximum (high)",
  ]);
  assert.equal(notifications.at(-1)?.message, "Updated test/fast: medium (configured cap)");
  const saved = JSON.parse(await readFile(join(testAgentDir, "jev-llm-rtr.json"), "utf8"));
  assert.equal(saved.thinkingLevelCaps["test/fast"], "medium");
});

test("/jev can remove a cap and restore the model maximum", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/fast": "low" } });
  const { ctx } = fixture();
  const { commands, notifications } = statusFixture(ctx, "tui", [
    "test/fast — low (configured cap)",
    "Use model maximum (high)",
  ]);
  await commands.get("jev")!("", ctx);
  const saved = JSON.parse(await readFile(join(testAgentDir, "jev-llm-rtr.json"), "utf8"));
  assert.equal(saved.thinkingLevelCaps["test/fast"], undefined);
  assert.equal(notifications.at(-1)?.message, "Updated test/fast: high (model maximum)");
});

test("rejects invalid thinking caps instead of routing without the limit", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/fast": "insane" } });
  const { ctx } = fixture();
  await assert.rejects(route(request(), ctx), /Invalid thinking-level cap/);
});


test("selects the cheapest sufficient model and Jev's thinking level", async () => {
  const { ctx, calls } = fixture();
  const selected = await route(request(), ctx);
  assert.equal(selected.model.id, "fast");
  assert.equal(selected.thinkingLevel, "low");
  assert.equal(calls.length, 1);
  const candidates = calls[0].input.state.candidates as { id: string }[];
  assert.deepEqual(
    candidates.map((candidate) => candidate.id),
    ["fast", "strong"],
  );
});

test("thinking-level caps restrict classifier choices and route fallback maxima", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/fast": "low", "test/strong": "medium" } });
  const { ctx, calls } = fixture({
    m0: choice({ low: 0.9, insufficient: 0.1 }),
    m1: choice({ medium: 0.95, insufficient: 0.05 }),
    strongest: choice({ m1: 1 }),
  });
  const selected = await route(request(), ctx);
  const candidates = calls[0].input.state.candidates as { id: string; thinkingLevels: string[] }[];
  assert.deepEqual(candidates.map(({ id, thinkingLevels }) => [id, thinkingLevels.at(-1)]), [["fast", "low"], ["strong", "medium"]]);
  assert.equal(selected.model.id, "strong");
  assert.equal(selected.thinkingLevel, "medium");
});

test("fresh continuation routing respects the configured native cap", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/strong": "medium" } });
  const { ctx } = fixture({ m0: choice({ insufficient: 1 }), m1: choice({ medium: 1 }), strongest: choice({ m1: 1 }) });
  const selected = await route(request({ reason: "continuation", previous: { model: strong, thinkingLevel: "high" } }), ctx);
  assert.equal(selected.model, strong);
  assert.equal(selected.thinkingLevel, "medium");
});

test("raises thinking to cover uncertainty between required effort levels", async () => {
  const { ctx } = fixture({
    m0: choice({ low: 0.45, medium: 0.52, high: 0.01, insufficient: 0.02 }),
    m1: choice({ low: 0.98, insufficient: 0.02 }),
    strongest: choice({ m1: 1 }),
  });
  const selected = await route(request(), ctx);
  assert.equal(selected.model.id, "fast");
  assert.equal(selected.thinkingLevel, "medium");
});

test("uses a stronger model when the cheap candidate is inadequate", async () => {
  const { ctx } = fixture({
    m0: choice({ high: 0.2, insufficient: 0.8 }),
    m1: choice({ high: 0.9, insufficient: 0.1 }),
    strongest: choice({ m1: 1 }),
  });
  const selected = await route(request(), ctx);
  assert.equal(selected.model.id, "strong");
  assert.equal(selected.thinkingLevel, "high");
});

test("when none qualifies, uses Jev's strongest candidate at maximum supported effort", async () => {
  const { ctx } = fixture({
    m0: choice({ high: 0.6, insufficient: 0.4 }),
    m1: choice({ high: 0.7, insufficient: 0.3 }),
    strongest: choice({ m1: 0.6, m0: 0.4 }),
  });
  const selected = await route(request(), ctx);
  assert.equal(selected.model.id, "strong");
  assert.equal(selected.thinkingLevel, "high");
});

test("quality fallback is Jev's decision, not the most expensive model", async () => {
  const { ctx } = fixture({
    m0: choice({ high: 0.4, insufficient: 0.6 }),
    m1: choice({ high: 0.3, insufficient: 0.7 }),
    strongest: choice({ m0: 0.9, m1: 0.1 }),
  });
  assert.equal((await route(request(), ctx)).model.id, "fast");
});

test("thinking choices use each model's supported levels, ignoring scoped default effort", async () => {
  const limited = model("limited", 1, {
    thinkingLevelMap: { off: null, minimal: null, low: null, high: null, xhigh: "xhigh" },
  });
  const { ctx, state, calls } = fixture();
  state.models = [limited];
  state.scoped = [{ model: limited, thinkingLevel: "high" }];
  state.response = result({ m0: choice({ medium: 0.5, xhigh: 0.5 }), strongest: choice({ m0: 1 }) });
  const selected = await route(request(), ctx);
  assert.equal(selected.thinkingLevel, "xhigh");
  const question = calls[0].input.questions.m0;
  assert.equal(question.type, "choice");
  assert.deepEqual(Object.keys(question.criteria), ["medium", "xhigh", "insufficient"]);
});

test("a sole non-reasoning candidate does not require a classifier", async () => {
  const ordinary = model("ordinary", 1, { reasoning: false });
  const { ctx, state, calls } = fixture();
  state.models = [ordinary];
  state.scoped = [{ model: ordinary }];
  state.classifiers = [];
  const selected = await route(request(), ctx);
  assert.equal(selected.model, ordinary);
  assert.equal(selected.thinkingLevel, "off");
  assert.equal(calls.length, 0);
});

test("ordinary tool turns independently raise and lower model and effort for current work", async () => {
  const { ctx, state, calls } = fixture();
  const first = await route(request(), ctx);
  state.response = result({ m0: choice({ insufficient: 1 }), m1: choice({ high: 1 }), strongest: choice({ m1: 1 }), phase: choice({ planning: 1 }) });
  const harder = await route(request({ reason: "continuation", previous: first, state: first.state, messages: plannedExecutionMessages() }), ctx);
  state.response = result({ m0: choice({ off: 1 }), m1: choice({ high: 1 }), strongest: choice({ m1: 1 }) });
  const mechanical = await route(request({ reason: "continuation", previous: harder, state: harder.state, messages: plannedExecutionMessages() }), ctx);
  state.response = result({ m0: choice({ low: 1 }), m1: choice({ high: 1 }), strongest: choice({ m1: 1 }), risk: choice({ high: 1 }) });
  const sensitive = await route(request({ reason: "continuation", previous: mechanical, state: mechanical.state, messages: plannedExecutionMessages() }), ctx);
  assert.deepEqual([first, harder, mechanical, sensitive].map(({ model, thinkingLevel }) => [model.id, thinkingLevel]),
    [["fast", "low"], ["strong", "high"], ["fast", "off"], ["strong", "high"]]);
  assert.equal(calls.length, 4);
});

test("non-availability retries reassess rather than pinning the failed model", async () => {
  const { ctx } = fixture({ m0: choice({ medium: 1 }), m1: choice({ high: 1 }), strongest: choice({ m1: 1 }) });
  const failed = { model: strong, thinkingLevel: "high", message: { stopReason: "error", errorMessage: "non-transient provider error" } } as ModelRouteRequest["failed"];
  const selected = await route(request({ reason: "retry", failed, previous: { model: fast, thinkingLevel: "low" } }), ctx);
  assert.equal(selected.model, fast);
  assert.equal(selected.thinkingLevel, "medium");
});

test("direct requests reuse the previous physical pair", async () => {
  const { ctx, calls } = fixture();
  const selected = await route(request({ reason: "direct", previous: { model: fast, thinkingLevel: "low" } }), ctx);
  assert.equal(selected.model, fast);
  assert.equal(selected.thinkingLevel, "low");
  assert.equal(calls.length, 0);
});

test("stored branch state cannot pin a retry without a physical response", async () => {
  const { ctx } = fixture();
  const selected = await route(request({ reason: "retry", state: { provider: "test", id: "strong", thinkingLevel: "high" } }), ctx);
  assert.equal(selected.model, fast);
  assert.equal(selected.thinkingLevel, "low");
});

test("new user prompts always reclassify, even with stored state", async () => {
  const { ctx, calls } = fixture();
  const selected = await route(
    request({
      previous: { model: strong, thinkingLevel: "high" },
      state: { provider: "test", id: "strong", thinkingLevel: "high" },
    }),
    ctx,
  );
  assert.equal(selected.model, fast);
  assert.equal(calls.length, 1);
});

test("removing a model from scope prevents sticking to it", async () => {
  const { ctx, state, calls } = fixture({ m0: choice({ low: 1 }), strongest: choice({ m0: 1 }) });
  state.scoped = [{ model: fast }];
  const selected = await route(request({ reason: "continuation", previous: { model: strong, thinkingLevel: "high" } }), ctx);
  assert.equal(selected.model, fast);
  assert.equal(calls.length, 1);
});

test("never offers unscoped, unavailable, duplicate, or virtual models", async () => {
  const unavailable = model("unavailable");
  const unscoped = model("unscoped");
  const { ctx, state, calls } = fixture();
  state.models.push(unscoped);
  state.scoped.push({ model: unavailable }, { model: fast });
  await route(request(), ctx);
  assert.deepEqual(
    (calls[0].input.state.candidates as { id: string }[]).map((m) => m.id),
    ["fast", "strong"],
  );
});

test("requires explicit scope and at least one available physical model", async () => {
  const { ctx, state, calls } = fixture();
  state.scoped = [];
  await assert.rejects(route(request(), ctx), /explicit model scope/);
  state.scoped = [{ model: virtual }];
  await assert.rejects(route(request(), ctx), /no available physical scoped models/);
  assert.equal(calls.length, 0);
});

test("image input excludes text-only models", async () => {
  const vision = model("vision", 10, { input: ["text", "image"] });
  const { ctx, state, calls } = fixture();
  state.models = [fast, vision];
  state.scoped = [{ model: fast }, { model: vision }];
  state.response = result({ m0: choice({ medium: 1 }), strongest: choice({ m0: 1 }) });
  const selected = await route(
    request({
      messages: [
        {
          role: "user",
          content: [{ type: "image", data: "PRIVATE_BASE64", mimeType: "image/png" }],
          timestamp: 0,
        },
      ],
    }),
    ctx,
  );
  assert.equal(selected.model, vision);
  assert.equal(calls[0].input.state.hasImages, true);
  assert.ok(!JSON.stringify(calls[0].input).includes("PRIVATE_BASE64"));
});

test("catalog estimates honor long-context pricing tiers", async () => {
  const tiered = model("tiered", 1, {
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      tiers: [{ inputTokensAbove: 10, input: 100, output: 100, cacheRead: 0, cacheWrite: 0 }],
    },
  });
  const { ctx, state } = fixture();
  state.models = [tiered, strong];
  state.scoped = [{ model: tiered }, { model: strong }];
  const longRequest = request({ messages: [{ role: "user", content: "x".repeat(1_000), timestamp: 0 }] });
  assert.equal((await route(longRequest, ctx)).model, strong);
});

test("unavailable Jev fails clearly instead of inventing a route", async () => {
  const { ctx, state } = fixture();
  state.classifiers = [];
  await assert.rejects(route(request(), ctx), /No authenticated Jev classifier/);
});

test("uses authenticated Jev from another provider when direct TypeSafe is unavailable", async () => {
  const { ctx, state } = fixture();
  state.classifiers = [{ ...jev, provider: "openrouter", id: "typesafe/jev-1.13" }];
  assert.equal((await route(request(), ctx)).model, fast);
});

test("does not substitute a non-Jev classifier", async () => {
  const { ctx, state } = fixture();
  state.classifiers = [{ ...jev, provider: "other", id: "not-jev" }];
  await assert.rejects(route(request(), ctx), /No authenticated Jev classifier/);
});

test("classifier service errors do not produce an arbitrary fallback", async () => {
  const { ctx, state } = fixture();
  state.response = { ...state.response, stopReason: "error", errorMessage: "service unavailable" };
  await assert.rejects(route(request(), ctx), /Jev routing failed: service unavailable/);
});

test("invalid probability distributions cannot qualify cheap models", async () => {
  const { ctx } = fixture({
    m0: choice({ low: 0.9, high: 0.9 }),
    m1: choice({ medium: 0.95, insufficient: 0.05 }),
    strongest: choice({ m1: 1 }),
  });
  assert.equal((await route(request(), ctx)).model, strong);
});

test("normalized rounding at full probability mass still dispatches an eligible control", async () => {
  const { ctx } = fixture({
    m0: choice({
      off: 0.00020516130820136697,
      minimal: 0.3012675624812012,
      low: 0.10294796415694833,
      medium: 0.25068881989927,
      high: 0.34489049215437906,
    }),
    m1: choice({ medium: 1 }),
    strongest: choice({ m1: 1 }),
  });
  const selected = await route(request(), ctx);
  assert.equal(selected.model, fast);
  assert.equal(selected.thinkingLevel, "high");
});

test("invalid thinking levels cannot qualify a model", async () => {
  const { ctx } = fixture({
    m0: choice({ max: 1 }),
    m1: choice({ medium: 1 }),
    strongest: choice({ m1: 1 }),
  });
  assert.equal((await route(request(), ctx)).model, strong);
});

test("an invalid fallback label cannot select an unscoped model", async () => {
  const { ctx, state } = fixture();
  state.response = result({ strongest: choice({ outside: 1 }) });
  await assert.rejects(route(request(), ctx), /no valid model decision/);
});

test("caller cancellation stops routing before classifier work", async () => {
  const { ctx, calls } = fixture();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(route(request({ signal: controller.signal }), ctx), { name: "AbortError" });
  assert.equal(calls.length, 0);
});

test("caller cancellation during classification cannot fall through to a route", async () => {
  const { ctx, state, calls } = fixture();
  const controller = new AbortController();
  state.onClassify = () => controller.abort();
  await assert.rejects(route(request({ signal: controller.signal }), ctx), { name: "AbortError" });
  assert.equal(calls[0].options?.signal?.aborted, true);
});


test("compaction and other direct routes do not overwrite the main status", async () => {
  const { ctx } = fixture();
  const { definition, statuses } = statusFixture(ctx);
  await definition.route(request({ reason: "direct", previous: { model: strong, thinkingLevel: "high" } }), ctx);
  assert.deepEqual(statuses, []);
});

test("JSON, print, and RPC routes do not write terminal status output", async () => {
  for (const mode of ["json", "print", "rpc"] as const) {
    const { ctx } = fixture();
    const display = statusFixture(ctx, mode);
    const { definition, statuses, emit } = display;
    await emit("session_start");
    await definition.route(request(), ctx);
    emit("model_select", { model: fast });
    emit("session_shutdown");
    assert.deepEqual(statuses, []);
    assert.deepEqual(display.notifications, []);
  }
});


test("context fit excludes a cheap model before Jev can select it", async () => {
  const tiny = model("tiny", 0.1, { contextWindow: 4_096 });
  const { ctx, state } = fixture({ m0: choice({ low: 1 }), strongest: choice({ m0: 1 }) });
  state.models = [tiny, strong];
  state.scoped = [{ model: tiny }, { model: strong }];
  const selected = await route(request({ messages: [{ role: "user", content: "x".repeat(40_000), timestamp: 0 }] }), ctx);
  assert.equal(selected.model, strong);
});

test("continuations cannot retain a model after the transcript outgrows its context", async () => {
  const tiny = model("tiny", 0.1, { contextWindow: 4_096 });
  const { ctx, state } = fixture({ m0: choice({ low: 1 }), strongest: choice({ m0: 1 }) });
  state.models = [tiny, strong];
  state.scoped = [{ model: tiny }, { model: strong }];
  const selected = await route(request({ reason: "continuation", previous: { model: tiny, thinkingLevel: "low" }, messages: [{ role: "user", content: "x".repeat(40_000), timestamp: 0 }] }), ctx);
  assert.equal(selected.model, strong);
});

test("normalized probability rounding cannot inflate a cheap candidate's eligibility", async () => {
  const { ctx } = fixture({ m0: choice({ low: 0.967, insufficient: 0.052 }), m1: choice({ medium: 1 }), strongest: choice({ m1: 1 }) });
  assert.equal((await route(request(), ctx)).model, strong);
});

test("risk protection overrides cheap nominal sufficiency", async () => {
  const { ctx } = fixture();
  const selected = await route(request({ messages: [{ role: "user", content: "Fix the authentication bypass without weakening access controls.", timestamp: 0 }] }), ctx);
  assert.equal(selected.model, strong);
  assert.equal(selected.thinkingLevel, "high");
});

test("effective native caps exclude forced-high models below their dispatched effort", async () => {
  const managed = model("managed", 0.1, { api: "anthropic-messages", compat: { supportsMidConvoEffort: true, forceAdaptiveThinking: true } });
  await writeRouterConfig({ thinkingLevelCaps: { "test/managed": "medium" } });
  const { ctx, state } = fixture({ m0: choice({ low: 1 }), strongest: choice({ m0: 1 }) });
  state.models = [managed, strong];
  state.scoped = [{ model: managed }, { model: strong }];
  assert.equal((await route(request(), ctx)).model, strong);
});

test("availability retries fail over within scope without treating a 429 as weak reasoning", async () => {
  const { ctx, state, calls } = fixture({ m0: choice({ low: 1 }), strongest: choice({ m0: 1 }) });
  const failed = { model: fast, thinkingLevel: "low", message: { stopReason: "error", errorMessage: "HTTP 429 rate limit" } } as ModelRouteRequest["failed"];
  const selected = await route(request({ reason: "retry", failed }), ctx);
  assert.equal(selected.model, strong);
  assert.equal(selected.thinkingLevel, "low");
  assert.equal(selected.state?.escalations, 0);
  assert.deepEqual(selected.state?.excluded, ["test/fast"]);
  const continued = await route(request({ reason: "continuation", previous: { model: fast, thinkingLevel: "low" }, state: selected.state }), ctx);
  assert.equal(continued.model, strong);
  state.response = result({ m0: choice({ low: 1 }), m1: choice({ medium: 1 }), strongest: choice({ m1: 1 }) });
  assert.equal((await route(request({ state: selected.state }), ctx)).model, fast);
  assert.ok(calls.length >= 2);
});

test("failover cannot escape scope when all scoped providers have failed", async () => {
  const { ctx, state } = fixture();
  state.scoped = [{ model: fast }];
  await assert.rejects(route(request({ reason: "retry", failed: { model: fast, message: { errorMessage: "overloaded" } } as ModelRouteRequest["failed"] }), ctx), /no remaining.*scoped candidate/);
});

test("verification evidence raises capability and exhaustion stops automatic recovery", async () => {
  const { ctx } = fixture();
  const selected = await route(request(), ctx);
  const escalated = await route(request({ reason: "continuation", previous: { model: fast, thinkingLevel: "low" }, state: { ...selected.state!, verificationFailed: true } }), ctx);
  assert.equal(escalated.model, strong);
  assert.equal(escalated.thinkingLevel, "high");
  assert.equal(escalated.state?.escalations, 1);
  await assert.rejects(route(request({ reason: "continuation", previous: { model: strong, thinkingLevel: "high" }, state: { ...escalated.state!, escalations: 2, verificationFailed: true } }), ctx), /escalation limit/);
});

test("a configured classifier identity never silently substitutes another Jev version", async () => {
  await writeRouterConfig({ classifier: { provider: "typesafe", id: "jev-1.13" } });
  const { ctx } = fixture();
  await assert.rejects(route(request(), ctx), /matching the configured identity/);
});

function plannedExecutionMessages(): Message[] {
  return [
    request().messages[0],
    { role: "assistant", api: fast.api, provider: fast.provider, model: fast.id, timestamp: 1, stopReason: "toolUse", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, content: [
      { type: "text", text: "Plan:\n1. Change the parser for the specified token boundary.\n2. Run npm test.\nAcceptance criteria: tests pass and invalid tokens are rejected." },
      { type: "toolCall", id: "edit-1", name: "edit", arguments: { path: "parser.ts", oldText: "old", newText: "new" } },
    ] },
    { role: "toolResult", toolCallId: "edit-1", toolName: "edit", isError: false, content: [{ type: "text", text: "Successfully updated parser.ts" }], timestamp: 2 },
  ];
}

test("planning and execution switch using fresh adequacy without calibration or phase locks", async () => {
  const { ctx, state } = fixture({ m0: choice({ insufficient: 1 }), m1: choice({ high: 1 }), strongest: choice({ m1: 1 }), phase: choice({ planning: 1 }) });
  const planning = await route(request(), ctx);
  state.response = result({ m0: choice({ low: 1 }), m1: choice({ high: 1 }), strongest: choice({ m1: 1 }) });
  const execution = await route(request({ reason: "continuation", previous: planning, state: planning.state, messages: plannedExecutionMessages() }), ctx);
  assert.equal(execution.model, fast);
  assert.equal(execution.thinkingLevel, "low");
  state.response = result({ m0: choice({ insufficient: 1 }), m1: choice({ high: 1 }), strongest: choice({ m1: 1 }), phase: choice({ planning: 1 }) });
  const nextPlanning = await route(request({ reason: "continuation", previous: execution, state: execution.state, messages: plannedExecutionMessages() }), ctx);
  assert.equal(nextPlanning.model, strong);
  assert.equal(nextPlanning.thinkingLevel, "high");
});

test("a previous high-risk phase does not pin an independently assessed mechanical turn", async () => {
  const { ctx } = fixture();
  const goal = { role: "user", content: "Investigate authentication correctness, then correct documentation typos.", timestamp: 0 } as const;
  const sensitive = await route(request({ messages: [goal] }), ctx);
  assert.equal(sensitive.model, strong);
  const messages = plannedExecutionMessages();
  messages[0] = goal;
  const mechanical = await route(request({ reason: "continuation", previous: sensitive, state: sensitive.state, messages }), ctx);
  assert.equal(mechanical.model, fast);
  assert.equal(mechanical.thinkingLevel, "low");
});

test("long system boilerplate does not force every otherwise qualified step onto the strongest model", async () => {
  const { ctx } = fixture();
  const selected = await route(request({ messages: [
    { role: "system", content: "Reference information. ".repeat(1_000), timestamp: 0 },
    { role: "user", content: "Correct the spelling in this sentence.", timestamp: 1 },
  ] }), ctx);
  assert.equal(selected.model, fast);
  assert.equal(selected.thinkingLevel, "low");
});

test("configured acceptance commands trigger bounded corrective continuation and record recovery-pair outcomes", async () => {
  await writeRouterConfig({ verification: { command: `node -e \"process.exit(require('fs').existsSync('accepted') ? 0 : 1)\"`, timeoutMs: 5_000, maxAttempts: 2 } });
  const { ctx } = fixture();
  const display = statusFixture(ctx, "json");
  const selected = await display.definition.route(request(), ctx);
  const push = (state: RouteState) => display.branch.push({ ...storedRoute(state.id, state.thinkingLevel), data: { provider: "jev", modelId: "auto", state } } as SessionEntry);
  push(selected.state!);
  const boundary = { outcome: "completed", continue: false, context: { llmMessages: plannedExecutionMessages() } };
  const first = (await display.emit("agent_before_settle", boundary)).find(Boolean) as { continue?: boolean; entries: { type: string; data?: { state: RouteState } }[] };
  assert.equal(first.continue, true);
  assert.equal(first.entries[0].data?.state.verificationFailed, true);
  push(first.entries[0].data!.state);
  const recovery = await display.definition.route(request({ reason: "continuation", previous: { model: fast, thinkingLevel: "low" }, state: first.entries[0].data!.state }), ctx);
  assert.equal(recovery.model, strong);
  push(recovery.state!);
  await writeFile(join(testAgentDir, "accepted"), "accepted");
  try {
    const second = (await display.emit("agent_before_settle", boundary)).find(Boolean) as { continue?: boolean; entries: { data?: { state: RouteState } }[] };
    assert.notEqual(second.continue, true);
    assert.equal(second.entries[0].data?.state.verificationFailed, false);
    assert.equal(second.entries[0].data?.state.verificationAttempts, 2);
    const persisted = await readFile(join(testAgentDir, "jev-llm-rtr-history.json"), "utf8");
    assert.ok(!persisted.includes("existsSync") && !persisted.includes("parser.ts"));
  } finally {
    await rm(join(testAgentDir, "accepted"), { force: true });
  }
});

test("explicit task labels calibrate the routed pair without duplicate samples", async () => {
  await writeRouterConfig({ classifier: { provider: "typesafe", id: "jev-1.13" } });
  const { ctx, state } = fixture();
  state.classifiers = [{ ...jev, id: "jev-1.13" }];
  state.response.model = "jev-1.13";
  const display = statusFixture(ctx, "json");
  const selected = await display.definition.route(request(), ctx);
  display.branch.push({ ...storedRoute(selected.model.id, selected.thinkingLevel!), data: { provider: "jev", modelId: "auto", state: selected.state } } as SessionEntry);
  await display.commands.get("jev")!("outcome pass", ctx);
  await display.commands.get("jev")!("outcome pass", ctx);
  const routed = selected.state!;
  const calibration = calibratedProbability(await loadHistory(), routed.modelKey!, routed.controlKey!, "coding", routed.classifierVersion!, routed.prediction!);
  assert.equal(calibration.samples, 1);
  assert.ok(calibration.probability < routed.prediction!);
});

test("unavailable acceptance checks cannot be mistaken for model quality failures", async () => {
  await writeRouterConfig({ verification: { command: "jev_deliberately_missing_acceptance_binary", timeoutMs: 5_000, maxAttempts: 2 } });
  const { ctx } = fixture();
  const display = statusFixture(ctx, "json");
  const selected = await display.definition.route(request(), ctx);
  display.branch.push({ ...storedRoute(selected.model.id, selected.thinkingLevel!), data: { provider: "jev", modelId: "auto", state: selected.state } } as SessionEntry);
  const results = await display.emit("agent_before_settle", { outcome: "completed", continue: false, context: { llmMessages: plannedExecutionMessages() } });
  const boundary = results.find(Boolean) as { continue?: boolean; entries: { data?: { state: RouteState } }[] };
  assert.notEqual(boundary.continue, true);
  assert.equal(boundary.entries[0].data?.state.verificationFailed, false);
  assert.equal(boundary.entries[0].data?.state.verificationAttempts, 1);
  await assert.rejects(readFile(join(testAgentDir, "jev-llm-rtr-history.json")), { code: "ENOENT" });
});
