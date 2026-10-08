import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, test } from "node:test";
import type { Api, ClassifierAnswer, ClassifierContext, ClassifierModel, ClassifierResult, Message, Model, ModelsClassifierOptions } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ExtensionVirtualModel, ModelRouteRequest, ScopedModel, SessionEntry } from "@earendil-works/pi-coding-agent";
import extension, { route, routingContext } from "../src/jev-llm-rtr.ts";

const originalAgentDir = process.env.PI_CODING_AGENT_DIR;
let testAgentDir: string;
before(async () => {
  testAgentDir = await mkdtemp(join(tmpdir(), "jev-router-test-"));
  process.env.PI_CODING_AGENT_DIR = testAgentDir;
});
beforeEach(async () => {
  await rm(join(testAgentDir, "jev-llm-rtr.json"), { force: true });
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
  return { api: jev.api, provider: jev.provider, model: jev.id, answers, stopReason: "stop", timestamp: 0 };
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
  answers = {
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
  const handlers = new Map<string, (event: { type: string; [key: string]: unknown }, ctx: ExtensionContext) => void | Promise<void>>();
  const commands = new Map<string, (args: string, ctx: ExtensionContext) => Promise<void>>();
  const statuses: (string | undefined)[] = [];
  const notifications: { message: string; type?: string }[] = [];
  const selectCalls: { title: string; options: string[] }[] = [];
  const branch: SessionEntry[] = [];
  Object.assign(ctx, {
    mode,
    hasUI: mode === "tui" || mode === "rpc",
    model: virtual,
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
    on: (event: string, handler: (event: { type: string; [key: string]: unknown }, ctx: ExtensionContext) => void | Promise<void>) => handlers.set(event, handler),
    registerCommand: (name: string, options: { handler: (args: string, ctx: ExtensionContext) => Promise<void> }) => commands.set(name, options.handler),
    registerVirtualModel: (value: typeof definition) => {
      definition = value;
    },
  } as unknown as ExtensionAPI);
  return { definition: definition!, statuses, notifications, selectCalls, commands, branch, emit: (type: string, data = {}) => handlers.get(type)!({ type, ...data }, ctx) };
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

test("registers a virtual model with automatic physical thinking", () => {
  const { ctx } = fixture();
  const { definition } = statusFixture(ctx);
  assert.equal(definition.provider, "jev");
  assert.equal(definition.id, "auto");
  assert.deepEqual(definition.thinkingLevels, ["off"]);
});

test("selects the cheapest sufficient model and Jev's thinking level", async () => {
  const { ctx, calls } = fixture();
  const selected = await route(request(), ctx);
  assert.equal(selected.model.id, "fast");
  assert.equal(selected.thinkingLevel, "low");
  assert.deepEqual(selected.state, { provider: "test", id: "fast", thinkingLevel: "low" });
  assert.equal(calls.length, 1);
  const candidates = calls[0].input.state.candidates as { id: string }[];
  assert.deepEqual(
    candidates.map((candidate) => candidate.id),
    ["fast", "strong"],
  );
  assert.deepEqual(Object.keys(calls[0].input.questions), ["m0", "m1", "strongest"]);
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

test("caps clamp sticky routes to the configured maximum", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/strong": "medium" } });
  const { ctx, calls } = fixture();
  const selected = await route(request({ reason: "continuation", previous: { model: strong, thinkingLevel: "high" } }), ctx);
  assert.equal(selected.model, strong);
  assert.equal(selected.thinkingLevel, "medium");
  assert.equal(calls.length, 0);
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

test("non-reasoning models are routed with thinking off", async () => {
  const ordinary = model("ordinary", 1, { reasoning: false });
  const { ctx, state, calls } = fixture();
  state.models = [ordinary];
  state.scoped = [{ model: ordinary }];
  state.response = result({ m0: choice({ off: 1 }), strongest: choice({ m0: 1 }) });
  assert.equal((await route(request(), ctx)).thinkingLevel, "off");
  assert.deepEqual(Object.keys(calls[0].input.questions.m0.criteria), ["off", "insufficient"]);
});

test("keeps the pair through tool continuations without another classifier call", async () => {
  const { ctx, calls } = fixture();
  const selected = await route(request(), ctx);
  const continuation = await route(
    request({
      reason: "continuation",
      previous: { model: selected.model, thinkingLevel: selected.thinkingLevel },
    }),
    ctx,
  );
  assert.equal(continuation.model, selected.model);
  assert.equal(continuation.thinkingLevel, "low");
  assert.equal(calls.length, 1);
});

test("retries prefer the failed pair over the previous successful pair", async () => {
  const { ctx, calls } = fixture();
  const failed = { model: strong, thinkingLevel: "high", message: {} } as ModelRouteRequest["failed"];
  const selected = await route(request({ reason: "retry", failed, previous: { model: fast, thinkingLevel: "low" } }), ctx);
  assert.equal(selected.model, strong);
  assert.equal(selected.thinkingLevel, "high");
  assert.equal(calls.length, 0);
});

test("direct requests reuse the previous physical pair", async () => {
  const { ctx, calls } = fixture();
  const selected = await route(request({ reason: "direct", previous: { model: fast, thinkingLevel: "low" } }), ctx);
  assert.equal(selected.model, fast);
  assert.equal(selected.thinkingLevel, "low");
  assert.equal(calls.length, 0);
});

test("stored branch state keeps a retry sticky when no physical response is available", async () => {
  const { ctx, calls } = fixture();
  const selected = await route(request({ reason: "retry", state: { provider: "test", id: "strong", thinkingLevel: "high" } }), ctx);
  assert.equal(selected.model, strong);
  assert.equal(selected.thinkingLevel, "high");
  assert.equal(calls.length, 0);
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
  const { ctx, state, calls } = fixture();
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

test("passes prompt, system instructions, recent text and tool context, but not hidden thinking", () => {
  const messages: Message[] = [
    { role: "system", content: "Follow repository rules", sections: { rules: "Keep edits small" }, timestamp: 0 },
    { role: "user", content: "Earlier task", timestamp: 0 },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "PRIVATE_REASONING", thinkingSignature: "SECRET_SIGNATURE" },
        { type: "toolCall", id: "1", name: "read", arguments: { path: "src/parser.ts" } },
      ],
    } as Message,
    { role: "toolResult", toolCallId: "1", toolName: "read", content: [{ type: "text", text: "function parse() {}" }], isError: false, timestamp: 0 },
    { role: "user", content: "Now fix the edge case", timestamp: 0 },
  ];
  const context = routingContext(messages);
  assert.equal(context.prompt, "Now fix the edge case");
  assert.match(context.system, /Keep edits small/);
  assert.match(JSON.stringify(context.recent), /src\/parser.ts/);
  assert.match(JSON.stringify(context.recent), /function parse/);
  assert.ok(!JSON.stringify(context).includes("PRIVATE_REASONING"));
  assert.ok(!JSON.stringify(context).includes("SECRET_SIGNATURE"));
});

test("bounds text context while preserving both ends of the latest prompt", () => {
  const messages: Message[] = [{ role: "system", content: "s".repeat(100_000), timestamp: 0 }, ...Array.from({ length: 20 }, (_, i): Message => ({ role: "user", content: `${i}:` + "c".repeat(10_000), timestamp: 0 })), { role: "user", content: "START" + "p".repeat(100_000) + "END", timestamp: 0 }];
  const context = routingContext(messages);
  assert.ok(context.prompt.startsWith("START"));
  assert.ok(context.prompt.endsWith("END"));
  assert.ok(context.prompt.length + context.system.length + context.recent.reduce((sum, m) => sum + m.text.length, 0) <= 48_000);
  assert.ok(context.contextTokensEstimate > 75_000);
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

test("shows the chosen model and effort before the routed request returns", async () => {
  const { ctx } = fixture();
  const { definition, statuses } = statusFixture(ctx);
  await definition.route(request(), ctx);
  assert.deepEqual(statuses, ["Jev: choosing model…", "Jev: test/fast · low"]);
});

test("sticky turns show their physical pair without a new choosing status", async () => {
  const { ctx, calls } = fixture();
  const { definition, statuses } = statusFixture(ctx);
  await definition.route(request({ reason: "continuation", previous: { model: strong, thinkingLevel: "high" } }), ctx);
  assert.deepEqual(statuses, ["Jev: test/strong · high"]);
  assert.equal(calls.length, 0);
});

test("shows the clamped effort if a sticky model no longer supports its old level", async () => {
  const { ctx, state } = fixture();
  const limited = model("limited", 1, { thinkingLevelMap: { low: null } });
  state.models = [limited];
  state.scoped = [{ model: limited }];
  const { definition, statuses } = statusFixture(ctx);
  await definition.route(request({ reason: "continuation", previous: { model: limited, thinkingLevel: "low" } }), ctx);
  assert.deepEqual(statuses, ["Jev: test/limited · medium"]);
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

test("new sessions show awaiting prompt, and reload restores the latest branch route", async () => {
  const { ctx } = fixture();
  const { statuses, branch, emit } = statusFixture(ctx);
  await emit("session_start");
  branch.push(storedRoute("fast", "low"), storedRoute("strong", "high"));
  await emit("session_start", { reason: "reload" });
  assert.deepEqual(statuses, ["Jev: awaiting prompt", "Jev: last dispatched test/strong · high"]);
});

test("tree navigation updates the status from the new branch rather than global history", () => {
  const { ctx } = fixture();
  const { statuses, branch, emit } = statusFixture(ctx);
  branch.push(storedRoute("strong", "high"));
  emit("session_tree");
  branch.splice(0, branch.length, storedRoute("fast", "low"));
  emit("session_tree");
  assert.deepEqual(statuses, ["Jev: last dispatched test/strong · high", "Jev: last dispatched test/fast · low"]);
});

test("manual model selection clears the Jev status, and reselecting auto restores it", () => {
  const { ctx } = fixture();
  const { statuses, branch, emit } = statusFixture(ctx);
  branch.push(storedRoute("fast", "low"));
  emit("model_select", { model: fast });
  emit("model_select", { model: virtual });
  emit("session_shutdown");
  assert.deepEqual(statuses, [undefined, "Jev: last dispatched test/fast · low", undefined]);
});

test("invalid or unrelated router state never produces a misleading dispatch status", async () => {
  const { ctx } = fixture();
  const { statuses, branch, emit } = statusFixture(ctx);
  branch.push(storedRoute("strong", "high", "another-router"));
  await emit("session_start");
  branch.push(storedRoute("strong", "unsupported"));
  await emit("session_start");
  assert.deepEqual(statuses, ["Jev: awaiting prompt", "Jev: awaiting prompt"]);
});

test("startup reports current scoped caps without selecting a route or exposing model defaults and config paths", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/fast": "low" } });
  const { ctx, calls } = fixture();
  const display = statusFixture(ctx);
  display.branch.push(storedRoute("strong", "high"));
  await display.emit("session_start", { reason: "resume" });
  const rendered = display.notifications.at(-1)!.message;
  assert.match(rendered, /jev\/auto \(active\)/);
  assert.match(rendered, /test\/fast: low \(configured cap\)/);
  assert.match(rendered, /test\/strong: high \(model maximum\)/);
  assert.match(rendered, /Last dispatched: test\/strong · high/);
  assert.doesNotMatch(rendered, /Selected model:|Thinking:|Config:|jev-llm-rtr\.json/);
  assert.deepEqual(calls, []);
});

test("inactive startup reports only its state, even with stored routes and invalid config", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/fast": "invalid" } });
  const { ctx } = fixture();
  const display = statusFixture(ctx);
  display.branch.push(storedRoute("strong", "high"));
  ctx.model = fast;
  await display.emit("session_start");
  assert.deepEqual(display.notifications, [{ message: "Router: jev/auto (inactive)", type: "info" }]);
});

test("startup reports missing scope and candidates excluded by unsupported caps", async () => {
  const { ctx, state } = fixture();
  const display = statusFixture(ctx);
  state.scoped = [];
  await display.emit("session_start");
  assert.match(display.notifications.at(-1)!.message, /no explicit model scope/);
  const limited = model("limited", 1, { thinkingLevelMap: { off: null, minimal: null, low: null } });
  state.models = [limited];
  state.scoped = [{ model: limited }];
  await writeRouterConfig({ thinkingLevelCaps: { "test/limited": "low" } });
  await display.emit("session_start", { reason: "reload" });
  assert.match(display.notifications.at(-1)!.message, /test\/limited: low .*excluded/);
});

test("startup reports invalid config and reload reads corrected settings", async () => {
  await writeRouterConfig({ thinkingLevelCaps: { "test/fast": "invalid" } });
  const { ctx } = fixture();
  const display = statusFixture(ctx);
  await display.emit("session_start");
  const failed = display.notifications.at(-1)!.message;
  assert.match(failed, /Invalid thinking-level cap for test\/fast/);
  assert.doesNotMatch(failed, /model maximum/);
  await writeRouterConfig({ thinkingLevelCaps: { "test/fast": "medium" } });
  await display.emit("session_start", { reason: "reload" });
  const corrected = display.notifications.at(-1)!.message;
  assert.match(corrected, /test\/fast: medium \(configured cap\)/);
  assert.doesNotMatch(corrected, /settings unavailable|Invalid thinking-level/);
});

test("routing failures replace the pending status instead of leaving a stale pair", async () => {
  const { ctx, state } = fixture();
  state.classifiers = [];
  const { definition, statuses } = statusFixture(ctx);
  await assert.rejects(async () => definition.route(request(), ctx), /No authenticated Jev classifier/);
  assert.deepEqual(statuses, ["Jev: choosing model…", "Jev: routing failed"]);
});

test("routing cancellation is reflected in the status without hiding the abort", async () => {
  const { ctx, state } = fixture();
  const controller = new AbortController();
  state.onClassify = () => controller.abort();
  const { definition, statuses } = statusFixture(ctx);
  await assert.rejects(async () => definition.route(request({ signal: controller.signal }), ctx), { name: "AbortError" });
  assert.deepEqual(statuses, ["Jev: choosing model…", "Jev: routing canceled"]);
});
