import assert from "node:assert/strict";
import dns from "node:dns/promises";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, test } from "node:test";
import type { Api, ClassifierContext, ClassifierModel, Model } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { route } from "../src/jev-llm-rtr.ts";
import { localModelName } from "../src/local-model-name.ts";

const originalAgentDir = process.env.PI_CODING_AGENT_DIR;
let testAgentDir: string;
before(async () => {
  testAgentDir = await mkdtemp(join(tmpdir(), "jev-local-test-"));
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

function localModel(overrides: Partial<Model<Api>> = {}): Model<Api> {
  return {
    id: "llama.cpp", name: "alb:8081", provider: "albert:8081", api: "openai-completions",
    baseUrl: "http://127.0.0.1:8081/v1", reasoning: false, input: ["text"],
    contextWindow: 131072, maxTokens: 16384,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    ...overrides,
  };
}

function fixture(models = [localModel()]) {
  const calls: ClassifierContext[] = [];
  const ctx = {
    scopedModels: models.map((model) => ({ model })),
    modelRegistry: {
      getAvailable: () => models,
      getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "test-secret", headers: { "X-Local": "test-header" } }),
      getAvailableOfType: async () => [{ provider: "typesafe", id: "jev-latest" } as ClassifierModel<string>],
      classify: async (_model: unknown, input: ClassifierContext) => {
        calls.push(input);
        return {
          stopReason: "stop",
          answers: Object.fromEntries([
            ...models.map((_, i) => [`m${i}`, { type: "choice", choice: "off", probabilities: { off: 1 } }]),
            ["strongest", { type: "choice", choice: "m0", probabilities: { m0: 1 } }],
          ]),
        };
      },
    },
  } as unknown as ExtensionContext;
  return { ctx, calls, models };
}

function request(overrides: Partial<Parameters<typeof route>[0]> = {}): Parameters<typeof route>[0] {
  return {
    model: localModel({ provider: "jev", id: "auto", api: "pi-virtual" }),
    thinkingLevel: "off", reason: "user",
    messages: [{ role: "user", content: "Fix the parser.", timestamp: 0 }],
    ...overrides,
  };
}

const signal = () => new AbortController().signal;

test("local API names reach Jev's metadata and strongest question without changing dispatch", async (t) => {
  const local = Object.freeze(localModel({ baseUrl: "http://albert.bamf:8081/v1" }));
  const { ctx, calls } = fixture([local]);
  t.mock.method(dns, "lookup", async (hostname: string) => {
    assert.equal(hostname, "albert.bamf");
    return [{ address: "10.0.0.230", family: 4 }];
  });
  const fetch = t.mock.method(globalThis, "fetch", async (url: URL, options: RequestInit) => {
    assert.equal(url.href, "http://albert.bamf:8081/v1/models");
    const headers = new Headers(options.headers);
    assert.equal(headers.get("Authorization"), "Bearer test-secret");
    assert.equal(headers.get("X-Local"), "test-header");
    assert.equal(options.redirect, "error");
    assert.ok(options.signal);
    return Response.json({ data: [{ id: "/models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf" }] });
  });
  const selected = await route(request(), ctx);
  assert.equal(fetch.mock.callCount(), 1);
  assert.equal(selected.model, local);
  assert.equal(selected.state?.id, "llama.cpp");
  assert.equal(local.name, "alb:8081");
  const candidate = (calls[0].state.candidates as { id: string; name: string }[])[0];
  assert.deepEqual({ id: candidate.id, name: candidate.name }, { id: "llama.cpp", name: "Qwen3.8-Flash-Next-UD-IQ4_XS" });
  assert.match(JSON.stringify(calls[0].questions.strongest), /Qwen3\.8-Flash-Next-UD-IQ4_XS/);
  assert.doesNotMatch(JSON.stringify(calls[0]), /test-secret|test-header|alb:8081|\/models\//);
});

test("local discovery matches IDs and aliases instead of choosing the first server model", async (t) => {
  const models = [localModel({ id: "dense" }), localModel({ id: "moe" })];
  const { ctx, calls } = fixture(models);
  t.mock.method(globalThis, "fetch", async () => Response.json({ data: [
    { id: "Qwen3.6-35B-A3B-Q4_K_M", aliases: ["moe"] },
    { id: "dense", name: " Qwen3.6-27B-Q4_K_M " },
  ] }));
  await route(request(), ctx);
  assert.deepEqual((calls[0].state.candidates as { name: string }[]).map((m) => m.name), ["Qwen3.6-27B-Q4_K_M", "Qwen3.6-35B-A3B-Q4_K_M"]);
});

test("local discovery refreshes on new prompts but not sticky follow-ups", async (t) => {
  const { ctx, calls } = fixture();
  let name = "first-model";
  const fetch = t.mock.method(globalThis, "fetch", async () => Response.json({ data: [{ id: name }] }));
  const selected = await route(request(), ctx);
  name = "replacement-model";
  for (const reason of ["continuation", "retry", "direct"] as const) {
    await route(request({ reason, previous: selected }), ctx);
  }
  assert.equal(fetch.mock.callCount(), 1);
  await route(request({ previous: selected }), ctx);
  assert.equal(fetch.mock.callCount(), 2);
  assert.equal((calls[1].state.candidates as { name: string }[])[0].name, name);
});

test("local discovery leaves public endpoints alone, even with zero catalog cost", async (t) => {
  const remote = localModel({ baseUrl: "https://api.example.com/v1" });
  const { ctx } = fixture([remote]);
  t.mock.method(dns, "lookup", async () => [{ address: "203.0.113.7", family: 4 }]);
  const auth = t.mock.method(ctx.modelRegistry, "getApiKeyAndHeaders");
  const fetch = t.mock.method(globalThis, "fetch", async () => { throw new Error("Unexpected HTTP request"); });
  assert.equal(await localModelName(remote, ctx.modelRegistry, signal()), remote.name);
  assert.equal(auth.mock.callCount(), 0);
  assert.equal(fetch.mock.callCount(), 0);
});

test("local discovery handles loopback, private IPv4 and IPv6 endpoints and base paths", async (t) => {
  const { ctx } = fixture();
  const fetch = t.mock.method(globalThis, "fetch", async () => Response.json({ data: [{ id: "actual-model" }] }));
  for (const baseUrl of ["http://127.0.0.2:8080", "http://10.0.0.230:8081/v1/", "http://172.16.0.2/v1", "http://192.168.1.2/v1", "http://100.100.0.2/v1", "http://[::1]:8080/v1", "http://[fd00::1]/prefix/v1/"]) {
    assert.equal(await localModelName(localModel({ baseUrl }), ctx.modelRegistry, signal()), "actual-model", baseUrl);
    const url = fetch.mock.calls.at(-1)!.arguments[0] as unknown as URL;
    assert.ok(url.pathname.endsWith("/v1/models"));
  }
});

test("local discovery respects resolved authentication headers and base URL overrides", async (t) => {
  const { ctx, models } = fixture();
  t.mock.method(ctx.modelRegistry, "getApiKeyAndHeaders", async () => ({
    ok: true, apiKey: "unused-key", baseUrl: "http://127.0.0.1:8082/proxy/v1/",
    headers: { authorization: "Bearer header-key", "X-Removed": null },
  }));
  const fetch = t.mock.method(globalThis, "fetch", async (url: URL, options: RequestInit) => {
    assert.equal(url.href, "http://127.0.0.1:8082/proxy/v1/models");
    const headers = new Headers(options.headers);
    assert.equal(headers.get("Authorization"), "Bearer header-key");
    assert.equal(headers.has("X-Removed"), false);
    return Response.json({ data: [{ id: "actual-model" }] });
  });
  assert.equal(await localModelName(models[0], ctx.modelRegistry, signal()), "actual-model");
  assert.equal(fetch.mock.callCount(), 1);
});

test("local discovery does not send credentials to a public auth URL override", async (t) => {
  const { ctx, models } = fixture();
  t.mock.method(ctx.modelRegistry, "getApiKeyAndHeaders", async () => ({ ok: true, apiKey: "test-secret", baseUrl: "https://203.0.113.7/v1" }));
  const fetch = t.mock.method(globalThis, "fetch");
  assert.equal(await localModelName(models[0], ctx.modelRegistry, signal()), models[0].name);
  assert.equal(fetch.mock.callCount(), 0);
});

test("local discovery falls back for missing, malformed, or ambiguous model lists", async (t) => {
  const { ctx, models } = fixture();
  for (const body of [null, {}, { data: null }, { data: [] }, { data: [null] }, { data: [{ id: " " }] }, { data: [{ name: 7 }] }, { data: [{ id: "a" }, { id: "b" }] }, { data: [{ id: "a", aliases: ["llama.cpp"] }, { id: "b", aliases: ["llama.cpp"] }] }]) {
    const fetch = t.mock.method(globalThis, "fetch", async () => Response.json(body));
    assert.equal(await localModelName(models[0], ctx.modelRegistry, signal()), models[0].name, JSON.stringify(body));
    fetch.mock.restore();
  }
});

test("local discovery failure does not prevent routing with the configured name", async (t) => {
  const { ctx, calls } = fixture();
  for (const response of [new Response("denied", { status: 401 }), new Response("unavailable", { status: 503 }), new Response("not json"), new Response(null, { status: 302, headers: { Location: "https://example.com" } })]) {
    const fetch = t.mock.method(globalThis, "fetch", async () => response);
    await route(request(), ctx);
    assert.equal((calls.at(-1)!.state.candidates as { name: string }[])[0].name, "alb:8081");
    fetch.mock.restore();
  }
});

test("local discovery tolerates DNS, network, and auth failures", async (t) => {
  const { ctx, models } = fixture();
  const dnsMock = t.mock.method(dns, "lookup", async () => { throw new Error("DNS unavailable"); });
  const fetch = t.mock.method(globalThis, "fetch", async () => { throw new Error("Connection refused"); });
  assert.equal(await localModelName(localModel({ baseUrl: "http://albert.bamf:8081/v1" }), ctx.modelRegistry, signal()), "alb:8081");
  assert.equal(fetch.mock.callCount(), 0);
  assert.equal(await localModelName(models[0], ctx.modelRegistry, signal()), "alb:8081");
  assert.equal(fetch.mock.callCount(), 1);
  dnsMock.mock.restore();
  t.mock.method(ctx.modelRegistry, "getApiKeyAndHeaders", async () => ({ ok: false, error: "Missing credentials" }));
  assert.equal(await localModelName(models[0], ctx.modelRegistry, signal()), "alb:8081");
  assert.equal(fetch.mock.callCount(), 1);
});

test("local discovery timeout bounds waits for DNS, authentication, and HTTP", async (t) => {
  for (const stage of ["dns", "auth", "fetch"] as const) {
    await t.test(stage, async (t) => {
      const { ctx } = fixture();
      const timeout = new AbortController();
      t.mock.method(AbortSignal, "timeout", (ms: number) => {
        assert.equal(ms, 2_000);
        return timeout.signal;
      });
      const never = () => new Promise<never>(() => {
        setImmediate(() => timeout.abort(new DOMException("Timed out", "TimeoutError")));
      });
      const fetch = t.mock.method(globalThis, "fetch", stage === "fetch" ? never : async () => Response.json({ data: [{ id: "unexpected" }] }));
      if (stage === "dns") t.mock.method(dns, "lookup", never);
      if (stage === "auth") t.mock.method(ctx.modelRegistry, "getApiKeyAndHeaders", never);
      const model = localModel({ baseUrl: stage === "dns" ? "http://albert.bamf:8081/v1" : "http://127.0.0.1:8081/v1" });
      assert.equal(await localModelName(model, ctx.modelRegistry, signal()), model.name);
      assert.equal(fetch.mock.callCount(), stage === "fetch" ? 1 : 0);
    });
  }
});

test("cancellation during local discovery stops routing before Jev", async (t) => {
  const { ctx, calls } = fixture();
  const controller = new AbortController();
  t.mock.method(globalThis, "fetch", async () => {
    controller.abort();
    return Response.json({ data: [{ id: "actual-model" }] });
  });
  await assert.rejects(route(request({ signal: controller.signal }), ctx), { name: "AbortError" });
  assert.equal(calls.length, 0);
});

test("local discovery never queries models outside the available scope", async (t) => {
  const { ctx, models } = fixture([localModel(), localModel({ id: "unscoped", baseUrl: "http://127.0.0.1:8082/v1" })]);
  const unavailable = localModel({ id: "unavailable", baseUrl: "http://127.0.0.1:8083/v1" });
  Object.assign(ctx, { scopedModels: [{ model: models[0] }, { model: unavailable }] });
  const fetch = t.mock.method(globalThis, "fetch", async (url: URL) => {
    assert.equal(url.port, "8081");
    return Response.json({ data: [{ id: "actual-model" }] });
  });
  await route(request(), ctx);
  assert.equal(fetch.mock.callCount(), 1);
});
