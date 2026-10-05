import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import {
  initTheme,
  type ExtensionToolContext,
  type LoadExtensionsResult,
  type Theme,
  type ToolDefinition,
} from "@earendil-works/pi-coding-agent";

interface Details {
  mode: "single" | "parallel" | "chain";
  results: { model?: string; thinkingLevel?: string; messages: unknown[] }[];
}

// Set this to the installed subagent extension to exercise the local integration.
const extensionPath = process.env.PI_SUBAGENT_EXTENSION;
test("installed subagent dispatch status", { skip: !extensionPath, timeout: 15_000 }, async (t) => {
  const loaderUrl = new URL("./core/extensions/loader.js", import.meta.resolve("@earendil-works/pi-coding-agent"));
  const { loadExtensions } = await import(loaderUrl.href) as {
    loadExtensions(paths: string[], cwd: string): Promise<LoadExtensionsResult>;
  };
  const loaded = await loadExtensions([extensionPath!], process.cwd());
  assert.deepEqual(loaded.errors, []);
  const tool = loaded.extensions[0].tools.get("subagent")!.definition;
  const cwd = await mkdtemp(join(tmpdir(), "jev-subagent-status-"));
  await mkdir(join(cwd, ".pi", "agents"), { recursive: true });
  await writeFile(join(cwd, ".pi", "agents", "worker.md"), "---\nname: worker\ndescription: Synthetic status test\n---\n");
  const previousScript = process.argv[1];
  process.argv[1] = fileURLToPath(new URL("./fixtures/subagent-json.mjs", import.meta.url));
  initTheme("dark", false);
  const theme = { fg: (_color: string, text: string) => text, bold: (text: string) => text } as unknown as Theme;
  const ctx = { cwd, hasUI: false, model: { provider: "jev", id: "auto" }, thinkingLevel: "off" } as ExtensionToolContext;
  const renderContext = {} as Parameters<NonNullable<ToolDefinition["renderResult"]>>[3];
  const execute = async (params: Record<string, unknown>) => {
    const updates: Details[] = [];
    const result = await tool.execute("status-test", {
      ...params, agentScope: "project", confirmProjectAgents: false,
    }, undefined, (partial) => updates.push(structuredClone(partial.details) as Details), ctx);
    return { result, details: result.details as Details, updates };
  };
  const rendered = (result: Awaited<ReturnType<typeof execute>>["result"], expanded: boolean) =>
    tool.renderResult!(result, { expanded, isPartial: false }, theme, renderContext).render(200).join("\n");
  try {
    await t.test("single shows dispatch before an assistant message completes", async () => {
      const { result, details, updates } = await execute({ agent: "worker", task: "fast" });
      assert.equal(updates[0].results[0].model, "test/fast");
      assert.equal(updates[0].results[0].thinkingLevel, "low");
      assert.deepEqual(updates[0].results[0].messages, []);
      assert.equal(details.results[0].model, "test/fast");
      for (const expanded of [false, true]) assert.match(rendered(result, expanded), /test\/fast · low/);
    });
    await t.test("parallel shows each agent's pair in collapsed and expanded views", async () => {
      const { result, details } = await execute({ tasks: [
        { agent: "worker", task: "fast" }, { agent: "worker", task: "strong" },
      ] });
      assert.deepEqual(details.results.map((r) => [r.model, r.thinkingLevel]), [["test/fast", "low"], ["test/strong", "high"]]);
      for (const expanded of [false, true]) {
        const text = rendered(result, expanded);
        assert.match(text, /test\/fast · low/);
        assert.match(text, /test\/strong · high/);
      }
    });
    await t.test("chain shows each step's pair in collapsed and expanded views", async () => {
      const { result } = await execute({ chain: [
        { agent: "worker", task: "fast" }, { agent: "worker", task: "strong" },
      ] });
      for (const expanded of [false, true]) {
        const text = rendered(result, expanded);
        assert.match(text, /test\/fast · low/);
        assert.match(text, /test\/strong · high/);
      }
    });
    await t.test("later dispatches replace the originally requested virtual model", async () => {
      const { result, details, updates } = await execute({ agent: "worker", task: "fast reroute" });
      assert.equal(updates[0].results[0].model, "test/fast");
      assert.equal(details.results[0].model, "test/strong");
      assert.equal(details.results[0].thinkingLevel, "high");
      assert.match(rendered(result, false), /test\/strong · high/);
    });
    await t.test("physical assistant messages provide a fallback without router state", async () => {
      const { details, updates } = await execute({ agent: "worker", task: "no-route" });
      assert.equal(updates[0].results[0].model, "test/fast");
      assert.equal(updates[0].results[0].thinkingLevel, undefined);
      assert.equal(details.results[0].thinkingLevel, "low");
    });
    await t.test("invalid route state does not supply a false model or effort", async () => {
      const { details, updates } = await execute({ agent: "worker", task: "invalid-state no-route" });
      assert.equal(updates[0].results[0].model, "test/fast");
      assert.equal(updates[0].results[0].thinkingLevel, undefined);
      assert.equal(details.results[0].model, "test/fast");
    });
  } finally {
    process.argv[1] = previousScript;
    await rm(cwd, { recursive: true, force: true });
  }
});
