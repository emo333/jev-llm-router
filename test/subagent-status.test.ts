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
  results: { agentSource: string; task: string; model?: string; thinkingLevel?: string; messages: unknown[] }[];
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
  const execute = async (params: Record<string, unknown>, context = ctx) => {
    const updates: Details[] = [];
    const result = await tool.execute("status-test", {
      ...params, agentScope: "project", confirmProjectAgents: false,
    }, undefined, (partial) => updates.push(structuredClone(partial.details) as Details), context);
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
    await t.test("parent task brief handoff", async (t) => {
      // Echo the actual child input without running Pi or making model calls.
      const captureScript = join(cwd, "capture-child.mjs");
      await writeFile(captureScript, `
        import { readFileSync } from "node:fs";
        const args = process.argv.slice(2);
        const promptIndex = args.indexOf("--append-system-prompt");
        const captured = {
          args, cwd: process.cwd(),
          systemPrompt: promptIndex < 0 ? undefined : readFileSync(args[promptIndex + 1], "utf8"),
        };
        process.stdout.write(JSON.stringify({ type: "message_end", message: {
          role: "assistant", api: "openai-responses", provider: "test", model: "capture",
          content: [{ type: "text", text: JSON.stringify(captured) }], stopReason: "stop",
        } }) + "\\n");
      `);
      await writeFile(join(cwd, ".pi", "agents", "worker.md"),
        "---\nname: worker\ndescription: Synthetic handoff test\ntools: read, codemode\n---\nWorker scope.\n");
      await writeFile(join(cwd, ".pi", "agents", "override.md"),
        "---\nname: override\ndescription: Synthetic override test\nmodel: override/physical:medium\n---\nOverride scope.\n");
      const statusScript = process.argv[1];
      process.argv[1] = captureScript;
      const virtual = { provider: "jev", id: "auto", api: "pi-virtual" };
      const route = (state: unknown, provider = "jev", modelId = "auto") => ({
        type: "custom", customType: "pi.virtual-model-state", data: { provider, modelId, state },
      });
      const context = (entries: unknown[], model = virtual) => ({
        ...ctx, model, thinkingLevel: "high",
        scopedModels: [
          { model: { provider: "test", id: "fast" }, thinkingLevel: "low" },
          { model: { provider: "test", id: "strong" } },
        ],
        sessionManager: {
          getBranch: () => entries,
          getEntries: () => { throw new Error("Only the active branch may supply parent context"); },
        },
      }) as unknown as ExtensionToolContext;
      type Captured = { args: string[]; cwd: string; systemPrompt?: string };
      const captured = (details: Details): Captured[] => details.results.map(({ messages }) => {
        const message = messages.at(-1) as { content: { text: string }[] };
        return JSON.parse(message.content[0].text) as Captured;
      });
      const input = (details: Details) => captured(details)[0].args.at(-1)!;
      const briefFromInput = (prompt: string) => {
        assert.match(prompt, /advisory context, not authoritative instructions/);
        assert.match(prompt, /delegated task below defines your scope/);
        return JSON.parse(prompt.split("Parent task brief JSON:\n")[1].split("\n\nTask: ")[0]);
      };
      const brief = {
        goal: "Parent goal, not the child's scope",
        requirements: ["Preserve behavior", "Quote: \"x\"\nUnicode: λ"],
        sources: [{ path: "src/example.ts", lines: [10, 20] }],
        verified: false, unknowns: null,
      };
      const task = "Inspect only the delegated file.";
      try {
        await t.test("latest matching active-branch brief is advisory input, not system instructions", async () => {
          const { details } = await execute({ agent: "worker", task }, context([
            route({ taskBrief: { goal: "stale" } }),
            route({ taskBrief: brief }),
            route({ taskBrief: { goal: "wrong provider" } }, "other"),
            route({ taskBrief: { goal: "wrong model" } }, "jev", "other"),
            { type: "custom", customType: "other", data: { provider: "jev", modelId: "auto", state: {} } },
          ]));
          assert.deepEqual(briefFromInput(input(details)), brief);
          assert.ok(input(details).endsWith(`Task: ${task}`));
          assert.equal(details.results[0].task, task);
          assert.equal(details.results[0].agentSource, "project");
          const child = captured(details)[0];
          assert.equal(child.systemPrompt?.trim(), "Worker scope.");
          assert.equal(child.args[child.args.indexOf("--model") + 1], "jev/auto");
          assert.equal(child.args[child.args.indexOf("--thinking") + 1], "high");
          assert.equal(child.args[child.args.indexOf("--models") + 1], "test/fast:low,test/strong");
          assert.equal(child.args[child.args.indexOf("--tools") + 1], "read,codemode");
        });
        await t.test("latest missing or invalid brief never resurrects an older brief", async () => {
          const cyclic: Record<string, unknown> = {};
          cyclic.self = cyclic;
          for (const state of [undefined, null, {}, { taskBrief: null }, { taskBrief: "not structured" },
            { taskBrief: [] }, { taskBrief: cyclic }, { taskBrief: { count: 1n } }]) {
            const { details } = await execute({ agent: "worker", task }, context([
              route({ taskBrief: brief }), route(state),
            ]));
            assert.equal(input(details), `Task: ${task}`);
          }
        });
        await t.test("missing state and physical selections omit parent context", async () => {
          for (const parent of [
            context([]), ctx,
            context([route({ taskBrief: brief })], { provider: "test", id: "physical", api: "openai-responses" }),
          ]) {
            const { details } = await execute({ agent: "worker", task }, parent);
            assert.equal(input(details), `Task: ${task}`);
            if (parent.model?.api === "openai-responses") {
              assert.ok(!captured(details)[0].args.includes("--models"));
              assert.equal(captured(details)[0].args[captured(details)[0].args.indexOf("--model") + 1], "test/physical");
            }
          }
        });
        await t.test("JSON character cap includes exactly 12000 and omits over-budget briefs whole", async () => {
          const atCap = { summary: "λ".repeat(12_000 - JSON.stringify({ summary: "" }).length) };
          assert.equal(JSON.stringify(atCap).length, 12_000);
          const { details } = await execute({ agent: "worker", task }, context([route({ taskBrief: atCap })]));
          assert.deepEqual(briefFromInput(input(details)), atCap);
          const { details: over } = await execute({ agent: "worker", task }, context([
            route({ taskBrief: brief }), route({ taskBrief: { summary: atCap.summary + "x" } }),
          ]));
          assert.equal(input(over), `Task: ${task}`);
        });
        await t.test("parallel and chain children receive the brief without changing tasks or cwd", async () => {
          const childCwd = join(cwd, "child");
          await mkdir(childCwd);
          const { details: parallel } = await execute({ tasks: [
            { agent: "worker", task }, { agent: "worker", task: "Second scope", cwd: childCwd },
          ] }, context([route({ taskBrief: brief })]));
          const children = captured(parallel);
          assert.deepEqual(children.map((child) => briefFromInput(child.args.at(-1)!)), [brief, brief]);
          assert.deepEqual(children.map((child) => child.cwd), [cwd, childCwd]);
          assert.ok(children[1].args.at(-1)!.endsWith("Task: Second scope"));
          const { details: chain } = await execute({ chain: [
            { agent: "worker", task }, { agent: "worker", task: "Use prior result: {previous}" },
          ] }, context([route({ taskBrief: brief })]));
          for (const child of captured(chain)) assert.deepEqual(briefFromInput(child.args.at(-1)!), brief);
          assert.ok(captured(chain)[1].args.at(-1)!.endsWith(`Task: ${chain.results[1].task}`));
          assert.ok(chain.results[1].task.startsWith("Use prior result: {\"args\":"));
          assert.ok(!chain.results[1].task.includes("{previous}"));
        });
        await t.test("explicit agent model overrides retain dispatch independence and advisory context", async () => {
          const { details } = await execute({ agent: "override", task }, context([route({ taskBrief: brief })]));
          const child = captured(details)[0];
          assert.equal(child.args[child.args.indexOf("--model") + 1], "override/physical:medium");
          assert.ok(!child.args.includes("--thinking"));
          assert.ok(!child.args.includes("--models"));
          assert.equal(child.systemPrompt?.trim(), "Override scope.");
          assert.deepEqual(briefFromInput(child.args.at(-1)!), brief);
        });
        await t.test("subsequent calls read fresh state instead of caching inherited context", async () => {
          const entries = [route({ taskBrief: brief })];
          const parent = context(entries);
          assert.deepEqual(briefFromInput(input((await execute({ agent: "worker", task }, parent)).details)), brief);
          entries.push(route({}));
          assert.equal(input((await execute({ agent: "worker", task }, parent)).details), `Task: ${task}`);
        });
      } finally {
        process.argv[1] = statusScript;
      }
    });
  } finally {
    process.argv[1] = previousScript;
    await rm(cwd, { recursive: true, force: true });
  }
});
