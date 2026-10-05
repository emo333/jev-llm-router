// Synthetic Pi child process. No model calls, credentials, or network access.
const task = process.argv.at(-1) ?? "";
const usage = {
  input: 10, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 11,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};
const emit = (event) => process.stdout.write(JSON.stringify(event) + "\n");
const dispatch = (model, thinkingLevel, includeRoute = true) => {
  if (includeRoute) emit({
    type: "entry_appended",
    entry: {
      type: "custom", customType: "pi.virtual-model-state",
      data: { provider: "jev", modelId: "auto", state: { provider: "test", id: model, thinkingLevel } },
    },
  });
  emit({ type: "message_start", message: { role: "assistant", api: "openai-responses", provider: "test", model, content: [] } });
  emit({ type: "message_end", message: {
    role: "assistant", api: "openai-responses", provider: "test", model, thinkingLevel,
    content: [{ type: "text", text: "Synthetic task completed." }], usage, stopReason: "stop", timestamp: 0,
  } });
};
if (task.includes("invalid-state")) emit({
  type: "entry_appended",
  entry: { type: "custom", customType: "pi.virtual-model-state", data: { provider: "jev", modelId: "auto", state: { provider: "test", id: "wrong", thinkingLevel: "unsupported" } } },
});
dispatch(task.includes("strong") ? "strong" : "fast", task.includes("strong") ? "high" : "low", !task.includes("no-route"));
if (task.includes("reroute")) dispatch("strong", "high");
