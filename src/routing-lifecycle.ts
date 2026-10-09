import type { AssistantMessage } from "@earendil-works/pi-ai";
import { VIRTUAL_MODEL_STATE_ENTRY, type ExtensionAPI, type ExtensionContext, type SessionBoundaryDraft } from "@earendil-works/pi-coding-agent";
import { executionEvidence } from "./execution-evidence.ts";
import { loadRouterConfig, routingPolicy } from "./router-config.ts";
import { recordOutcome, recordUsage } from "./routing-history.ts";
import type { RouteState } from "./routing-types.ts";

export function latestRouteState(ctx: ExtensionContext): RouteState | undefined {
  const branch = ctx.sessionManager.getBranch();
  for (let i = branch.length - 1; i >= 0; i--) {
    const entry = branch[i];
    if (entry.type !== "custom" || entry.customType !== VIRTUAL_MODEL_STATE_ENTRY) continue;
    const data = entry.data as { provider?: string; modelId?: string; state?: RouteState } | undefined;
    if (data?.provider === "jev" && data.modelId === "auto") return data.state;
  }
}

export async function recordFeedback(ctx: ExtensionContext, success: boolean): Promise<void> {
  const config = await loadRouterConfig();
  if (!routingPolicy(config).historyEnabled) throw new Error("Jev outcome history is disabled in the routing policy.");
  const state = latestRouteState(ctx);
  if (!state?.taskId || !state.modelKey || !state.controlKey || !state.assessment || !state.classifierVersion) throw new Error("No classified routed task is available on this session branch.");
  await recordOutcome({
    id: `user:${state.taskId}:${state.modelKey}:${state.controlKey}`, taskId: state.taskId,
    modelKey: state.modelKey, controlKey: state.controlKey, family: state.assessment.family,
    classifierVersion: state.classifierVersion, prediction: state.prediction ?? 0, success, source: "user",
  });
}

export function installRoutingLifecycle(pi: ExtensionAPI): void {
  const starts = new Map<string, number>();
  pi.on("message_start", (event, ctx) => {
    if (ctx.model?.provider !== "jev" || ctx.model.id !== "auto" || event.message.role !== "assistant") return;
    const message = event.message as AssistantMessage;
    starts.set(`${message.provider}/${message.model}:${message.timestamp}`, performance.now());
  });
  pi.on("message_end", async (event, ctx) => {
    if (event.message.role !== "assistant") return;
    const message = event.message as AssistantMessage;
    const id = `${message.provider}/${message.model}:${message.timestamp}`;
    const start = starts.get(id);
    starts.delete(id);
    if (start === undefined || ctx.model?.provider !== "jev" || ctx.model.id !== "auto") return;
    const state = latestRouteState(ctx);
    if (!state?.modelKey || !state.controlKey || !state.assessment || state.provider !== message.provider || state.id !== message.model) return;
    if (message.stopReason === "error" || message.stopReason === "aborted") return;
    try {
      if (!routingPolicy(await loadRouterConfig()).historyEnabled) return;
      await recordUsage({ id: `${state.taskId}:${id}`, modelKey: state.modelKey, controlKey: state.controlKey, family: state.assessment.family, usage: message.usage, durationMs: performance.now() - start, timestamp: message.timestamp });
    } catch (error) {
      ctx.ui.notify(`Jev could not record routing measurements: ${error instanceof Error ? error.message : String(error)}`, "warning");
    }
  });
  pi.on("session_shutdown", () => { starts.clear(); });
  pi.on("session_start", () => { starts.clear(); });
  pi.on("agent_before_settle", async (event, ctx) => {
    if (ctx.model?.provider !== "jev" || ctx.model.id !== "auto" || event.outcome !== "completed" || event.continue) return;
    const config = await loadRouterConfig();
    const verification = config.verification;
    const policy = routingPolicy(config);
    const state = latestRouteState(ctx);
    if (!verification || !state?.taskId || !state.assessment?.verifiable || state.assessment.risk !== "low") return;
    const evidence = executionEvidence(event.context.llmMessages);
    if (!evidence.edited || (state.verificationAttempts ?? 0) >= verification.maxAttempts) return;
    const result = await pi.exec("bash", ["-lc", verification.command], { cwd: ctx.cwd, timeout: verification.timeoutMs });
    const success = result.code === 0 && !result.killed;
    const attempt = (state.verificationAttempts ?? 0) + 1;
    if (result.killed || result.code === 126 || result.code === 127) {
      const nextState: RouteState = { ...state, verificationAttempts: attempt, verificationFailed: false };
      ctx.ui.notify("Jev could not execute the configured acceptance check. The task remains unverified, and no capability failure was recorded.", "error");
      return { entries: [
        { type: "custom", customType: VIRTUAL_MODEL_STATE_ENTRY, data: { provider: "jev", modelId: "auto", state: nextState } },
        { type: "custom_message", customType: "jev-verification", content: `The configured acceptance check was unavailable${result.killed ? " or timed out" : ` (exit ${result.code})`}. Fix the check environment before relying on task verification.\n${`${result.stdout}\n${result.stderr}`.slice(-8_000)}`, display: true },
      ] };
    }
    if (policy.historyEnabled && state.modelKey && state.controlKey && state.classifierVersion) {
      await recordOutcome({ id: `contract:${state.taskId}:${state.modelKey}:${state.controlKey}`, taskId: state.taskId, modelKey: state.modelKey, controlKey: state.controlKey, family: state.assessment.family, classifierVersion: state.classifierVersion, prediction: state.prediction ?? 0, success, source: "contract" });
    }
    const nextState: RouteState = { ...state, verificationAttempts: attempt, verificationFailed: !success };
    const entries: SessionBoundaryDraft[] = [{ type: "custom", customType: VIRTUAL_MODEL_STATE_ENTRY, data: { provider: "jev", modelId: "auto", state: nextState } }];
    if (success) {
      entries.push({ type: "custom_message", customType: "jev-verification", content: "Jev: the configured acceptance command passed. This verifies only the configured contract, not untested requirements.", display: true });
      return { entries };
    }
    const canEscalate = attempt < verification.maxAttempts && (state.escalations ?? 0) < policy.maxEscalations;
    const output = `${result.stdout}\n${result.stderr}`.slice(-8_000);
    entries.push({ type: "custom_message", customType: "jev-verification", content: `Jev acceptance command failed (exit ${result.code}). ${canEscalate ? "Correct the failing acceptance checks before finishing. The next request will reassess capability." : "Automatic recovery is exhausted. The task has not met its configured acceptance contract."}\nCommand: ${verification.command}\n${output}`, display: true, details: { success: false, attempt } });
    if (!canEscalate) ctx.ui.notify("Jev acceptance checks failed. Automatic recovery is exhausted.", "error");
    return { entries, continue: canEscalate };
  });
}
