import { getSupportedThinkingLevels, type Api, type Model, type ModelThinkingLevel, type OpenAICompletionsCompat } from "@earendil-works/pi-ai";
import type { EffectiveControl } from "./routing-types.ts";

const LEVELS: readonly ModelThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
const BUDGETS = [0, 1024, 2048, 8192, 16384, 16384, 16384];
const rank = (level: ModelThinkingLevel): number => LEVELS.indexOf(level);
const nativeRank = (value: string, fallback: number): number => value === "none" || value === "disabled" ? 0 : LEVELS.includes(value as ModelThinkingLevel) ? rank(value as ModelThinkingLevel) : fallback;

function completionCompat(model: Model<Api>): OpenAICompletionsCompat {
  const p = model.provider;
  const u = model.baseUrl;
  const zai = p === "zai" || p === "zai-coding-cn" || u.includes("api.z.ai") || u.includes("open.bigmodel.cn");
  const together = p === "together" || u.includes("api.together.ai") || u.includes("api.together.xyz");
  const ant = p === "ant-ling" || u.includes("api.ant-ling.com");
  const deepseek = p === "deepseek" || u.toLowerCase().includes("deepseek.com");
  const openrouter = p === "openrouter" || u.includes("openrouter.ai");
  const unsupported = zai || together || ant || p === "xai" || u.includes("api.x.ai") || p === "moonshotai" || p === "moonshotai-cn" || u.includes("api.moonshot.") || p === "cloudflare-ai-gateway" || u.includes("gateway.ai.cloudflare.com") || p === "nvidia" || u.includes("integrate.api.nvidia.com");
  return { supportsReasoningEffort: !unsupported, thinkingFormat: deepseek ? "deepseek" : zai ? "zai" : together ? "together" : ant ? "ant-ling" : openrouter ? "openrouter" : "openai", ...(model.compat as OpenAICompletionsCompat | undefined) };
}

function samplingControlOverride(model: Model<Api>): { native: string; effortRank: number; ambiguous: boolean } | undefined {
  // These are the adapters that merge model samplingParams after their own control fields.
  if (!["openai-completions", "openai-responses", "azure-openai-responses"].includes(model.api)) return undefined;
  const params = model.samplingParams;
  if (!params) return undefined;
  const keys = ["reasoning", "reasoning_effort", "thinking", "enable_thinking", "chat_template_kwargs", "chat_template_args", "thinking_token_budget", "thinking_budget", "thinking_budget_tokens"];
  const c = model.api === "openai-completions" ? completionCompat(model) : undefined;
  if (c?.thinkingTokenBudgetField && !keys.includes(c.thinkingTokenBudgetField)) keys.push(c.thinkingTokenBudgetField);
  if (c?.thinkingTokenBudgetField || c?.supportsThinkingTokenBudget) keys.push("max_tokens", "max_completion_tokens");
  const overrides = Object.fromEntries(keys.filter((key) => Object.hasOwn(params, key)).sort().map((key) => [key, params[key]]));
  const present = Object.keys(overrides);
  if (!present.length) return undefined;
  // A single native effort override completely replaces this adapter's only knob.
  const soleEffort = model.api !== "openai-completions"
    ? present.length === 1 && present[0] === "reasoning" && overrides.reasoning && typeof overrides.reasoning === "object"
      ? (overrides.reasoning as Record<string, unknown>).effort : undefined
    : present.length === 1 && present[0] === "reasoning_effort" && c?.thinkingFormat === "openai" && !c.thinkingTokenBudgetField && !c.supportsThinkingTokenBudget ? overrides.reasoning_effort : undefined;
  if (typeof soleEffort === "string" && (soleEffort === "none" || LEVELS.includes(soleEffort as ModelThinkingLevel))) {
    return { native: `override:effort:${soleEffort}`, effortRank: nativeRank(soleEffort, 6), ambiguous: false };
  }
  // Arbitrary template objects, switches, removed fields, or conflicting knobs have
  // no established effort bound. Do not claim they comply with any configured cap.
  return { native: `override:${JSON.stringify(overrides)}`, effortRank: 6, ambiguous: true };
}

/** Resolves the installed Pi adapters' wire controls, rather than treating Pi labels as independent knobs. */
export function effectiveControls(model: Model<Api>, cap?: ModelThinkingLevel): EffectiveControl[] {
  const capRank = cap === undefined ? 6 : rank(cap);
  const override = samplingControlOverride(model);
  if (override?.ambiguous && cap !== undefined) return [];
  if (!model.reasoning && !override) return [{ level: "off", native: "off", key: "off", effortRank: 0, outputReserve: Math.min(1024, model.maxTokens) }];
  const compat = model.compat as OpenAICompletionsCompat & { forceAdaptiveThinking?: boolean; supportsMidConvoEffort?: boolean } | undefined;
  const controls = new Map<string, EffectiveControl>();
  for (const level of getSupportedThinkingLevels(model)) {
    const requestedRank = rank(level);
    if (requestedRank > capRank) continue;
    const mapped = model.thinkingLevelMap?.[level];
    let native = typeof mapped === "string" ? mapped : level;
    let effortRank = nativeRank(native, requestedRank);
    let reserve = BUDGETS[Math.min(effortRank, 6)] + 1024;
    const budget = Math.min(BUDGETS[requestedRank], Math.max(0, model.maxTokens - 1024));
    if (model.api === "anthropic-messages" || model.api === "bedrock-converse-stream") {
      const bedrockNames = `${model.id} ${model.name}`.toLowerCase();
      const bedrockAdaptive = model.api === "bedrock-converse-stream" && /(?:opus-4-[678]|opus-5|sonnet-4-6|sonnet-5|fable-5)/.test(bedrockNames.replace(/[\s_.:]+/g, "-"));
      if (model.api === "bedrock-converse-stream" && !/anthropic[./]claude/.test(model.id.toLowerCase()) && !/claude/.test(model.name.toLowerCase())) {
        native = "provider-default";
        effortRank = rank("high");
        reserve = model.maxTokens;
      } else if (model.api === "anthropic-messages" && compat?.supportsMidConvoEffort) {
        native = "adaptive:high";
        effortRank = rank("high");
        reserve = 17408;
      } else if (level === "off") {
        native = "disabled";
        effortRank = 0;
        reserve = 1024;
      } else if (compat?.forceAdaptiveThinking || bedrockAdaptive) {
        const bedrockXhigh = bedrockAdaptive && level === "xhigh" && /(?:opus-4-[78]|opus-5|sonnet-5|fable-5)/.test(bedrockNames.replace(/[\s_.:]+/g, "-"));
        const effort = bedrockXhigh ? "xhigh" : typeof mapped === "string" ? mapped : level === "minimal" ? "low" : level === "xhigh" || level === "max" ? "high" : level;
        native = `adaptive:${effort}`;
        effortRank = nativeRank(effort, requestedRank);
        reserve = BUDGETS[effortRank] + 1024;
      } else {
        // Anthropic's zero budget falls back to 1024 on the wire.
        const tokens = model.api === "anthropic-messages" ? budget || 1024 : budget;
        native = `budget:${tokens}`;
        effortRank = BUDGETS.findIndex((value) => value >= tokens);
        reserve = tokens + 1024;
      }
    } else if (model.api === "google-generative-ai" || model.api === "google-vertex") {
      const id = model.id.toLowerCase();
      const discrete = /gemini-3(?:\.\d+)?-(?:pro|flash)|gemma-?4/.test(id) || id === "gemini-flash-latest" || id === "gemini-flash-lite-latest";
      if (level === "off") {
        native = "budget:0";
        effortRank = 0;
        reserve = 1024;
      } else if (discrete) {
        effortRank = nativeRank(native.toLowerCase(), requestedRank);
        reserve = BUDGETS[effortRank] + 1024;
        native = `level:${native.toLowerCase()}`;
      } else {
        const resolvedRank = nativeRank(native.toLowerCase(), requestedRank);
        const tokens = id.includes("2.5-pro") ? [0, 128, 2048, 8192, 32768][resolvedRank] : id.includes("2.5-flash-lite") ? [0, 512, 2048, 8192, 24576][resolvedRank] : id.includes("2.5-flash") ? [0, 128, 2048, 8192, 24576][resolvedRank] : -1;
        if (tokens === undefined) continue;
        native = `budget:${tokens}`;
        // Dynamic thinking has no controllable lower effort bound.
        effortRank = tokens === -1 ? rank("high") : resolvedRank;
        reserve = tokens === -1 ? model.maxTokens : tokens + 1024;
      }
    } else if (model.api === "mistral-conversations") {
      if (model.thinkingLevelMap) {
        native = level === "off" ? typeof mapped === "string" ? mapped : "none" : typeof mapped === "string" ? mapped : "high";
        effortRank = nativeRank(native, requestedRank);
      } else {
        native = level === "off" ? "disabled" : "reasoning";
        effortRank = level === "off" ? 0 : rank("high");
      }
      reserve = BUDGETS[effortRank] + 1024;
    } else if (model.api === "openai-completions") {
      const c = completionCompat(model);
      const enabled = level !== "off";
      const fields: Record<string, unknown> = {};
      const format = c.thinkingFormat;
      let templateSwitch = false;
      let templateBudget = false;
      let templateEffort = false;
      if (format === "chat-template" || format === "baseten") {
        for (const [key, value] of Object.entries((format === "baseten" ? c.chatTemplateArgs : c.chatTemplateKwargs) ?? {}).sort(([a], [b]) => a.localeCompare(b))) {
          if (typeof value !== "object" || value === null) fields[key] = value;
          else if (!(value.omitWhenOff && !enabled)) {
            templateSwitch ||= value.$var === "thinking.enabled";
            templateBudget ||= value.$var === "thinking.budget";
            templateEffort ||= value.$var === "thinking.effort";
            const resolved = value.$var === "thinking.enabled" ? enabled : value.$var === "thinking.budget" ? enabled && budget > 0 ? budget : undefined : typeof mapped === "string" ? mapped : enabled ? level : undefined;
            if (resolved !== undefined) fields[key] = resolved;
          }
        }
      } else if (["zai", "qwen", "deepseek", "together", "qwen-chat-template"].includes(format ?? "")) fields.enabled = enabled;
      if (format === "openrouter" || format === "string-thinking") fields.effort = enabled ? native : typeof mapped === "string" ? mapped : "none";
      else if (format === "ant-ling") { if (enabled && typeof mapped === "string") fields.effort = mapped; }
      else if (c.supportsReasoningEffort && format !== "chat-template" && format !== "qwen-chat-template") {
        if (enabled || typeof mapped === "string") fields.effort = native;
      }
      if (enabled && budget > 0 && (c.thinkingTokenBudgetField || c.supportsThinkingTokenBudget)) fields.budget = budget;
      native = JSON.stringify(fields);
      const effectiveBudget = typeof fields.budget === "number" ? fields.budget : enabled && templateBudget && budget > 0 ? budget : undefined;
      if (!enabled && (templateSwitch || fields.enabled === false || fields.effort === "none" || fields.effort === "off")) effortRank = 0;
      else if (typeof fields.effort === "string") effortRank = nativeRank(fields.effort, requestedRank);
      else if (templateEffort && enabled) effortRank = nativeRank(typeof mapped === "string" ? mapped : level, requestedRank);
      else if (effectiveBudget !== undefined) effortRank = BUDGETS.findIndex((value) => value >= effectiveBudget);
      else effortRank = rank("high");
      reserve = effectiveBudget !== undefined ? effectiveBudget + 1024 : BUDGETS[effortRank] + 1024;
    } else {
      native = level === "off" ? typeof mapped === "string" ? mapped : "none" : native;
      effortRank = nativeRank(native, requestedRank);
      if (model.api === "openai-responses" && model.provider === "github-copilot" && level === "off") {
        native = "provider-default";
        effortRank = rank(getSupportedThinkingLevels(model).at(-1) ?? "high");
        reserve = model.maxTokens;
      }
    }
    if (override) {
      native = override.native;
      effortRank = override.effortRank;
      reserve = override.ambiguous ? model.maxTokens : BUDGETS[effortRank] + 1024;
    }
    if (effortRank > capRank) continue;
    const key = `${model.api}:${native}`;
    const existing = controls.get(key);
    if (!existing || requestedRank === effortRank && rank(existing.level) !== effortRank || override?.ambiguous && requestedRank > rank(existing.level)) controls.set(key, { level, native, key, effortRank, outputReserve: Math.min(model.maxTokens, reserve) });
  }
  return [...controls.values()].sort((a, b) => a.effortRank - b.effortRank || rank(a.level) - rank(b.level));
}
