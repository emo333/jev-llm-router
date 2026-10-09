import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { dirname, join } from "node:path";
import { type Api, type Model, type ModelThinkingLevel } from "@earendil-works/pi-ai";
import { getAgentDir, withFileMutationQueue } from "@earendil-works/pi-coding-agent";
import { effectiveControls } from "./effective-controls.ts";

const THINKING_LEVELS: readonly ModelThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

export interface RoutingPolicy {
  qualityThreshold: number;
  protectedThreshold: number;
  contextSafetyTokens: number;
  latencyUsdPerSecond: number;
  maxEscalations: number;
  historyEnabled: boolean;
  minimumCalibrationSamples: number;
}

const DEFAULT_POLICY: RoutingPolicy = {
  qualityThreshold: 0.967, protectedThreshold: 0.995, contextSafetyTokens: 1024,
  latencyUsdPerSecond: 0, maxEscalations: 2, historyEnabled: true,
  minimumCalibrationSamples: 30,
};

function objectSetting(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`Invalid Jev router setting "${name}": expected an object.`);
  return value as Record<string, unknown>;
}

export function routingPolicy(config: RouterConfig): RoutingPolicy {
  const raw = config.policy === undefined ? {} : objectSetting(config.policy, "policy");
  const policy = { ...DEFAULT_POLICY };
  for (const key of Object.keys(raw)) {
    if (!Object.hasOwn(DEFAULT_POLICY, key)) throw new Error(`Invalid Jev router policy setting "${key}".`);
    const field = key as keyof RoutingPolicy;
    const value = raw[key];
    if (typeof DEFAULT_POLICY[field] === "boolean") {
      if (typeof value !== "boolean") throw new Error(`Invalid Jev router policy "${key}": expected a boolean.`);
    } else {
      if (typeof value !== "number" || !Number.isFinite(value) || value < 0) throw new Error(`Invalid Jev router policy "${key}": expected a finite nonnegative number.`);
      if ((key === "qualityThreshold" || key === "protectedThreshold") && (value <= 0 || value > 1)) throw new Error(`Invalid Jev router policy "${key}": expected a probability in (0, 1].`);
      if (["contextSafetyTokens", "maxEscalations", "minimumCalibrationSamples"].includes(key) && !Number.isSafeInteger(value)) throw new Error(`Invalid Jev router policy "${key}": expected a nonnegative safe integer.`);
      if (key === "minimumCalibrationSamples" && value === 0) throw new Error(`Invalid Jev router policy "${key}": expected a positive integer.`);
    }
    Object.assign(policy, { [key]: value });
  }
  if (policy.protectedThreshold < policy.qualityThreshold) throw new Error("Invalid Jev router policy: protectedThreshold must be at least qualityThreshold.");
  return policy;
}

function validateFeatures(config: RouterConfig): void {
  routingPolicy(config);
  if (config.classifier !== undefined) {
    const classifier = objectSetting(config.classifier, "classifier");
    for (const key of ["provider", "id"]) if (typeof classifier[key] !== "string" || !(classifier[key] as string).trim()) throw new Error(`Invalid Jev router classifier "${key}": expected a nonempty string.`);
  }
  if (config.verification !== undefined) {
    const verification = objectSetting(config.verification, "verification");
    if (typeof verification.command !== "string" || !verification.command.trim()) throw new Error("Invalid Jev router verification command: expected a nonempty string.");
    for (const key of ["timeoutMs", "maxAttempts"]) if (typeof verification[key] !== "number" || !Number.isSafeInteger(verification[key]) || (verification[key] as number) <= 0) throw new Error(`Invalid Jev router verification "${key}": expected a positive safe integer.`);
  }
}

export interface RouterConfig {
  thinkingLevelCaps: Record<string, ModelThinkingLevel>;
  policy?: Partial<RoutingPolicy>;
  classifier?: { provider: string; id: string };
  verification?: { command: string; timeoutMs: number; maxAttempts: number };
  [key: string]: unknown;
}

export function routerConfigPath(): string {
  return join(getAgentDir(), "jev-llm-rtr.json");
}

export async function loadRouterConfig(path = routerConfigPath()): Promise<RouterConfig> {
  let content: string;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { thinkingLevelCaps: {} };
    throw new Error(`Could not read Jev router config at ${path}: ${error instanceof Error ? error.message : String(error)}`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch (error) {
    throw new Error(`Invalid JSON in Jev router config at ${path}: ${error instanceof Error ? error.message : String(error)}`);
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`Invalid Jev router config at ${path}: expected a JSON object.`);
  }
  const raw = parsed as Record<string, unknown>;
  const caps = raw.thinkingLevelCaps;
  validateFeatures({ ...raw, thinkingLevelCaps: {} } as RouterConfig);
  if (caps === undefined) return { ...raw, thinkingLevelCaps: {} };
  if (!caps || typeof caps !== "object" || Array.isArray(caps)) {
    throw new Error(`Invalid Jev router config at ${path}: "thinkingLevelCaps" must be an object.`);
  }

  const thinkingLevelCaps: Record<string, ModelThinkingLevel> = {};
  for (const [modelKey, level] of Object.entries(caps)) {
    if (typeof level !== "string" || !THINKING_LEVELS.includes(level as ModelThinkingLevel)) {
      throw new Error(`Invalid thinking-level cap for ${modelKey} in ${path}: expected one of ${THINKING_LEVELS.join(", ")}.`);
    }
    thinkingLevelCaps[modelKey] = level as ModelThinkingLevel;
  }
  return { ...raw, thinkingLevelCaps };
}

export async function saveThinkingLevelCap(modelKey: string, level: ModelThinkingLevel | undefined, path = routerConfigPath()): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await withFileMutationQueue(path, async () => {
    const config = await loadRouterConfig(path);
    const thinkingLevelCaps = { ...config.thinkingLevelCaps };
    if (level === undefined) delete thinkingLevelCaps[modelKey];
    else thinkingLevelCaps[modelKey] = level;
    const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporaryPath, `${JSON.stringify({ ...config, thinkingLevelCaps }, null, 2)}\n`, { encoding: "utf8", mode: 0o600, flag: "wx" });
      await rename(temporaryPath, path);
    } catch (error) {
      await unlink(temporaryPath).catch(() => {});
      throw new Error(`Could not update Jev router config at ${path}: ${error instanceof Error ? error.message : String(error)}`);
    }
  });
}

export function thinkingLevelsForModel(model: Model<Api>, config: RouterConfig): ModelThinkingLevel[] {
  const cap = config.thinkingLevelCaps[`${model.provider}/${model.id}`];
  return effectiveControls(model, cap).map((control) => control.level);
}

export function describeThinkingCap(model: Model<Api>, config: RouterConfig): string {
  const supported = effectiveControls(model).map((control) => control.level);
  const cap = config.thinkingLevelCaps[`${model.provider}/${model.id}`];
  if (cap === undefined) return `${supported.at(-1) ?? "off"} (model maximum)`;

  const effective = thinkingLevelsForModel(model, config).at(-1);
  if (effective === undefined) return `${cap} (configured cap; no supported level at or below it, excluded)`;
  if (effective === cap) return `${effective} (configured cap)`;
  return `${effective} (effective cap; configured ${cap}, model maximum ${supported.at(-1) ?? "off"})`;
}

export function clampToCandidateLevels(levels: readonly ModelThinkingLevel[], requested: ModelThinkingLevel): ModelThinkingLevel {
  if (levels.includes(requested)) return requested;
  const requestedIndex = THINKING_LEVELS.indexOf(requested);
  return levels.find((level) => THINKING_LEVELS.indexOf(level) >= requestedIndex) ?? levels.at(-1) ?? "off";
}
