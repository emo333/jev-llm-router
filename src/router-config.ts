import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { dirname, join } from "node:path";
import { getSupportedThinkingLevels, type Api, type Model, type ModelThinkingLevel } from "@earendil-works/pi-ai";
import { getAgentDir, withFileMutationQueue } from "@earendil-works/pi-coding-agent";

const THINKING_LEVELS: readonly ModelThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

export interface RouterConfig {
  thinkingLevelCaps: Record<string, ModelThinkingLevel>;
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
  const levels = getSupportedThinkingLevels(model);
  const cap = config.thinkingLevelCaps[`${model.provider}/${model.id}`];
  if (cap === undefined) return levels;
  const capIndex = THINKING_LEVELS.indexOf(cap);
  return levels.filter((level) => THINKING_LEVELS.indexOf(level) <= capIndex);
}

export function describeThinkingCap(model: Model<Api>, config: RouterConfig): string {
  const supported = getSupportedThinkingLevels(model);
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
