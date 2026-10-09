import { getSystemMessageText, type Message } from "@earendil-works/pi-ai";
import { estimateContextTokens } from "@earendil-works/pi-ai/utils/estimate";

const CONTENT_BUDGET = 48_000;
const OMISSION = "\n[...source text omitted...]\n";
const REQUIREMENT = /\b(?:must(?:\s+not)?|never|required?|requirements?|constraints?|acceptance(?:\s+criteria)?|only|do\s+not|preserve|ensure|without|avoid|shall|critical|mandatory)\b/gi;

export interface RoutingContext {
  prompt: string;
  system: string;
  recent: { role: string; text: string }[];
  contextTokensEstimate: number;
  hasImages: boolean;
  projectionNote: string;
  omissions: { prompt: boolean; system: boolean; recent: boolean };
  requirements: string[];
  toolFailures: string[];
}

function bounded(text: string, limit: number): string {
  if (text.length <= limit) return text;
  if (limit <= OMISSION.length) return text.slice(0, limit);
  const head = Math.ceil((limit - OMISSION.length) / 2);
  return text.slice(0, head) + OMISSION + text.slice(-(limit - OMISSION.length - head));
}

/** Visible source text only: image bytes, thinking, and provider signatures never leave Pi. */
export function messageText(message: Message): string {
  if (message.role === "system") {
    const tools = message.toolsAdded?.length || message.toolsRemoved?.length
      ? `\n${JSON.stringify({ toolsAdded: message.toolsAdded, toolsRemoved: message.toolsRemoved })}` : "";
    return getSystemMessageText(message) + tools;
  }
  if (typeof message.content === "string") return message.content;
  return message.content.flatMap((block) => {
    if (block.type === "text") return [block.text];
    if (block.type === "image") return ["[image attachment: not inspected by Jev]"];
    if (block.type === "toolCall") return [`${block.name}(${JSON.stringify(block.arguments)})`];
    return [];
  }).join("\n");
}

/** Uses Pi's usage-aware estimator on the original transcript, not its classifier projection. */
export function requestSize(messages: readonly Message[]): { tokens: number; hasImages: boolean } {
  return {
    tokens: estimateContextTokens(messages).tokens,
    hasImages: messages.some((message) => typeof message.content !== "string" && message.content.some((block) => block.type === "image")),
  };
}

interface Excerpt { source: number; offset: number; text: string }

// Find actual constraint-bearing excerpts, including the middle of a single very long line.
// Retain bounded first/last pools per source rather than allocating every match in a large transcript.
function requirementExcerpts(text: string, source: number, role: string): Excerpt[] {
  const criticalFirst: Excerpt[] = [];
  const criticalLast: Excerpt[] = [];
  const ordinaryFirst: Excerpt[] = [];
  const ordinaryLast: Excerpt[] = [];
  let previousEnd = -1;
  REQUIREMENT.lastIndex = 0;
  for (let match = REQUIREMENT.exec(text); match; match = REQUIREMENT.exec(text)) {
    if (match.index < previousEnd) continue;
    const start = Math.max(text.lastIndexOf("\n", match.index) + 1, match.index - 160);
    const newline = text.indexOf("\n", match.index);
    const end = Math.min(newline < 0 ? text.length : newline, match.index + 480);
    previousEnd = end;
    const excerpt = { source, offset: start, text: `[${role} message ${source + 1}, chars ${start}-${end}] ${text.slice(start, end).trim()}` };
    const critical = /\b(?:must|never|required?|acceptance|critical|mandatory|shall|do\s+not)\b/i.test(text.slice(start, end));
    const first = critical ? criticalFirst : ordinaryFirst;
    const last = critical ? criticalLast : ordinaryLast;
    if (first.length < 8) first.push(excerpt);
    else {
      last.push(excerpt);
      if (last.length > 8) last.shift();
    }
  }
  return [...criticalFirst, ...criticalLast, ...ordinaryFirst, ...ordinaryLast];
}

/** Bounded, source-labeled evidence; excerpts are instructions/data, never classifier commands. */
export function routingContext(messages: readonly Message[], size = requestSize(messages)): RoutingContext {
  const texts = messages.map(messageText);
  const lastUser = messages.findLastIndex((message) => message.role === "user");
  const promptSource = texts[lastUser] ?? "";
  const systemSource = messages.flatMap((message, i) => message.role === "system" ? [texts[i]] : []).join("\n\n");
  const prompt = bounded(promptSource, 16_000);
  const system = bounded(systemSource, 8_000);
  const omissions = { prompt: promptSource.length > prompt.length, system: systemSource.length > system.length, recent: false };

  // Round-robin across old and new instruction sources so a verbose latest prompt cannot
  // consume the entire requirements budget and hide an earlier task constraint.
  const sources = messages.flatMap((message, i) => message.role === "user" || message.role === "system"
    ? [requirementExcerpts(message.role === "system" ? getSystemMessageText(message) : texts[i], i, message.role)] : []).filter((excerpts) => excerpts.length);
  const order: Excerpt[][] = [];
  for (let left = 0, right = sources.length - 1; left <= right; left++, right--) {
    order.push(sources[right]);
    if (left !== right) order.push(sources[left]);
  }
  const selected: Excerpt[] = [];
  let requirementBudget = 8_000;
  for (let depth = 0; depth < 32 && requirementBudget > 0; depth++) {
    for (const excerpts of order) {
      const excerpt = excerpts[depth];
      if (!excerpt || excerpt.text.length > requirementBudget) continue;
      selected.push(excerpt);
      requirementBudget -= excerpt.text.length;
    }
  }
  const requirements = selected.sort((a, b) => a.source - b.source || a.offset - b.offset).map((excerpt) => excerpt.text);

  const toolFailures: string[] = [];
  let failureBudget = 3_000;
  let omittedFailures = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role !== "toolResult" || !message.isError) continue;
    if (failureBudget <= 0) { omittedFailures++; continue; }
    const source = `[toolResult:${message.toolName} message ${i + 1}, failure] ${texts[i]}`;
    const excerpt = bounded(source, Math.min(1_000, failureBudget));
    toolFailures.unshift(excerpt);
    failureBudget -= excerpt.length;
    if (excerpt.length < source.length) omittedFailures++;
  }

  // Reserve a fixed allowance for source labels and the omission notice; all transmitted
  // content fields, not just prompt/system/recent, share the same character budget.
  let remaining = CONTENT_BUDGET - 1_000 - prompt.length - system.length
    - requirements.reduce((sum, text) => sum + text.length, 0) - toolFailures.reduce((sum, text) => sum + text.length, 0);
  const recent: { role: string; text: string }[] = [];
  let omittedMessages = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (i === lastUser || message.role === "system" || !texts[i]) continue;
    const role = message.role === "toolResult" ? `toolResult:${message.toolName}` : message.role;
    if (remaining <= role.length) { omissions.recent = true; omittedMessages++; continue; }
    const text = bounded(texts[i], Math.min(8_000, remaining - role.length));
    recent.unshift({ role, text });
    remaining -= role.length + text.length;
    if (text.length < texts[i].length) { omissions.recent = true; omittedMessages++; }
  }
  const projectionNote = `Visible text only; total content budget ${CONTENT_BUDGET} characters. Omitted source text: prompt=${omissions.prompt}, system=${omissions.system}, recent=${omissions.recent} (${omittedMessages} messages wholly or partly omitted); ${omittedFailures} tool failures wholly or partly omitted. Requirements are source excerpts, not a complete task specification. Images and hidden reasoning are not transmitted; request size uses Pi's full-transcript usage estimator.`;
  return { prompt, system, recent, contextTokensEstimate: size.tokens, hasImages: size.hasImages, projectionNote, omissions, requirements, toolFailures };
}
