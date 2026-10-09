import { createHash } from "node:crypto";
import type { AssistantMessage, Message } from "@earendil-works/pi-ai";
import type { ExecutionEvidence } from "./routing-types.ts";

const MATCH_LIMIT = 32_000;
const ENVIRONMENT_FAILURE = /(?:command not found|not recognized as (?:an internal|a command)|cannot find (?:package|module)|module not found|no module named|missing script|enoent|eacces|permission denied|no space left|enospc|read-only file system|could not resolve host|enotfound|econn(?:refused|reset)|etimedout|network (?:error|unreachable)|connection (?:refused|timed out)|rate.?limit|too many requests|service unavailable|HTTP\s+429\b|command (?:aborted|timed out)|invalid (?:arguments?|parameters?|timeout)|could not (?:find|edit) file|oldtext|must match a unique)/i;
const DIAGNOSTIC = /(?:assertionerror|assertion failed|expected .* (?:to|but)|\berror TS\d+\b|\b(?:SyntaxError|TypeError|ReferenceError)\b|\berror\[E\d+\]|\bFAIL(?:ED)?\b|^\s*not ok\b|^\s*E\s+(?:assert|AssertionError)|\berror:)/i;
const CHECK_COMMAND = /(?:\b(?:test|tests|pytest|vitest|jest|mocha|tsc|eslint|clippy)\b|--test\b|\b(?:check|typecheck|lint|vet)\b)/i;

/** Public text only: hidden thinking and image payloads never become routing evidence. */
function publicText(message: Message): string {
  if (typeof message.content === "string") return message.content;
  return message.content.flatMap((block) => block.type === "text" ? [block.text] : []).join("\n");
}

function matchingText(text: string): string {
  if (text.length <= MATCH_LIMIT) return text;
  return text.slice(0, MATCH_LIMIT / 2) + "\n" + text.slice(-MATCH_LIMIT / 2);
}


/** Transport/capacity failures are not evidence that stronger reasoning is needed. */
export function failureKind(message: AssistantMessage): "availability" | "context" | "other" {
  const error = message.errorMessage ?? "";
  if (/(?:context[_ -]?(?:window|length)(?:[_ -]exceeded)?|maximum context|context limit|prompt (?:is )?too long|input (?:is )?too (?:long|large)|too many (?:input )?tokens|token limit exceeded|request too large|\b413\b)/i.test(error)) {
    return "context";
  }
  if (/(?:\b(?:401|403|408|429|500|502|503|504)\b|rate.?limit|too many requests|quota|overload|capacity|service unavailable|temporarily unavailable|server error|internal server error|authentication|unauthori[sz]ed|forbidden|invalid (?:api[ _-]?key|credentials)|billing|insufficient[_ ](?:quota|credits)|econn(?:refused|reset)|enotfound|etimedout|fetch failed|network (?:error|unreachable)|connection (?:error|refused|reset|timed out)|timed? ?out|timeout)/i.test(error)) {
    return "availability";
  }
  return "other";
}

function sourcePlan(text: string): boolean {
  // Quoted material/code is not an assistant-authored commitment to an approach.
  const prose = matchingText(text).replace(/```[^]*?```/g, "").split("\n").filter((line) => !/^\s*>/.test(line)).join("\n");
  if (!/(?:\b(?:plan|approach|steps)\b|\b(?:I(?:'ll| will)|we(?:'ll| will))\b)/i.test(prose)) return false;
  const action = /\b(?:inspect|read|trace|identify|edit|change|update|modify|fix|implement|add|remove|replace|migrate|refactor|compare|derive|calculate|analy[sz]e|validate|verify|run|test|check|assert|confirm)\b/gi;
  const steps = prose.split("\n").filter((line) => /^\s*(?:[-*+]\s+|\d+[.)]\s+)/.test(line) && line.search(action) >= 0);
  const proseSteps = /\b(?:I(?:'ll| will)|we(?:'ll| will)|plan is to|approach is to)\b/i.test(prose)
    && (prose.match(action) ?? []).length >= 2 && /\b(?:and|then|next)\b/i.test(prose);
  // Require a real approach and a concrete check, not "plan: edit then done".
  if ((steps.length < 2 && !proseSteps) || !/\b(?:edit|change|update|modify|fix|implement|add|remove|replace|migrate|refactor|derive|calculate|analy[sz]e|compare)\b/i.test(prose)) return false;
  const checks = prose.split(/\n|[.!?](?:\s|$)/)
    .filter((line) => !/\b(?:do not|don't|will not|won't|skip|avoid)\s+(?:run|execute|perform|tests?|checks?|verification|verify|validate|check)\b/i.test(line)).join("\n");
  return /\b(?:run|rerun|execute)\s+(?:the\s+)?(?:[^\n]*\b(?:tests?|checks?|suite|pytest|vitest|jest|tsc|lint|typecheck|npm|pnpm|cargo|go test|node --test)\b)/i.test(checks)
    || /\b(?:verify|assert|confirm|validate|check)\s+(?:that\s+[^\n]{8,}|[^\n]{5,}\b(?:returns?|rejects?|accepts?|preserves?|passes?|fails?|matches?|remains?|without)\b[^\n]*)/i.test(checks)
    || /\b(?:acceptance(?: criteria)?|success criteria|checks?|passes when|must pass)\s*:[^\n]*\b(?:tests?|passes?|returns?|rejects?|preserves?|npm|pnpm|pytest|tsc|vitest)\b/i.test(checks);
}

function diagnosticSignature(text: string, command: string): string | undefined {
  const output = matchingText(text).replace(/\x1b\[[0-9;]*m/g, "");
  if (ENVIRONMENT_FAILURE.test(output)) return undefined;
  const lines = output.split("\n");
  const diagnostics = lines.filter((line) => DIAGNOSTIC.test(line));
  if (!diagnostics.length) return undefined;
  // A generic shell failure is not a failed acceptance check. Assertions and
  // compiler diagnostics are meaningful even when a custom check script is used.
  if (!CHECK_COMMAND.test(command) && !/(?:assertion|\berror TS\d+|\berror\[E\d+\]|^\s*not ok\b)/im.test(output)) return undefined;
  const details = lines.filter((line) => /^\s*(?:expected|actual|operator|error|code|[-+]\s+[^-+])/i.test(line));
  const normalize = (line: string) => line
    .replace(/\b\d+(?:\.\d+)?\s*(?:ms|s|seconds?|secs?)\b/g, "<duration>")
    .replace(/\b\d{4}-\d\d-\d\dT[\d:.]+Z\b/g, "<timestamp>")
    .replace(/(?<=\S):\d+(?::\d+)?\b/g, ":<line>")
    .replace(/^(\s*not ok)\s+\d+/, "$1")
    .replace(/\s+/g, " ").trim();
  return [...diagnostics.slice(0, 20), ...details.slice(0, 20)].map(normalize).join("\n");
}


/** Meaningful, task-local evidence; the only persisted identifier is a digest. */
export function executionEvidence(messages: readonly Message[]): ExecutionEvidence {
  const start = messages.findLastIndex((message) => message.role === "user");
  const fingerprint = createHash("sha256");
  const result: ExecutionEvidence = {
    fingerprint: "", edited: false, hasPlan: false, repeatedFailure: false, verificationFailed: false, failures: 0,
  };
  if (start < 0) {
    result.fingerprint = fingerprint.digest("hex");
    return result;
  }
  // Calls/results are paired only inside this task, never against stale branch evidence.
  const calls = new Map<string, { name: string; arguments: unknown; command: string }>();
  const failedChecks = new Map<string, { signature: string; generation: number; repeated: boolean }>();
  let generation = 0;
  const event = (kind: string, value: unknown) => {
    fingerprint.update(kind).update("\0").update(JSON.stringify(value) ?? "").update("\0");
  };
  event("task", publicText(messages[start]));
  for (let index = start + 1; index < messages.length; index++) {
    const message = messages[index];
    if (message.role === "assistant") {
      for (const block of message.content) {
        if (block.type !== "toolCall") continue;
        calls.set(block.id, { name: block.name, arguments: block.arguments, command: typeof block.arguments.command === "string" ? block.arguments.command : "" });
      }
      const text = publicText(message);
      if (sourcePlan(text)) {
        result.hasPlan = true;
        event("plan", text);
      }
      if (message.stopReason === "error") event("provider-failure", [failureKind(message), message.errorMessage ?? ""]);
      continue;
    }
    if (message.role !== "toolResult") continue;
    const text = publicText(message);
    const call = calls.get(message.toolCallId);
    const matchingCall = call?.name === message.toolName ? call : undefined;
    if ((message.toolName === "edit" || message.toolName === "write") && !message.isError) {
      result.edited = true;
      generation++;
      event("edit", [message.toolName, matchingCall?.arguments, text, message.details]);
      continue;
    }
    if (message.toolName === "jev_verify") {
      result.verificationFailed = message.isError;
      if (message.isError) result.failures++;
      event("verification", [message.isError, text]);
      continue;
    }
    if (message.toolName !== "bash" || !matchingCall?.command) continue;
    const command = matchingCall.command.trim().replace(/\s+/g, " ");
    if (!message.isError && !/Command exited with code [1-9]\d*\b/i.test(text)) {
      // A successful rerun closes its failure streak, unlike an unrelated read.
      if (failedChecks.delete(command)) event("check-passed", command);
      continue;
    }
    const signature = diagnosticSignature(text, command);
    if (!signature) continue;
    result.failures++;
    const previous = failedChecks.get(command);
    const repeated = previous?.signature === signature && (generation > previous.generation || previous.repeated);
    failedChecks.set(command, { signature, generation, repeated });
    event("check-failed", [command, signature, generation]);
  }
  result.repeatedFailure = [...failedChecks.values()].some((failure) => failure.repeated);
  result.fingerprint = fingerprint.digest("hex");
  return result;
}

/** Conservative guard for safety/security, irreversible work and hard constraints. */
export function isProtectedTask(messages: readonly Message[]): boolean {
  const start = messages.findLastIndex((message) => message.role === "user");
  if (start < 0) return false;
  const protectedRisk = /\b(?:security|vulnerabilit(?:y|ies)|auth(?:entication|orization)?|cryptograph(?:y|ic)|encryption|secrets?|credentials?|private keys?|access controls?|permissions?|privilege|sandbox|exploit|injection|production|deploy(?:ment)?|database migration|schema migration|data loss|irreversible|destructive|delete (?:all|the database)|drop (?:table|database)|financial|payment|medical|safety[- ]critical|concurrency|race condition|deadlock|backward[- ]compatib(?:le|ility)|public API)\b|\brm\s+-[a-z]*r[a-z]*f\b|\b(?:must not|do not|never)\s+(?:weaken|expose|leak|lose|break|delete|remove)\b/i;
  for (let index = start; index < messages.length; index++) {
    // Classify the task itself and discovered public evidence, not router
    // directives or hidden reasoning. No conversation text is ever executed.
    if (messages[index].role !== "system" && protectedRisk.test(publicText(messages[index]))) return true;
  }
  return false;
}
