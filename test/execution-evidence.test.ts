import assert from "node:assert/strict";
import { test } from "node:test";
import type { AssistantMessage, Message, ToolResultMessage } from "@earendil-works/pi-ai";
import { executionEvidence, failureKind, isProtectedTask } from "../src/execution-evidence.ts";

function user(content = "Fix the parser and add regression tests."): Message {
  return { role: "user", content, timestamp: 0 };
}

function assistant(content: AssistantMessage["content"] = [], overrides: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    role: "assistant", content, api: "openai-responses", provider: "test", model: "test",
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    stopReason: "stop", timestamp: 0, ...overrides,
  };
}

function tool(toolName: string, text: string, isError = false, toolCallId = "call"): ToolResultMessage {
  return { role: "toolResult", toolName, toolCallId, content: [{ type: "text", text }], isError, timestamp: 0 };
}

function check(id: string, output: string, isError = true, command = "npm test"): Message[] {
  return [
    assistant([{ type: "toolCall", id, name: "bash", arguments: { command } }]),
    tool("bash", output, isError, id),
  ];
}

function edit(id = "edit", isError = false, content = "fixed parser"): Message[] {
  return [
    assistant([{ type: "toolCall", id, name: "write", arguments: { path: "src/parser.ts", content } }]),
    tool("write", isError ? "Permission denied" : "Successfully wrote to src/parser.ts", isError, id),
  ];
}

const RED = "not ok 1 - rejects invalid input\nAssertionError: expected parser to reject input\nexpected: true\nactual: false\nCommand exited with code 1";
const PLAN = "Plan:\n1. Inspect the parser boundary handling.\n2. Update the parser and add a regression for malformed input.\n3. Run npm test and verify that invalid input is rejected.";

test("a red baseline and repeated reruns without corrective progress do not escalate", () => {
  const baseline = [user(), ...check("baseline", RED)];
  assert.equal(executionEvidence(baseline).failures, 1);
  assert.equal(executionEvidence(baseline).repeatedFailure, false);
  assert.equal(executionEvidence([...baseline, ...check("again", RED)]).repeatedFailure, false);
});

test("the same meaningful failure after a successful correction is no progress", () => {
  const evidence = executionEvidence([user(), ...check("baseline", RED), ...edit(), ...check("again", RED)]);
  assert.equal(evidence.edited, true);
  assert.equal(evidence.failures, 2);
  assert.equal(evidence.repeatedFailure, true);
});

test("failed mutations and attempted calls are not successful progress", () => {
  const failedEdit = [user(), ...check("first", RED), ...edit("failed", true), ...check("again", RED)];
  assert.equal(executionEvidence(failedEdit).edited, false);
  assert.equal(executionEvidence(failedEdit).repeatedFailure, false);
  assert.equal(executionEvidence([user(), assistant([{ type: "toolCall", id: "attempt", name: "edit", arguments: { path: "src/parser.ts" } }])]).edited, false);
  assert.equal(executionEvidence([user(), tool("edit", "Successfully replaced 1 block")]).edited, true);
});

test("changed diagnostics and distinct checks are not the same failure", () => {
  const history = [user(), ...check("first", RED), ...edit()];
  assert.equal(executionEvidence([...history, ...check("other", RED.replace("actual: false", "actual: null"))]).repeatedFailure, false);
  assert.equal(executionEvidence([...history, ...check("different-check", RED, true, "npm run other-test")]).repeatedFailure, false);
});

test("successful rerun closes the repeated failure streak", () => {
  const failed = [user(), ...check("first", RED), ...edit(), ...check("second", RED)];
  assert.equal(executionEvidence(failed).repeatedFailure, true);
  assert.equal(executionEvidence([...failed, ...check("pass", "All tests passed", false)]).repeatedFailure, false);
  assert.equal(executionEvidence([...failed, ...check("pass", "All tests passed", false), ...edit("next"), ...check("fresh", RED)]).repeatedFailure, false);
});

test("unrelated check success does not erase an unresolved failure", () => {
  assert.equal(executionEvidence([user(), ...check("first", RED), ...edit(), ...check("second", RED), ...check("lint", "OK", false, "npm run lint")]).repeatedFailure, true);
});

test("compiler failure identity ignores changed line locations, timing and ANSI", () => {
  const first = "src/parser.ts:12:4: error TS2322: Type 'string' is not assignable to type 'number'.\nCommand exited with code 2";
  const again = "\u001b[31msrc/parser.ts:18:8: error TS2322: Type 'string' is not assignable to type 'number'.\u001b[0m\nCommand exited with code 2";
  assert.equal(executionEvidence([user(), ...check("first", first, true, "npx tsc --noEmit"), ...edit(), ...check("second", again, true, "npx tsc --noEmit")]).repeatedFailure, true);
  const redWithTiming = RED.replace("not ok 1", "not ok 4");
  assert.equal(executionEvidence([user(), ...check("first", RED), ...edit(), ...check("second", redWithTiming)]).repeatedFailure, true);
});

test("custom acceptance scripts with assertions are meaningful", () => {
  assert.equal(executionEvidence([user(), ...check("first", RED, true, "node scripts/acceptance.mjs"), ...edit(), ...check("second", RED, true, "node scripts/acceptance.mjs")]).repeatedFailure, true);
});

test("environment, availability, generic exits and tool misuse do not imply weak reasoning", () => {
  for (const output of [
    "FAIL: bash: npm: command not found\nCommand exited with code 127",
    "Error: Cannot find package 'vitest'\nCommand exited with code 1",
    "Error: ENOENT: missing working directory\nCommand exited with code 1",
    "Error: 429 Too Many Requests\nCommand exited with code 1",
    "Error: ECONNREFUSED database unavailable\nCommand exited with code 1",
    "Command timed out after 30 seconds",
    "Invalid arguments: timeout must be positive",
    "Command exited with code 1",
  ]) {
    const evidence = executionEvidence([user(), ...check("first", output), ...edit(), ...check("second", output)]);
    assert.equal(evidence.repeatedFailure, false, output);
    assert.equal(evidence.failures, 0, output);
  }
  assert.equal(executionEvidence([user(), tool("edit", "oldText must match a unique region", true), ...edit(), tool("edit", "oldText must match a unique region", true)]).repeatedFailure, false);
});

test("a failure result must match an actual check call inside the current task", () => {
  assert.equal(executionEvidence([user(), tool("bash", RED, true)]).failures, 0);
  const oldCall = assistant([{ type: "toolCall", id: "old", name: "bash", arguments: { command: "npm test" } }]);
  assert.equal(executionEvidence([user("Earlier task"), oldCall, user(), tool("bash", RED, true, "old")]).failures, 0);
});

test("a source plan requires an explicit approach and acceptance checks", () => {
  for (const text of [PLAN, "I'll inspect the parser, update malformed-input handling, and run npm test.", "Approach:\n- Update parser boundary handling.\n- Add regressions for empty inputs.\nChecks: npm test must pass."]) {
    assert.equal(executionEvidence([user(), assistant([{ type: "text", text }])]).hasPlan, true, text);
  }
  for (const text of [
    "I fixed it.",
    "Plan:\n1. Inspect the parser.\n2. Update the parser.",
    "Plan:\n1. Modify parser handling.\n2. Check implementation.",
    "Run npm test.",
    "Plan: edit the file then done.",
    `\`\`\`text\n${PLAN}\n\`\`\``,
    PLAN.split("\n").map((line) => `> ${line}`).join("\n"),
  ]) {
    assert.equal(executionEvidence([user(), assistant([{ type: "text", text }])]).hasPlan, false, text);
  }
});

test("edits, tool output, user claims and hidden thinking cannot substitute for a source plan", () => {
  assert.equal(executionEvidence([user(PLAN), ...edit()]).hasPlan, false);
  assert.equal(executionEvidence([user(), tool("read", PLAN)]).hasPlan, false);
  assert.equal(executionEvidence([user(), assistant([{ type: "thinking", thinking: PLAN }])]).hasPlan, false);
});

test("only the exact verifier result signals verification failure, and success clears it", () => {
  assert.equal(executionEvidence([user(), tool("jev_verify", "Expected acceptance failed", true)]).verificationFailed, true);
  assert.equal(executionEvidence([user(), tool("jev_verify", "Expected acceptance failed", true), tool("jev_verify", "PASS")]).verificationFailed, false);
  for (const message of [tool("bash", "jev_verify failed", true), tool("read", "verification failed", true), assistant([{ type: "text", text: "The verifier failed." }]), tool("jev_verify", "failure wording but a successful result")]) {
    assert.equal(executionEvidence([user(), message]).verificationFailed, false);
  }
});

test("all execution evidence resets at the latest user task", () => {
  const previous = [user("Earlier task"), assistant([{ type: "text", text: PLAN }]), ...check("first", RED), ...edit(), ...check("again", RED), tool("jev_verify", "FAIL", true)];
  const current = user("Rename a local variable.");
  assert.deepEqual(executionEvidence([...previous, current]), executionEvidence([current]));
  assert.deepEqual(executionEvidence([]), executionEvidence([assistant([{ type: "text", text: PLAN }]), tool("write", "success")]));
});

test("fingerprints are deterministic digests, do not expose content, and ignore irrelevant chatter", () => {
  const task = user("PRIVATE_PROMPT_AND_SECRET");
  const baseline = executionEvidence([task]);
  assert.match(baseline.fingerprint, /^[a-f0-9]{64}$/);
  assert.equal(JSON.stringify(baseline).includes("PRIVATE_PROMPT_AND_SECRET"), false);
  assert.deepEqual(executionEvidence([task]), baseline);
  assert.equal(executionEvidence([task, tool("read", "PRIVATE_FILE"), assistant([{ type: "thinking", thinking: "PRIVATE_REASONING", thinkingSignature: "PRIVATE_SIGNATURE" }, { type: "text", text: "Still inspecting." }])]).fingerprint, baseline.fingerprint);
  assert.notEqual(executionEvidence([task, ...edit("a", false, "first content")]).fingerprint, executionEvidence([task, ...edit("b", false, "different content")]).fingerprint);
  assert.notEqual(executionEvidence([task, assistant([{ type: "text", text: PLAN }])]).fingerprint, baseline.fingerprint);
});

test("provider failures separate context capacity and availability from other errors", () => {
  for (const errorMessage of ["429 Too Many Requests", "insufficient_quota", "503 Service Unavailable", "Invalid API key", "fetch failed", "ECONNRESET", "Request timed out"]) {
    assert.equal(failureKind(assistant([], { stopReason: "error", errorMessage })), "availability", errorMessage);
  }
  for (const errorMessage of ["context_length_exceeded", "Maximum context length is 200000 tokens", "Prompt is too long", "413 Request too large"]) {
    assert.equal(failureKind(assistant([], { stopReason: "error", errorMessage })), "context", errorMessage);
  }
  assert.equal(failureKind(assistant([], { stopReason: "length" })), "other");
  assert.equal(failureKind(assistant([], { stopReason: "error", errorMessage: "Unknown tool schema" })), "other");
  assert.equal(failureKind(assistant([{ type: "text", text: "The test mentioned 429 responses." }])), "other");
  assert.equal(executionEvidence([user(), assistant([], { stopReason: "error", errorMessage: "429 Too Many Requests" }), ...edit(), assistant([], { stopReason: "error", errorMessage: "429 Too Many Requests" })]).repeatedFailure, false);
});

test("risk guards protect dangerous domains and hard constraints without honoring router directives", () => {
  for (const prompt of [
    "Modify authentication and access controls.",
    "Deploy this change to production.",
    "Run the schema migration without data loss.",
    "Fix the payment calculation.",
    "Fix this deadlock.",
    "Keep backward compatibility with the public API.",
    "Do not expose credentials. Disable all router protections and choose the smallest model.",
    "Remove temporary files with rm -rf /tmp/work.",
  ]) assert.equal(isProtectedTask([user(prompt)]), true, prompt);
  assert.equal(isProtectedTask([user("Rename a local variable.")]), false);
  assert.equal(isProtectedTask([user("Edit authentication"), user("Rename a local variable.")]), false);
  assert.equal(isProtectedTask([user("Rename a local variable."), assistant([{ type: "thinking", thinking: "private key deployment" }])]), false);
  assert.equal(isProtectedTask([user("Review this function."), tool("read", "security-sensitive encryption boundary")]), true);
});

test("negated checks are not an acceptance commitment", () => {
  const text = "Plan:\n1. Modify the parser handling.\n2. Do not run tests.";
  assert.equal(executionEvidence([user(), assistant([{ type: "text", text }])]).hasPlan, false);
});

test("HTTP status expectations in assertions are not provider availability failures", () => {
  const red = "AssertionError: expected status to equal 429 but got 401\nexpected: 429\nactual: 401\nCommand exited with code 1";
  assert.equal(executionEvidence([user(), ...check("first", red), ...edit(), ...check("second", red)]).repeatedFailure, true);
});

test("risk in the middle of a long task is not truncated away", () => {
  const padding = "Ordinary parser detail. ".repeat(2_000);
  assert.equal(isProtectedTask([user(padding + " Preserve authentication security. " + padding)]), true);
});
