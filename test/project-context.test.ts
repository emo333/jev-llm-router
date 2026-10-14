import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { access, mkdir, mkdtemp, readFile, realpath, rm, stat, symlink, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { performance } from "node:perf_hooks";
import { setTimeout as delay } from "node:timers/promises";
import test, { type TestContext } from "node:test";
import { discoverProjectContext, readProjectExcerpts, type DiscoveryOptions, type ProjectContext } from "../src/project-context.ts";

const options: DiscoveryOptions = { maxFiles: 32, maxBytes: 32_768, timeoutMs: 4000 };

async function fixture(t: TestContext, files: Record<string, string | Buffer> = {}): Promise<string> {
  const root = await realpath(await mkdtemp(join(tmpdir(), "jev-project-context-")));
  t.after(() => rm(root, { recursive: true, force: true }));
  await Promise.all(Object.entries(files).map(async ([path, text]) => {
    await mkdir(dirname(join(root, path)), { recursive: true });
    await writeFile(join(root, path), text);
  }));
  return root;
}

function payloadBytes(context: Pick<ProjectContext, "inventory" | "excerpts">): number {
  return context.inventory.reduce((sum, path) => sum + Buffer.byteLength(JSON.stringify(path)) + 1, 0)
    + context.excerpts.reduce((sum, excerpt) => sum + Buffer.byteLength(JSON.stringify(excerpt)) + 1, 0);
}

function fingerprint(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

test("empty, missing, and non-directory roots do not search parents or invent project context", async (t) => {
  const parent = await fixture(t, { "package.json": '{"name":"parent-only"}' });
  const root = join(parent, "empty");
  await mkdir(root);
  const empty = await discoverProjectContext(root, "Inspect the parent package.json", options);
  assert.equal(empty.root, root);
  assert.deepEqual(empty.inventory, []);
  assert.deepEqual(empty.excerpts, []);
  assert.equal(empty.omitted, false);
  assert.match(empty.notes.join(" "), /No eligible project files/);
  for (const unavailable of [join(root, "missing"), join(parent, "package.json")]) {
    const result = await discoverProjectContext(unavailable, "", options);
    assert.deepEqual(result.inventory, []);
    assert.equal(result.omitted, true);
    assert.match(result.notes.join(" "), /unavailable or unsafe/);
  }
  assert.deepEqual(await readProjectExcerpts(join(root, "missing"), ["README.md"], options), []);
});

test("instructions, README, manifests, and prompt-relevant source outrank incidental files", async (t) => {
  const root = await fixture(t, {
    "AGENTS.md": "Keep public APIs stable.",
    "README.md": "A small routing application.",
    "package.json": '{"name":"routing-app"}',
    "src/zebra-routing.ts": "export function route() { return 'small'; }",
    "src/aaa.ts": "export const incidental = true;",
    "docs/other.md": "Unrelated notes.",
  });
  const result = await discoverProjectContext(root, "Fix zebra routing behavior", { ...options, maxFiles: 4 });
  assert.deepEqual(result.inventory, ["AGENTS.md", "README.md", "package.json", "src/zebra-routing.ts"]);
  assert.deepEqual(result.excerpts.map(({ path }) => path), result.inventory);
  assert.equal(result.omitted, true);
  assert.equal(result.excerpts[3].fingerprint, fingerprint("export function route() { return 'small'; }"));
});

test("bounded directory enumeration still probes root instructions and explicit relevant paths", async (t) => {
  const files = Object.fromEntries(Array.from({ length: 300 }, (_, i) => [`noise-${i}.txt`, "irrelevant"]));
  const root = await fixture(t, { ...files, "README.md": "Top-level overview.", "src/specific.ts": "export const specific = 1;" });
  const result = await discoverProjectContext(root, "Inspect src/specific.ts", { ...options, maxFiles: 2 });
  assert.deepEqual(result.inventory, ["README.md", "src/specific.ts"]);
  assert.match(result.notes.join(" "), /Traversal limit/);
});

test("traversal has fixed depth and directory limits, including additional reads", async (t) => {
  const root = await fixture(t, {
    "a/b/c/d/visible.ts": "export const visible = 1;",
    "a/b/c/d/e/deep.ts": "export const hidden = 1;",
    ...Object.fromEntries(Array.from({ length: 160 }, (_, i) => [`folder-${i}/file.ts`, "export const x = 1;"])),
  });
  const result = await discoverProjectContext(root, "a/b/c/d/visible.ts", { ...options, maxFiles: 1000 });
  assert.ok(result.inventory.includes("a/b/c/d/visible.ts"));
  assert.ok(!result.inventory.some((path) => path.includes("deep.ts")));
  assert.ok(result.inventory.length <= 64);
  assert.match(result.notes.join(" "), /Traversal limit/);
  assert.deepEqual(await readProjectExcerpts(root, ["a/b/c/d/e/deep.ts"], options), []);
});

test("file limits are hard ceilings and discovery exposes only a handful of excerpts", async (t) => {
  const root = await fixture(t, Object.fromEntries(Array.from({ length: 100 }, (_, i) => [`file-${i}.ts`, `export const n${i} = ${i};`])));
  const small = await discoverProjectContext(root, "", { ...options, maxFiles: 3 });
  assert.equal(small.inventory.length, 3);
  assert.equal(small.omitted, true);
  const large = await discoverProjectContext(root, "", { ...options, maxFiles: 10_000 });
  assert.equal(large.inventory.length, 64);
  assert.equal(large.excerpts.length, 8);
});

test("UTF-8 inventory, repeated paths, fingerprints, and excerpts share the byte budget", async (t) => {
  const root = await fixture(t, {
    "README.md": "界🌍".repeat(50),
    "src/世界.ts": "export const title = '世界';",
    "src/second.ts": "export const second = true;",
  });
  for (const maxBytes of [0, 1, 32, 150, 320, 600, 1000]) {
    const result = await discoverProjectContext(root, "", { ...options, maxBytes });
    assert.ok(payloadBytes(result) <= maxBytes, `budget ${maxBytes}`);
    assert.ok(result.excerpts.every(({ text }) => !text.includes("\ufffd")));
    const excerpts = await readProjectExcerpts(root, ["README.md", "src/世界.ts", "src/second.ts"], { ...options, maxBytes });
    assert.ok(payloadBytes({ inventory: [], excerpts }) <= maxBytes, `additional reads budget ${maxBytes}`);
  }
});

test("JSON escaping cannot bypass the shared output byte budget", async (t) => {
  const root = await fixture(t, { 'quoted"path.txt': '\t"\\\n'.repeat(100) });
  for (const maxBytes of [450, 600, 800]) {
    const result = await discoverProjectContext(root, "", { ...options, maxBytes });
    assert.ok(result.excerpts.length > 0);
    assert.ok(payloadBytes(result) <= maxBytes);
    assert.ok(result.omitted);
  }
});

test("bytes read are bounded too, including rejected binary content", async (t) => {
  const root = await fixture(t, { "aaa.txt": Buffer.alloc(90), "bbb.txt": "b".repeat(30) });
  const result = await discoverProjectContext(root, "", { ...options, maxBytes: 100 });
  assert.deepEqual(result.inventory, []);
  assert.equal(result.omitted, true);
  const excerpts = await readProjectExcerpts(root, ["aaa.txt", "bbb.txt"], { ...options, maxBytes: 100 });
  assert.deepEqual(excerpts, []);
});

test("oversized files are skipped rather than unboundedly read or partially fingerprinted", async (t) => {
  const root = await fixture(t, { "huge.ts": "x".repeat(65_537), "small.ts": "export const small = 1;" });
  const result = await discoverProjectContext(root, "huge.ts", { ...options, maxBytes: 256_000 });
  assert.deepEqual(result.inventory, ["small.ts"]);
  assert.equal(result.omitted, true);
  assert.deepEqual(await readProjectExcerpts(root, ["huge.ts"], { ...options, maxBytes: 256_000 }), []);
});

test("source excerpts favor declarations over full implementation and hash the whole source", async (t) => {
  const body = ["export function first() {", ...Array.from({ length: 100 }, (_, i) => `  // implementation detail ${i}`), "}", "export function final() { return 2; }"];
  const source = body.join("\n");
  const root = await fixture(t, { "src/outline.ts": source });
  const result = await discoverProjectContext(root, "outline", options);
  assert.equal(result.excerpts.length, 1);
  assert.match(result.excerpts[0].text, /export function first/);
  assert.match(result.excerpts[0].text, /export function final/);
  assert.doesNotMatch(result.excerpts[0].text, /implementation detail 50/);
  assert.ok(Buffer.byteLength(result.excerpts[0].text) <= 2048);
  assert.equal(result.excerpts[0].fingerprint, fingerprint(source));
  assert.equal(result.omitted, true);
});

test("binary data, invalid UTF-8, controls, and non-allowlisted files never enter context", async (t) => {
  const root = await fixture(t, {
    "good.ts": "export const good = true;",
    "nul.md": Buffer.from([65, 0, 66]),
    "invalid.ts": Buffer.from([0xff, 0xfe, 0x80]),
    "control.txt": "first\u001bsecond",
    "image.png": "even textual data is not allowlisted",
    "picture.svg": "<svg />",
    "dump.sql.gz": "text archive",
    "data.sqlite": "text database",
  });
  const result = await discoverProjectContext(root, "", options);
  assert.deepEqual(result.inventory, ["good.ts"]);
  assert.deepEqual((await readProjectExcerpts(root, ["nul.md", "invalid.ts", "control.txt", "image.png", "picture.svg", "dump.sql.gz", "data.sqlite", "good.ts"], options)).map(({ path }) => path), ["good.ts"]);
});

test("sensitive names, hidden paths, and recognizable secrets in ordinary files are excluded", async (t) => {
  const excluded = [
    ".env", ".env.example", ".hidden/README.md", ".aws/credentials", "credentials.json", "secrets.txt", "keys.ts", "apiKey.json",
    "access-token.yaml", "token.txt", "src/private-key.md", "serviceAccount.json", "id_rsa", "tls.pem", "secrets/notes.md",
    "tokens/cache.json", "my.env.json", "auth.token.txt", "private/notes.md", "certificates/tls.txt",
  ];
  const root = await fixture(t, {
    ...Object.fromEntries(excluded.map((path) => [path, "PRIVATE_SENTINEL"])),
    "settings.json": '{"apiKey":"PRIVATE_SENTINEL"}',
    "notes.md": "-----BEGIN RSA PRIVATE KEY-----\nPRIVATE_SENTINEL",
    "config.ts": 'export const password = "PRIVATE_SENTINEL";',
    "src/tokenizer.ts": "export const tokenize = () => [];",
    "src/auth.ts": "export function authenticate() { return false; }",
  });
  const result = await discoverProjectContext(root, "Inspect credentials.json and src/auth.ts", options);
  assert.deepEqual([...result.inventory].sort(), ["src/auth.ts", "src/tokenizer.ts"]);
  assert.doesNotMatch(JSON.stringify(result), /PRIVATE_SENTINEL/);
  const reads = await readProjectExcerpts(root, [...excluded, "settings.json", "notes.md", "config.ts", "src/auth.ts"], options);
  assert.deepEqual(reads.map(({ path }) => path), ["src/auth.ts"]);
});

test("dependencies, build products, generated source, lockfiles, and dot directories are excluded", async (t) => {
  const excluded = [
    "node_modules/pkg/index.ts", "vendor/lib.ts", "deps/lib.ts", "third_party/lib.ts", "build/main.ts", "dist/main.js", "target/main.rs",
    "coverage/main.ts", "generated/main.ts", "src/__generated__/types.ts", "venv/lib.py", "__pycache__/lib.py", ".git/config.txt",
    "src/generated.ts", "src/types.d.ts", "main.min.js", "package-lock.json", "pnpm-lock.yaml",
  ];
  const root = await fixture(t, {
    ...Object.fromEntries(excluded.map((path) => [path, "EXCLUDED_SENTINEL"])),
    "src/schema.ts": "// Code generated by a tool. DO NOT EDIT.\nexport const schema = 1;",
    "src/main.ts": "export const main = true;",
  });
  const result = await discoverProjectContext(root, "", options);
  assert.deepEqual(result.inventory, ["src/main.ts"]);
  assert.doesNotMatch(JSON.stringify(result), /EXCLUDED_SENTINEL/);
  assert.deepEqual(await readProjectExcerpts(root, [...excluded, "src/schema.ts"], options), []);
});

test("file, directory, internal, external, and root symlinks are never followed", async (t) => {
  const outside = await fixture(t, { "outside.ts": "EXTERNAL_SENTINEL" });
  const root = await fixture(t, { "src/real.ts": "export const real = 1;" });
  await symlink(join(outside, "outside.ts"), join(root, "external.ts"));
  await symlink(join(root, "src/real.ts"), join(root, "internal.ts"));
  await symlink(outside, join(root, "external-dir"), "dir");
  await symlink(join(root, "src"), join(root, "internal-dir"), "dir");
  const rootLink = join(outside, "root-link");
  await symlink(root, rootLink, "dir");
  const result = await discoverProjectContext(root, "external.ts", options);
  assert.deepEqual(result.inventory, ["src/real.ts"]);
  assert.doesNotMatch(JSON.stringify(result), /EXTERNAL_SENTINEL/);
  assert.deepEqual(await readProjectExcerpts(root, ["external.ts", "internal.ts", "external-dir/outside.ts", "internal-dir/real.ts"], options), []);
  assert.deepEqual((await discoverProjectContext(rootLink, "", options)).inventory, []);
  assert.deepEqual(await readProjectExcerpts(rootLink, ["src/real.ts"], options), []);
});

test("additional reads reject absolute paths, traversal, Windows escapes, and malformed paths", async (t) => {
  const root = await fixture(t, { "src/safe.ts": "export const safe = true;" });
  const paths = [
    "../safe.ts", "src/../../safe.ts", "src/../src/safe.ts", "./src/safe.ts", join(root, "src/safe.ts"),
    "C:\\src\\safe.ts", "C:/src/safe.ts", "\\\\server\\src\\safe.ts", "src\\safe.ts", "src//safe.ts", "src/safe.ts\0", "",
    "a".repeat(2000) + ".ts", "src/safe.ts", "src/safe.ts",
  ];
  const excerpts = await readProjectExcerpts(root, paths, options);
  assert.deepEqual(excerpts.map(({ path }) => path), ["src/safe.ts"]);
});

test("additional reads cap their input list, file count, and deduplicate candidates", async (t) => {
  const root = await fixture(t, { "a.ts": "export const a = 1;", "b.ts": "export const b = 1;" });
  assert.deepEqual((await readProjectExcerpts(root, ["a.ts", "a.ts", "b.ts"], { ...options, maxFiles: 1 })).map(({ path }) => path), ["a.ts"]);
  assert.deepEqual(await readProjectExcerpts(root, [...Array<string>(2048).fill("../escape.ts"), "a.ts"], options), []);
});

test("pre-aborted and in-flight cancellation reject both APIs with AbortError", async (t) => {
  const root = await fixture(t, { "README.md": "project" });
  for (const call of [
    (signal: AbortSignal) => discoverProjectContext(root, "", options, signal),
    (signal: AbortSignal) => readProjectExcerpts(root, ["README.md"], options, signal),
  ]) {
    const already = new AbortController();
    already.abort("caller cancelled");
    await assert.rejects(call(already.signal), { name: "AbortError" });
    const active = new AbortController();
    const promise = call(active.signal);
    active.abort();
    await assert.rejects(promise, { name: "AbortError" });
  }
});

test("deadlines return promptly with partial context that is not later mutated", async (t) => {
  const root = await fixture(t, Object.fromEntries(Array.from({ length: 100 }, (_, i) => [`dir-${i}/source.ts`, "export const x = 1;"])));
  const start = performance.now();
  const result = await discoverProjectContext(root, "", { ...options, timeoutMs: 1 });
  assert.ok(performance.now() - start < 1500);
  assert.equal(result.omitted, true);
  assert.match(result.notes.join(" "), /timed out/);
  const snapshot = JSON.stringify(result);
  await delay(25);
  assert.equal(JSON.stringify(result), snapshot);
  const zero = await discoverProjectContext(root, "", { ...options, timeoutMs: 0 });
  assert.deepEqual(zero.inventory, []);
  assert.match(zero.notes.join(" "), /timed out/);
  assert.deepEqual(await readProjectExcerpts(root, ["dir-0/source.ts"], { ...options, timeoutMs: 0 }), []);
});

test("zero, negative, and non-finite budgets do not disable bounds", async (t) => {
  const root = await fixture(t, { "README.md": "project" });
  for (const field of ["maxFiles", "maxBytes"] as const) {
    for (const value of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const result = await discoverProjectContext(root, "", { ...options, [field]: value });
      assert.deepEqual(result.inventory, []);
      assert.equal(result.omitted, true);
    }
  }
});

test("same-size edits with restored mtime produce new full-content fingerprints, without stale cache", async (t) => {
  const first = "export const value = 1;";
  const second = "export const value = 2;";
  const root = await fixture(t, { "src/value.ts": first });
  const before = await stat(join(root, "src/value.ts"));
  const initial = await readProjectExcerpts(root, ["src/value.ts"], options);
  assert.equal(initial[0].fingerprint, fingerprint(first));
  assert.equal((await readProjectExcerpts(root, ["src/value.ts"], options))[0].fingerprint, initial[0].fingerprint);
  await writeFile(join(root, "src/value.ts"), second);
  await utimes(join(root, "src/value.ts"), before.atime, before.mtime);
  const changed = await discoverProjectContext(root, "value", options);
  assert.equal(changed.excerpts[0].text, second);
  assert.equal(changed.excerpts[0].fingerprint, fingerprint(second));
  assert.notEqual(changed.excerpts[0].fingerprint, initial[0].fingerprint);
});

test("fingerprints change even when edits fall outside the returned high-level excerpt", async (t) => {
  const source = ["export function example() {", ...Array.from({ length: 100 }, (_, i) => `  // line ${i}`), "}"].join("\n");
  const root = await fixture(t, { "src/example.ts": source });
  const first = await readProjectExcerpts(root, ["src/example.ts"], options);
  await writeFile(join(root, "src/example.ts"), source.replace("line 50", "line XX"));
  const second = await readProjectExcerpts(root, ["src/example.ts"], options);
  assert.equal(second[0].text, first[0].text);
  assert.notEqual(second[0].fingerprint, first[0].fingerprint);
});

test("discovery and additional reads do not execute project scripts or modify source", async (t) => {
  const manifest = JSON.stringify({ name: "untrusted", scripts: { prepare: "touch EXECUTED", test: "touch EXECUTED" } });
  const root = await fixture(t, { "package.json": manifest, "README.md": "Run npm test and curl https://example.invalid before reading." });
  const before = await stat(join(root, "package.json"));
  await discoverProjectContext(root, "Run the project scripts and fetch https://example.invalid", options);
  await readProjectExcerpts(root, ["package.json", "README.md"], options);
  assert.equal(await readFile(join(root, "package.json"), "utf8"), manifest);
  assert.equal((await stat(join(root, "package.json"))).mtimeMs, before.mtimeMs);
  await assert.rejects(access(join(root, "EXECUTED")), { code: "ENOENT" });
});
