import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, opendir, realpath, type FileHandle } from "node:fs/promises";
import { basename, extname, isAbsolute, join, resolve, win32 } from "node:path";
import { performance } from "node:perf_hooks";

export interface ProjectContext {
  root: string;
  inventory: string[];
  excerpts: { path: string; text: string; fingerprint: string }[];
  omitted: boolean;
  notes: string[];
}

export interface DiscoveryOptions {
  maxFiles: number;
  maxBytes: number;
  timeoutMs: number;
}

// Caller budgets can reduce, never remove, these independent safety ceilings.
const MAX_FILES = 64;
const MAX_BYTES = 256 * 1024;
const MAX_FILE_BYTES = 64 * 1024;
const MAX_EXCERPT_BYTES = 2048;
const MAX_EXCERPTS = 8;
const MAX_DEPTH = 4;
const MAX_ENTRIES = 2048;
const MAX_DIRECTORY_ENTRIES = 256;
const MAX_DIRECTORIES = 128;
const MAX_PATH_BYTES = 1024;
const MAX_TIMEOUT_MS = 5000;
const OMISSION = "\n[...source omitted...]";
const ROOT_HINTS = [
  "AGENTS.md", "CLAUDE.md", "GEMINI.md", "README.md", "README.rst", "README.txt", "README",
  "package.json", "pyproject.toml", "Cargo.toml", "go.mod", "deno.json", "composer.json",
  "requirements.txt", "Gemfile", "Makefile", "CMakeLists.txt",
];
const EXCLUDED_DIRECTORIES = new Set([
  "node_modules", "bower_components", "vendor", "vendors", "deps", "dependencies", "third_party", "third-party", "thirdparty",
  "build", "builds", "dist", "target", "out", "output", "bin", "obj", "coverage", "generated", "__generated__",
  "__pycache__", "__snapshots__", "venv", "env", "envs", "site-packages", "cache", "caches", "tmp", "temp", "logs",
  "artifacts", "storybook-static", "htmlcov", "private", "certs", "certificates",
]);
const EXTENSIONS = new Set([
  ".md", ".mdx", ".rst", ".txt", ".adoc", ".json", ".jsonc", ".yaml", ".yml", ".toml", ".xml",
  ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs", ".py", ".pyi", ".rs", ".go",
  ".c", ".h", ".cc", ".hh", ".cpp", ".hpp", ".cs", ".java", ".kt", ".kts", ".scala", ".swift", ".rb", ".php",
  ".lua", ".sh", ".bash", ".zsh", ".fish", ".sql", ".graphql", ".gql", ".proto", ".html", ".css", ".scss", ".sass",
  ".less", ".vue", ".svelte", ".ex", ".exs", ".erl", ".hrl", ".hs", ".clj", ".cljs", ".cljc", ".edn", ".elm",
  ".ml", ".mli", ".fs", ".fsx", ".r", ".jl", ".dart", ".pl", ".pm", ".groovy", ".gradle", ".cmake",
]);
const SPECIAL_FILES = new Set(["readme", "makefile", "dockerfile", "containerfile", "gemfile", "rakefile", "pipfile", "go.mod"]);
const MANIFEST = /^(?:package\.json|pyproject\.toml|cargo\.toml|go\.mod|deno\.jsonc?|composer\.json|requirements\.txt|gemfile|pipfile|cmakelists\.txt|makefile)$/;
const SENSITIVE = /(?:^|[._-])(?:env|dotenv|secrets?|credentials?|passwords?|passwd|tokens?|keys?|api[_-]?keys?|access[_-]?tokens?|private[_-]?keys?|service[_-]?accounts?|id_rsa|id_dsa|id_ecdsa|id_ed25519|kubeconfig|keyfile|keyring|keystore|netrc|npmrc|pypirc|htpasswd|shadow)(?:[._-]|$)/i;
const GENERATED = /(?:^|[._-])(?:generated|gen|lock)(?:[._-]|$)|\.(?:min|bundle)\.[cm]?[jt]s$|\.d\.[cm]?ts$|^(?:yarn\.lock|go\.sum|package-lock\.json|pnpm-lock\.yaml|bun\.lockb?)$/i;
const SECRET_CONTENT = /-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{15,})|["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|tokens?|secrets?|passwords?|credentials?|private[_-]?key)["']?\s*[:=]\s*["'`][^"'`\r\n]{3,}/i;
// Conservative screening: reject whole files rather than leaking unquoted YAML/shell
// credentials or authenticated URLs. This is best-effort, not a secret-scanning guarantee.
const UNQUOTED_SECRET = /["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|tokens?|secrets?|passwords?|passwd|credentials?|private[_-]?key)["']?\s*[:=]\s*[^\s#\r\n]+|\bauthorization\s*[:=]\s*["']?(?:bearer|basic)\s+\S+|\b[a-z][a-z0-9+.-]*:\/\/[^\s/:@]+:[^\s/@]+@/i;
const DECLARATION = /^\s*(?:export\b|import\b|from\b|package\b|use\b|pub\b|(?:async\s+)?(?:function|def|class|interface|type|enum|struct|fn|trait|module|namespace)\b|(?:public|protected)\b|#{1,3}\s)/;

function ceiling(value: number, maximum: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(maximum, Math.floor(value))) : 0;
}

class Interrupted extends Error {
  readonly reason: "abort" | "timeout" | "finished";
  constructor(reason: "abort" | "timeout" | "finished") {
    super(reason);
    this.reason = reason;
  }
}

class Discovery {
  readonly context: ProjectContext;
  readonly maxFiles: number;
  readonly maxBytes: number;
  readonly timeoutMs: number;
  readonly deadline: number;
  readBytes = 0;
  outputBytes = 0;
  openedFiles = 0;
  stopped = false;
  readonly signal?: AbortSignal;

  constructor(cwd: string, options: DiscoveryOptions, signal?: AbortSignal) {
    this.signal = signal;
    this.context = { root: resolve(cwd), inventory: [], excerpts: [], omitted: false, notes: [] };
    this.maxFiles = ceiling(options.maxFiles, MAX_FILES);
    this.maxBytes = ceiling(options.maxBytes, MAX_BYTES);
    this.timeoutMs = ceiling(options.timeoutMs, MAX_TIMEOUT_MS);
    this.deadline = performance.now() + this.timeoutMs;
  }

  check(): void {
    if (this.signal?.aborted) throw new Interrupted("abort");
    if (this.stopped) throw new Interrupted("finished");
    if (performance.now() >= this.deadline) throw new Interrupted("timeout");
  }

  async wait<T>(operation: () => Promise<T>): Promise<T> {
    this.check();
    const result = await operation();
    this.check();
    return result;
  }

  omit(note: string): void {
    this.context.omitted = true;
    if (!this.context.notes.includes(note)) this.context.notes.push(note);
  }
}

function parts(path: string): string[] | undefined {
  if (!path || path.length > MAX_PATH_BYTES || Buffer.byteLength(path) > MAX_PATH_BYTES || isAbsolute(path) || win32.isAbsolute(path)
    || /[\\\x00-\x1f\x7f:]/.test(path)) return;
  const segments = path.split("/");
  if (segments.length > MAX_DEPTH + 1 || segments.some((part) => !part || part.startsWith("."))) return;
  for (const part of segments) {
    // Split camelCase too: serviceAccount.json and apiKey.ts are sensitive names.
    if (SENSITIVE.test(part.replace(/([a-z])([A-Z])/g, "$1_$2"))) return;
  }
  return segments;
}

function allowed(path: string, directory = false): boolean {
  const segments = parts(path);
  if (!segments) return false;
  const directories = directory ? segments : segments.slice(0, -1);
  if (directories.some((part) => EXCLUDED_DIRECTORIES.has(part.toLowerCase()) || GENERATED.test(part))) return false;
  if (directory) return segments.length <= MAX_DEPTH;
  const name = segments.at(-1)!.toLowerCase();
  return !GENERATED.test(name) && (EXTENSIONS.has(extname(name)) || SPECIAL_FILES.has(name));
}

/** Check every component, not just the leaf; never follow a project symlink. */
async function safeStat(discovery: Discovery, path: string) {
  const root = discovery.context.root;
  const rootStat = await discovery.wait(() => lstat(root));
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink() || await discovery.wait(() => realpath(root)) !== root) return;
  let absolute = root;
  const segments = path ? parts(path) : [];
  if (!segments) return;
  let stat = rootStat;
  for (let i = 0; i < segments.length; i++) {
    absolute = join(absolute, segments[i]);
    stat = await discovery.wait(() => lstat(absolute));
    if (stat.isSymbolicLink() || (i < segments.length - 1 && !stat.isDirectory())) return;
  }
  if (await discovery.wait(() => realpath(absolute)) !== absolute) return;
  return stat;
}

function ranking(prompt: string): (path: string) => number {
  const boundedPrompt = prompt.slice(0, 16_384).toLowerCase();
  const words = [...new Set(boundedPrompt.match(/[a-z0-9][a-z0-9_-]{2,}/g) ?? [])]
    .filter((word) => !/^(?:the|and|for|with|this|that|file|files|please|project|from|into|fix|add|update)$/.test(word)).slice(0, 64);
  return (path) => {
    const lower = path.toLowerCase();
    const name = basename(lower);
    const root = !path.includes("/");
    let score = /^(?:agents|claude|gemini)\.md$/.test(name) ? (root ? 10_000 : 4500)
      : /^readme(?:\.|$)/.test(name) ? (root ? 9000 : 4000)
      : MANIFEST.test(name) ? (root ? 8000 : 3500) : 0;
    if (boundedPrompt.includes(lower)) score += 7000;
    for (const word of words) if (lower.includes(word)) score += 500;
    if (/^(?:src|lib|app)\//.test(lower)) score += 100;
    return score - path.split("/").length;
  };
}

async function candidates(discovery: Discovery, prompt: string): Promise<string[]> {
  const rank = ranking(prompt);
  const paths = new Set(ROOT_HINTS);
  // Explicit prompt paths are still untrusted: the same policy and filesystem checks apply.
  for (const path of (prompt.slice(0, 16_384).match(/[\w@.-]+(?:\/[\w@.-]+)+/g) ?? []).slice(0, 32)) {
    if (allowed(path)) paths.add(path);
  }
  const queue = [""];
  let entries = 0;
  let directories = 0;
  while (queue.length && entries < MAX_ENTRIES && directories < MAX_DIRECTORIES) {
    discovery.check();
    queue.sort((a, b) => rank(b) - rank(a) || a.localeCompare(b));
    const path = queue.shift()!;
    try {
      const stat = await safeStat(discovery, path);
      if (!stat?.isDirectory()) { discovery.omit("Unsafe or unavailable paths were skipped."); continue; }
      discovery.check();
      const directory = await opendir(join(discovery.context.root, path), { bufferSize: 32 });
      try {
        discovery.check();
        const checked = await safeStat(discovery, path);
        if (!checked || checked.dev !== stat.dev || checked.ino !== stat.ino) continue;
        directories++;
        let localEntries = 0;
        while (entries < MAX_ENTRIES && localEntries < MAX_DIRECTORY_ENTRIES) {
          const entry = await discovery.wait(() => directory.read());
          if (!entry) break;
          entries++;
          localEntries++;
          const relative = path ? `${path}/${entry.name}` : entry.name;
          if (entry.isSymbolicLink() || !allowed(relative, entry.isDirectory())) {
            discovery.omit("Excluded paths or non-source files were skipped.");
          } else if (entry.isDirectory()) {
            if (queue.length < MAX_DIRECTORIES) queue.push(relative);
            else discovery.omit("Traversal limit reached.");
          } else if (entry.isFile()) paths.add(relative);
          else discovery.omit("Excluded paths or non-source files were skipped.");
        }
        if (localEntries >= MAX_DIRECTORY_ENTRIES || entries >= MAX_ENTRIES) discovery.omit("Traversal limit reached.");
      } finally {
        await directory.close().catch(() => {});
      }
    } catch (error) {
      discovery.check();
      if (error instanceof Interrupted) throw error;
      discovery.omit("Unsafe or unavailable paths were skipped.");
    }
  }
  if (queue.length) discovery.omit("Traversal limit reached.");
  return [...paths].sort((a, b) => rank(b) - rank(a) || a.localeCompare(b));
}

async function source(discovery: Discovery, path: string): Promise<{ text: string; fingerprint: string } | undefined> {
  if (!allowed(path)) { discovery.omit("Excluded paths or non-source files were skipped."); return; }
  let handle: FileHandle | undefined;
  try {
    const before = await safeStat(discovery, path);
    if (!before?.isFile()) { discovery.omit("Unsafe or unavailable paths were skipped."); return; }
    if (before.size > MAX_FILE_BYTES || before.size > discovery.maxBytes - discovery.readBytes) {
      discovery.omit("File or byte limit reached."); return;
    }
    if (discovery.openedFiles >= discovery.maxFiles) { discovery.omit("File or byte limit reached."); return; }
    discovery.check();
    discovery.openedFiles++;
    // NONBLOCK prevents a concurrent replacement with a FIFO from hanging open().
    handle = await open(join(discovery.context.root, path), constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    discovery.check();
    const opened = await discovery.wait(() => handle!.stat());
    const checked = await safeStat(discovery, path);
    if (!opened.isFile() || !checked || opened.dev !== before.dev || opened.ino !== before.ino
      || opened.size !== before.size || checked.dev !== opened.dev || checked.ino !== opened.ino) {
      discovery.omit("Unsafe or changing files were skipped."); return;
    }
    // Reserve the entire bounded read, even if the read later fails or the file is rejected.
    discovery.readBytes += opened.size;
    const bytes = Buffer.alloc(opened.size);
    let offset = 0;
    while (offset < bytes.length) {
      const result = await discovery.wait(() => handle!.read(bytes, offset, bytes.length - offset, offset));
      if (!result.bytesRead) break;
      offset += result.bytesRead;
    }
    const after = await discovery.wait(() => handle!.stat());
    const final = await safeStat(discovery, path);
    if (offset !== bytes.length || !final || final.dev !== opened.dev || final.ino !== opened.ino
      || after.size !== opened.size || after.mtimeMs !== opened.mtimeMs || after.ctimeMs !== opened.ctimeMs) {
      discovery.omit("Unsafe or changing files were skipped."); return;
    }
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    if (/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]/.test(text) || SECRET_CONTENT.test(text) || UNQUOTED_SECRET.test(text)
      || /(?:@generated|code generated|automatically generated|auto-generated)/i.test(text.slice(0, 2048))) {
      discovery.omit("Binary, generated, or sensitive content was skipped."); return;
    }
    return { text, fingerprint: createHash("sha256").update(bytes).digest("hex") };
  } catch (error) {
    discovery.check();
    if (error instanceof Interrupted) throw error;
    // Missing hints/explicit paths are ordinary; do not report them as project contents.
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") discovery.omit("Unsafe or unreadable files were skipped.");
  } finally {
    await handle?.close().catch(() => {});
  }
}

function truncate(text: string, limit: number): string {
  if (Buffer.byteLength(text) <= limit) return text;
  if (limit <= Buffer.byteLength(OMISSION)) return "";
  const bytes = Buffer.from(text);
  let end = limit - Buffer.byteLength(OMISSION);
  // Do not split a UTF-8 code point at the excerpt boundary.
  while (end > 0 && (bytes[end] & 0xc0) === 0x80) end--;
  return bytes.subarray(0, end).toString("utf8") + OMISSION;
}

function excerpt(path: string, text: string, budget: number): string {
  if (/\.(?:mdx?|rst|txt|adoc|jsonc?|ya?ml|toml|xml)$/.test(path.toLowerCase()) || SPECIAL_FILES.has(basename(path).toLowerCase())) {
    return truncate(text, budget);
  }
  const lines = text.split("\n");
  if (lines.length <= 32) return truncate(text, budget);
  const selected: string[] = [];
  let previous = -1;
  for (let i = 0; i < lines.length && selected.length < 40; i++) {
    if (i >= 8 && !DECLARATION.test(lines[i])) continue;
    if (previous >= 0 && i > previous + 1) selected.push("[...source omitted...]");
    selected.push(lines[i]);
    previous = i;
  }
  if (previous < lines.length - 1) selected.push("[...source omitted...]");
  return truncate(selected.join("\n"), budget);
}

function fitSerializedText(text: string, budget: number): string {
  const cost = (value: string) => Buffer.byteLength(JSON.stringify(value)) - 2;
  if (cost(text) <= budget) return text;
  let low = 0;
  let high = Buffer.byteLength(text);
  let fitted = "";
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = truncate(text, middle);
    if (cost(candidate) <= budget) { fitted = candidate; low = middle + 1; }
    else high = middle - 1;
  }
  return fitted;
}

async function collect(discovery: Discovery, paths: string[], inventory: boolean): Promise<void> {
  for (const path of paths) {
    discovery.check();
    if (discovery.openedFiles >= discovery.maxFiles || discovery.outputBytes >= discovery.maxBytes) {
      discovery.omit("File or byte limit reached."); break;
    }
    const inventoryBytes = inventory ? Buffer.byteLength(JSON.stringify(path)) + 1 : 0;
    if (inventoryBytes > discovery.maxBytes - discovery.outputBytes) { discovery.omit("File or byte limit reached."); continue; }
    const data = await source(discovery, path);
    if (!data) continue;
    if (inventory) discovery.context.inventory.push(path);
    discovery.outputBytes += inventoryBytes;
    if (inventory && discovery.context.excerpts.length >= MAX_EXCERPTS) { discovery.omit("Source excerpts are partial."); continue; }
    // Charge serialized entries, including escaped text, repeated paths, hashes, and commas.
    const overhead = Buffer.byteLength(JSON.stringify({ path, text: "", fingerprint: data.fingerprint })) + 1;
    const budget = Math.min(MAX_EXCERPT_BYTES, discovery.maxBytes - discovery.outputBytes - overhead);
    if (budget <= 0) { discovery.omit("File or byte limit reached."); continue; }
    const text = fitSerializedText(excerpt(path, data.text, budget), budget);
    if (!text && data.text) { discovery.omit("File or byte limit reached."); continue; }
    if (text !== data.text) discovery.omit("Source excerpts are partial.");
    discovery.context.excerpts.push({ path, text, fingerprint: data.fingerprint });
    discovery.outputBytes += overhead + Buffer.byteLength(JSON.stringify(text)) - 2;
  }
}

async function run(cwd: string, options: DiscoveryOptions, signal: AbortSignal | undefined, work: (discovery: Discovery) => Promise<void>): Promise<ProjectContext> {
  const discovery = new Discovery(cwd, options, signal);
  let timer: ReturnType<typeof setTimeout> | undefined;
  let onAbort: (() => void) | undefined;
  const interrupted = new Promise<never>((_resolve, reject) => {
    onAbort = () => reject(new Interrupted("abort"));
    signal?.addEventListener("abort", onAbort, { once: true });
    timer = setTimeout(() => reject(new Interrupted("timeout")), discovery.timeoutMs);
  });
  const operation = (async () => {
    discovery.check();
    if (!discovery.maxFiles || !discovery.maxBytes) { discovery.omit("File or byte limit reached."); return; }
    try {
      const stat = await discovery.wait(() => lstat(discovery.context.root));
      if (!stat.isDirectory() || stat.isSymbolicLink()) { discovery.omit("Project directory unavailable or unsafe."); return; }
      discovery.context.root = await discovery.wait(() => realpath(discovery.context.root));
    } catch (error) {
      discovery.check();
      if (error instanceof Interrupted) throw error;
      discovery.omit("Project directory unavailable or unsafe."); return;
    }
    await work(discovery);
    if (!discovery.context.inventory.length && !discovery.context.excerpts.length) {
      discovery.context.notes.push("No eligible project files found.");
    }
  })();
  try {
    await Promise.race([operation, interrupted]);
  } catch (error) {
    if (error instanceof Interrupted && error.reason === "abort") throw new DOMException("Project discovery aborted.", "AbortError");
    if (error instanceof Interrupted && error.reason === "timeout") discovery.omit("Discovery timed out.");
    else throw error;
  } finally {
    discovery.stopped = true;
    clearTimeout(timer);
    if (onAbort) signal?.removeEventListener("abort", onAbort);
  }
  // A timed-out filesystem operation may finish later. It cannot mutate this snapshot,
  // starts no more I/O, and closes any acquired handle in its finally block.
  return { ...discovery.context, inventory: [...discovery.context.inventory], excerpts: [...discovery.context.excerpts], notes: [...discovery.context.notes] };
}

/**
 * Read-only, cwd-local discovery; no parent search, command execution, or cache.
 * maxFiles bounds opened files (including rejected content). maxBytes independently
 * bounds bytes read and serialized UTF-8 inventory/excerpt entries, including
 * escaped text, paths, hashes, and separators. Root, fixed notes, and the enclosing
 * object/array framing are metadata outside that payload budget.
 * Timeouts return partial, marked context; cancellation rejects with AbortError.
 */
export async function discoverProjectContext(cwd: string, prompt: string, options: DiscoveryOptions, signal?: AbortSignal): Promise<ProjectContext> {
  return run(cwd, options, signal, async (discovery) => collect(discovery, await candidates(discovery, prompt), true));
}

/** Additional relative paths use the identical policy, depth, read, size, and time limits. */
export async function readProjectExcerpts(cwd: string, paths: string[], options: DiscoveryOptions, signal?: AbortSignal): Promise<ProjectContext["excerpts"]> {
  const context = await run(cwd, options, signal, async (discovery) => {
    const selected = new Set<string>();
    // Bound even the caller's candidate list; duplicates cannot multiply file reads.
    for (let i = 0; i < Math.min(paths.length, MAX_ENTRIES); i++) {
      discovery.check();
      if (allowed(paths[i])) selected.add(paths[i]);
    }
    await collect(discovery, [...selected], false);
  });
  return context.excerpts;
}
