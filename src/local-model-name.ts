import dns from "node:dns/promises";
import { BlockList, isIP } from "node:net";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

const LOOKUP_TIMEOUT_MS = 2_000;
const localAddresses = new BlockList();
for (const [address, prefix] of [["127.0.0.0", 8], ["10.0.0.0", 8], ["172.16.0.0", 12], ["192.168.0.0", 16], ["169.254.0.0", 16], ["100.64.0.0", 10]] as const) {
  localAddresses.addSubnet(address, prefix, "ipv4");
}
localAddresses.addAddress("::1", "ipv6");
localAddresses.addSubnet("fc00::", 7, "ipv6");
localAddresses.addSubnet("fe80::", 10, "ipv6");

async function isLocalEndpoint(url: URL): Promise<boolean> {
  if (url.protocol !== "http:" && url.protocol !== "https:") return false;
  const hostname = url.hostname.replace(/^\[|\]$/g, "");
  const addresses = isIP(hostname) ? [{ address: hostname }] : await dns.lookup(hostname, { all: true });
  return addresses.length > 0 && addresses.every(({ address }) => localAddresses.check(address, isIP(address) === 6 ? "ipv6" : "ipv4"));
}

// DNS and Pi's auth resolver do not accept a signal. Stop waiting for either on timeout/cancel.
function abortable<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener("abort", abort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
    if (signal.aborted) abort();
  });
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function apiModelName(body: unknown, id: string): string | undefined {
  if (!body || typeof body !== "object" || !("data" in body) || !Array.isArray(body.data)) return undefined;
  const entries = body.data.filter((entry): entry is Record<string, unknown> => !!entry && typeof entry === "object");
  const matches = entries.filter((entry) => entry.id === id || (Array.isArray(entry.aliases) && entry.aliases.includes(id)));
  // A single-model server may accept a generic configured ID such as "llama.cpp".
  // Never guess which entry belongs to a candidate on an ambiguous multi-model server.
  const entry = matches.length === 1 ? matches[0] : matches.length === 0 && body.data.length === 1 ? entries[0] : undefined;
  const name = entry && (text(entry.name) ?? text(entry.id));
  if (!name) return undefined;
  // Preserve the model and quantization, not the server's directory or GGUF shard suffix.
  return /\.gguf$/i.test(name) ? name.split(/[\\/]/).at(-1)!.replace(/(?:-\d{5}-of-\d{5})?\.gguf$/i, "") || undefined : name;
}

async function lookupName(model: Model<Api>, registry: ExtensionContext["modelRegistry"], signal: AbortSignal): Promise<string | undefined> {
  let url = new URL(model.baseUrl);
  if (!(await isLocalEndpoint(url))) return undefined;
  signal.throwIfAborted();
  const auth = await registry.getApiKeyAndHeaders(model);
  signal.throwIfAborted();
  if (!auth.ok) return undefined;
  if (auth.baseUrl && auth.baseUrl !== model.baseUrl) {
    url = new URL(auth.baseUrl);
    if (!(await isLocalEndpoint(url))) return undefined;
    signal.throwIfAborted();
  }
  url.pathname = `${url.pathname.replace(/\/+$/, "") || "/v1"}/models`;
  const headers = new Headers();
  if (auth.apiKey) headers.set("Authorization", `Bearer ${auth.apiKey}`);
  for (const [key, value] of Object.entries(auth.headers ?? {})) {
    if (value === null) headers.delete(key);
    else headers.set(key, value);
  }
  const response = await fetch(url, { headers, signal, redirect: "error" });
  if (!response.ok) {
    await response.body?.cancel();
    return undefined;
  }
  return apiModelName(await response.json(), model.id);
}

/** Resolve only classifier-facing names. The registered model and its dispatch ID stay unchanged. */
export async function localModelName(model: Model<Api>, registry: ExtensionContext["modelRegistry"], signal: AbortSignal): Promise<string> {
  signal.throwIfAborted();
  if (model.api !== "openai-completions" && model.api !== "openai-responses") return model.name;
  const lookupSignal = AbortSignal.any([signal, AbortSignal.timeout(LOOKUP_TIMEOUT_MS)]);
  try {
    return (await abortable(lookupName(model, registry, lookupSignal), lookupSignal)) ?? model.name;
  } catch {
    // Discovery is best-effort, but a canceled/expired route must not proceed to Jev.
    signal.throwIfAborted();
    return model.name;
  }
}
