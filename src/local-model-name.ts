import { createHash } from "node:crypto";
import dns from "node:dns/promises";
import { BlockList, isIP } from "node:net";
import type { Api, Model } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

const LOOKUP_TIMEOUT_MS = 2_000;
const CACHE_TTL_MS = 30_000;
const CACHE_LIMIT = 64;
type Registry = ExtensionContext["modelRegistry"];
interface CacheEntry { name: string; expiresAt: number }
interface Flight { controller: AbortController; promise: Promise<string | undefined>; waiters: number }
interface DiscoveryCache { entries: Map<string, CacheEntry>; flights: Map<string, Flight> }
const caches = new WeakMap<Registry, DiscoveryCache>();

// Hash complete native configuration and resolved authentication rather than retaining
// credentials in cache keys. Metadata edits (including endpoint/headers) change identity.
function fingerprint(value: unknown): string {
  const json = JSON.stringify(value, (_key, item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return item;
    return Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]]));
  });
  return createHash("sha256").update(json ?? "").digest("hex");
}

function cacheFor(registry: Registry): DiscoveryCache {
  let cache = caches.get(registry);
  if (!cache) {
    cache = { entries: new Map(), flights: new Map() };
    caches.set(registry, cache);
  }
  const now = Date.now();
  for (const [key, entry] of cache.entries) if (entry.expiresAt <= now) cache.entries.delete(key);
  return cache;
}
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
function abortable<T>(pending: Promise<T>, signal: AbortSignal): Promise<T> {
  const { promise, resolve, reject } = Promise.withResolvers<T>();
  const abort = () => { signal.removeEventListener("abort", abort); reject(signal.reason); };
  signal.addEventListener("abort", abort, { once: true });
  pending.then(
    (value) => { signal.removeEventListener("abort", abort); resolve(value); },
    (error) => { signal.removeEventListener("abort", abort); reject(error); },
  );
  if (signal.aborted) abort();
  return promise;
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

async function lookupName(model: Model<Api>, registry: Registry, cache: DiscoveryCache, identity: string, signal: AbortSignal): Promise<string | undefined> {
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
  const key = fingerprint({ identity, provider: model.provider, id: model.id, endpoint: url.href, apiKey: auth.apiKey, headers: auth.headers });
  const cached = cache.entries.get(key);
  if (cached && cached.expiresAt > Date.now()) {
    cache.entries.delete(key);
    cache.entries.set(key, cached);
    return cached.name;
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
  const name = apiModelName(await response.json(), model.id);
  signal.throwIfAborted();
  if (name) {
    if (cache.entries.size >= CACHE_LIMIT) cache.entries.delete(cache.entries.keys().next().value!);
    cache.entries.set(key, { name, expiresAt: Date.now() + CACHE_TTL_MS });
  }
  return name;
}

/** Resolve only classifier-facing names. The registered model and its dispatch ID stay unchanged. */
export async function localModelName(model: Model<Api>, registry: ExtensionContext["modelRegistry"], signal: AbortSignal): Promise<string> {
  signal.throwIfAborted();
  if (model.api !== "openai-completions" && model.api !== "openai-responses") return model.name;
  const cache = cacheFor(registry);
  let identity: string;
  try { identity = fingerprint(model); } catch { return model.name; }
  let flight = cache.flights.get(identity);
  if (flight?.controller.signal.aborted) {
    cache.flights.delete(identity);
    flight = undefined;
  }
  if (!flight) {
    // Do not queue unbounded discovery work; configured names remain usable.
    if (cache.flights.size >= CACHE_LIMIT) return model.name;
    const controller = new AbortController();
    const lookupSignal = AbortSignal.any([controller.signal, AbortSignal.timeout(LOOKUP_TIMEOUT_MS)]);
    // Snapshot the configured identity for a flight: mutation during auth/DNS cannot
    // associate discovery metadata with a different registered model configuration.
    let snapshot: Model<Api>;
    try { snapshot = structuredClone(model); } catch { return model.name; }
    const pending = abortable(lookupName(snapshot, registry, cache, identity, lookupSignal), lookupSignal)
      .catch(() => undefined)
      .finally(() => {
        if (cache.flights.get(identity) === created) cache.flights.delete(identity);
      });
    const created: Flight = { controller, promise: pending, waiters: 0 };
    flight = created;
    cache.flights.set(identity, created);
  }
  flight.waiters++;
  try {
    const name = await abortable(flight.promise, signal);
    signal.throwIfAborted();
    try { if (fingerprint(model) !== identity) return model.name; } catch { return model.name; }
    return name ?? model.name;
  } finally {
    flight.waiters--;
    if (flight.waiters === 0 && cache.flights.get(identity) === flight) {
      cache.flights.delete(identity);
      flight.controller.abort();
    }
  }
}
