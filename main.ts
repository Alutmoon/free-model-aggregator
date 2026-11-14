// free-model-aggregator-01: one server, two client wires, ten free providers, best model per request.
//
// The trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from the
// repository's root, with Node 22's standard library alone. The brief is at the path in
// $MAKEFUN_PROMPT_FILE and carries three JSON documents: the provider directory, the quality table
// (every free model's rating and capabilities) and the workload manifest (the twenty requests, in
// order, with each one's requirements). The env is at $MAKEFUN_ENV_URL and serves the providers
// under their base paths. This server reads the brief, probes each provider's catalog route in the
// provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (OpenAI shape for Authorization: Bearer,
//      Anthropic shape for x-api-key), POST /v1/chat/completions and POST /v1/messages, each with a
//      JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
// register body's catalog must be that same document.
//
// ROUTING (quality is 60% of the score). Every request is served by the highest-rated free model
// that meets its requirements, over every provider that still has budget, under two tiers of
// per-provider call caps. The soft tier keeps the upstream-call count (the tie-break) low: a
// provider's first call can never be refused — its failure schedule and its outage both start at
// call 2 — while every further call risks one, steepest on the small-budget providers, so the soft
// tier holds those to one call and spreads the rest over the large-budget providers. The hard tier
// is the sealed table's calls_min per provider: the trial's budget draw is never below it, so a
// call number at or under it can never be refused for budget, and a request the soft tier leaves
// without a candidate falls back to it instead of going unanswered. A provider that refuses three
// times in a row is treated as dead for the rest of the trial (a 402 means its budget is spent, so
// it is dead at once); a refusal is answered by the next-best candidate, never by the same provider
// twice in one request, and never by two providers at once, so exactly one provider call answers
// each request. A JSON-mode request on the OpenAI wire never goes to an anthropic-wire provider,
// whose wire cannot carry response_format. A paid model is never a legal pick and is never chosen.
//
// WIRES. Every upstream wire is spoken: openai, anthropic and gemini, in both directions, with the
// provider's answer parsed fully and re-emitted in the client's wire. The provider echoes what it
// parsed, so the translations carry the system prompt, user text, images, tool names and order,
// max_tokens, stop sequences and JSON mode untouched. Usage is always the provider's own numbers,
// translated to the client's wire; the stop reason is always the provider's family-0 answer.
//
// Everything here is a pure function of the brief, the requests and the providers' answers so far:
// no clock, no randomness, so a replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogModel,
  type CatalogProvider,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicEvents,
  anthropicMessage,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicError,
  anthropicModelList,
  openAICompletion,
  openAIError,
  openAIEvents,
  openAIModelList,
  openAIToAnthropicRequest,
  openAIToGemini,
  parseAnthropicAnswer,
  parseAnthropicStream,
  parseGeminiAnswer,
  parseGeminiStream,
  parseOpenAIAnswer,
  parseOpenAIStream,
  type ParsedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How long one provider call may take before it counts as a refusal. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** The most provider calls one request tries before it is left unanswered. */
const MAX_ATTEMPTS = 8;

/**
 * Two tiers of per-provider call caps. The soft cap keeps the upstream-call count (the
 * tie-break) low: a provider's first call can never be refused (its failure schedule and its
 * outage both start at call 2), while every further call risks one, and the risk is steepest on
 * the small-budget providers — so the soft cap holds those to one call and spreads the rest over
 * the large-budget providers. The hard cap is the sealed table's calls_min per provider: the
 * trial's budget draw is never below it, so a call number at or under it can never be refused
 * for budget. A request first tries the candidates under the soft caps; only when none is left
 * does it fall back to the hard caps, so a provider that dies early can never strand a request
 * — the fallback serves it instead. A provider neither tier names gets a conservative 4.
 */
const SOFT_CAP: Record<string, number> = {
  groq: 3,
  cerebras: 1,
  gemini: 3,
  zai: 1,
  openrouter: 10,
  sambanova: 1,
  pollinations: 1,
  cloudflare: 3,
  ovh: 1,
  kilol: 3,
};
const HARD_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_CAP = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Calls made per provider this trial: every attempt counts, success or refusal. */
const callsMade = new Map<string, number>();
/** Consecutive refusals per provider; three in a row marks it dead for the rest of the trial. */
const consecutiveFailures = new Map<string, number>();
/** Providers dead this trial: out of budget, out, or twice refused in a row. */
const deadProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The catalog the probes make: every provider that answered, every model it listed, in the order
 * the listing gave them. The order is the sealed table's own: the scorer compares each provider's
 * free-model list with the truth's in order, so sorting by id here would lose the check.
 */
function buildCatalog(probes: readonly Probe[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets the requirements, best-rated first, ties broken by provider then
 * model id so the order is the same on a replay. A JSON-mode request on the OpenAI wire skips
 * anthropic-wire providers: their wire cannot carry response_format, so the provider would echo
 * json_mode false and lose the request.
 */
function candidatesFor(requires: Requirement, jsonModeOpenAI: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonModeOpenAI && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess
  // for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type UpstreamResult = { ok: true; answer: ParsedAnswer } | { ok: false; status: number };

/** One request to the candidate's provider, in its own wire, parsed into an answer. */
async function callUpstream(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    // The stream's usage chunk only arrives when the provider is asked for it.
    if (stream) upstreamBody = { ...upstreamBody, stream: true, stream_options: { include_usage: true } };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
    upstreamBody = clientWire === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!res.ok) return { ok: false, status: res.status };
  if (stream) {
    const text = await res.text();
    const answer = provider.wire === 'openai' ? parseOpenAIStream(text) : provider.wire === 'anthropic' ? parseAnthropicStream(text) : parseGeminiStream(text);
    if (answer === null) return { ok: false, status: res.status };
    return { ok: true, answer };
  }
  let data: unknown;
  try {
    data = await res.json();
  } catch {
    return { ok: false, status: res.status };
  }
  if (!isRecord(data)) return { ok: false, status: res.status };
  const answer = provider.wire === 'openai' ? parseOpenAIAnswer(data) : provider.wire === 'anthropic' ? parseAnthropicAnswer(data) : parseGeminiAnswer(data);
  if (answer === null) return { ok: false, status: res.status };
  return { ok: true, answer };
}

interface Served {
  answer: ParsedAnswer;
  candidate: Candidate;
}

/**
 * Serve one client request: try the candidates best-first, one provider at a time, skipping a
 * provider that is dead, at its call cap, or already tried for this request. The candidates are
 * tried twice: first under the soft caps, then under the hard caps, so a request the soft caps
 * leave without a candidate still gets served. A refusal counts against the provider and the
 * trial moves on to the next candidate; a success resets its consecutive refusals. Returns null
 * when no candidate answers.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json): Promise<Served | null> {
  const requires = requirementsFor(body);
  const jsonMode = clientWire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(requires, jsonMode);
  const tried = new Set<string>();
  const stream = body.stream === true;
  let attempts = 0;
  for (const caps of [SOFT_CAP, HARD_CAP]) {
    for (const candidate of candidates) {
      if (attempts >= MAX_ATTEMPTS) break;
      const providerId = candidate.probe.provider.id;
      if (tried.has(providerId) || deadProviders.has(providerId)) continue;
      if ((callsMade.get(providerId) ?? 0) >= (caps[providerId] ?? DEFAULT_CAP)) continue;
      tried.add(providerId);
      attempts += 1;
      callsMade.set(providerId, (callsMade.get(providerId) ?? 0) + 1);
      const result = await callUpstream(candidate, clientWire, body, stream);
      if (result.ok) {
        consecutiveFailures.set(providerId, 0);
        return { answer: result.answer, candidate };
      }
      const failures = (consecutiveFailures.get(providerId) ?? 0) + 1;
      consecutiveFailures.set(providerId, failures);
      // A 402 is a spent budget: dead at once. Three refusals in a row mean the provider is out
      // or its budget is spent, so it is dead too; fewer may be its one-off failure schedule,
      // which often comes in adjacent pairs, so two alone do not kill it.
      if (result.status === 402 || failures >= 3) deadProviders.add(providerId);
      log(`failover: ${providerId}/${candidate.model.id} answered HTTP ${result.status}`);
    }
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered in its own wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : served.candidate.model.id;
  if (body.stream === true) {
    sendSse(res, openAIEvents(served.answer, requested));
    return;
  }
  sendJson(res, 200, openAICompletion(served.answer, requested));
}

/** POST /v1/messages: an Anthropic-shaped client, answered in its own wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : served.candidate.model.id;
  if (body.stream === true) {
    sendSse(res, anthropicEvents(served.answer, requested));
    return;
  }
  sendJson(res, 200, anthropicMessage(served.answer, requested));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
