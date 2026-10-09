# jev-llm-router

A Pi extension that selects a model and thinking level for each new user prompt using Jev.

## Install

Requires [Pi](https://pi.dev) 1.0.0 or newer, Node.js 22.19.0 or newer, and Git. Install directly from GitHub:

```bash
pi install git:github.com/emo333/jev-llm-router
```

This installs for your Pi user. Add `--local` to install for the current project instead. Pi loads the TypeScript source directly, so no build step or npm publication is needed.

The command above tracks the repository's default branch. To pin a published release, append its tag to the source, such as `@v0.1.0` once that release exists. Tagged installs stay pinned when updating.

## Use

Start a new Pi session after installing:

```bash
pi --model jev/auto
```

Before sending a prompt, authenticate your candidate providers with `/login` and select the candidate models with `/scoped-models`. An explicit scope is required. The router never expands it to all available models.

Authenticate Jev through `/login typesafe` or set `TYPESAFE_API_KEY` before starting Pi. It can also use an authenticated Jev classifier from another Pi provider, including OpenRouter.

The router uses your saved scope. You can override the scope for one invocation with Pi's `--models` option. Alternatively, load the extension and select `jev/auto` through `/model`. Selecting a physical model bypasses routing.

Pi's footer shows the dispatched model and thinking level. The virtual model's own level is `off`, meaning it supplies no fixed reasoning budget. Jev chooses the physical level automatically and ignores thinking-level defaults attached to scoped model entries.

### Thinking-level caps

Run `/jev` to open the scoped-candidate picker. Each model shows its effective highest allowed thinking level. Select a model, then choose one of its supported levels as the cap. Choose `Use model maximum` to remove its cap. Pi saves changes automatically and applies them on the next routed request.

Caps are stored in `<agent-dir>/jev-llm-rtr.json`, where `<agent-dir>` is `~/.pi/agent` by default or the directory selected by `PI_CODING_AGENT_DIR`. You can also edit this file directly:

```json
{
  "thinkingLevelCaps": {
    "anthropic/claude-sonnet-4-5": "high",
    "openai/gpt-5": "medium"
  }
}
```

Keys use the exact `provider/model-id`. A model without an entry defaults to its highest supported effective control. Caps apply to the provider-native control, not just Pi's label. Equivalent levels are collapsed. A forced-high model is excluded by a lower cap, and ambiguous sampling overrides are excluded when a cap is configured. The config is read on each route, so direct file edits apply without reloading Pi. Without a UI, `/jev` prints scoped levels.

## Update or remove

Update the default-branch install:

```bash
pi update git:github.com/emo333/jev-llm-router
```

Or remove it:

```bash
pi remove git:github.com/emo333/jev-llm-router
```

For a tagged install, use the same tagged source when updating or removing. To switch releases, run `pi install` with the new tag. Add `--local` when removing a project install. Restart Pi after updating.

## Policy

The router first filters authenticated, explicitly scoped models by image support, effective thinking caps, and full-transcript context fit. Pi's usage-aware token estimate includes tools, images, and thinking without sending private thinking to Jev. Context fit reserves output/reasoning room plus a configurable safety margin. Tokenization and later context hooks can still change the final provider payload.

One batched Jev call estimates each candidate's minimum sufficient **effective native control**, the strongest permitted candidate, task family, risk, verifiability, execution readiness, phase, and answer length. Every qualifying model/control pair is ranked, rather than choosing a model before pricing its effort. The default cumulative threshold is 0.967. This is an advisory classifier quantile, not a measured 96.7% task-success rate.

High-risk tasks, uncertain risk, truncated prompt/system inputs, and images use the strongest scoped candidate at its highest fitting permitted control. Routine low-risk tasks use the cheapest qualifying estimated pair. No stronger model outside your scope or above your caps is introduced.

Cost uses catalog tiers, predicted total billable output including reasoning, cumulative context growth, observed cache-write rates, and inferred recovery overhead. Recent witnessed same-model cache hits can provide a conservative cache-read discount. Observed output and elapsed provider-call time replace cold estimates for the same model/control/task family. Reasoning is a subset of output and is not counted twice. Catalog prices are not subscription billing, and cold latency, task turns, recovery work, and future cache hits remain estimates.

Ordinary tool continuations keep their pair without constructing a classifier projection. Availability failures exclude the failed provider/model for the current task and reassess only remaining scoped candidates. Repeated substantive check failures after corrective edits, configured verification failures, and newly discovered protected work can trigger reassessment. A baseline red test, an unchanged rerun, or an environment error is not automatically evidence of inadequate intelligence. Escalations are bounded. New user prompts reset task-local exclusions and are reassessed.

Planning can hand off to cheaper bounded execution only after an explicit source plan and acceptance checks, successful edit progress, a fresh confident execution assessment, and enough versioned **user-accepted task outcomes** for the execution pair. A successful first edit or a generic passing check cannot authorize a downgrade. Without that evidence, the planning pair is retained. State follows Pi's session branches. Direct requests, including compaction, reuse a fitting permitted previous pair when possible.

Local model discovery uses a registry-local 30-second cache, bounded to 64 entries and simultaneous lookups. Endpoint, authentication, and model-configuration changes invalidate identity. Hot-swapped server models with unchanged configuration can remain cached until the TTL expires. Classification is skipped when only one eligible model/control pair exists; that deterministic path does not establish task verifiability.

### Policy settings

Optional settings share `jev-llm-rtr.json` with thinking caps:

```json
{
  "thinkingLevelCaps": {},
  "classifier": {
    "provider": "openrouter",
    "id": "typesafe/jev-1.13"
  },
  "policy": {
    "qualityThreshold": 0.967,
    "protectedThreshold": 0.995,
    "contextSafetyTokens": 1024,
    "latencyUsdPerSecond": 0,
    "maxEscalations": 2,
    "historyEnabled": true,
    "phaseRouting": true,
    "minimumCalibrationSamples": 30
  }
}
```

The configured classifier must be authenticated and available in Pi. If omitted, the router prefers direct `typesafe/jev-latest`, then another authenticated Jev. A configured identity is never silently substituted. Mutable classifier aliases do not produce version-stable quality calibration. Pin an available release before collecting calibration labels.

`protectedThreshold` governs the stricter calibrated execution handoff. `minimumCalibrationSamples` is only a floor; conservative lower bounds may require substantially more observations to satisfy the threshold. `latencyUsdPerSecond` assigns a dollar-equivalent penalty to elapsed seconds. At its default of zero, latency breaks cost ties. Unknown policy keys or invalid values are rejected.

### Outcomes and learned profiles

Use `/jev stats` to inspect observed usage, task-family reliability profiles, and calibration sample counts. After independently checking the complete task, use `/jev outcome pass` or `/jev outcome fail` to label the latest routed pair on the current branch.

History is stored in `<agent-dir>/jev-llm-rtr-history.json` with a versioned schema, bounded numeric aggregates, hashed identities, and queued atomic writes. It does not store conversation text, image data, credentials, or hidden reasoning. Model configuration/identity, native control, task family, classifier release, and prediction bin isolate quality calibration. The first physical reliability observation and first explicit user-quality label per task/pair are retained independently. Later labels do not correct or inflate them. Recovery pairs can receive separate outcomes.

Quality calibration uses conservative lower bounds from explicit user-acceptance labels and never raises the raw prediction. Configured check outcomes affect reliability/recovery estimates only. These records are not an independent benchmark or a guarantee of task correctness. Evaluate representative held-out tasks against a quality-first scoped baseline before lowering thresholds.

Set `historyEnabled` to `false` to disable learning and persistence. Existing history is left untouched. Corrupt or incompatible history is reported rather than silently reset. Usage-recording failures emit a warning.

### Independent acceptance checks

Checks are opt-in because they execute a shell command in the project directory:

```json
{
  "thinkingLevelCaps": {},
  "verification": {
    "command": "npm test",
    "timeoutMs": 60000,
    "maxAttempts": 3
  }
}
```

Use a trusted, task-appropriate command without destructive side effects. The check runs before settlement only for edited tasks classified as low-risk and independently verifiable. Failure inserts actionable check output and requests bounded corrective continuation with capability reassessment. Timeouts and shell exits 126/127 leave the task unverified without recording a capability failure. Exhaustion reports unmet acceptance checks instead of claiming success. A passing command verifies only its configured contract and does not create a task-quality calibration sample.

## Context and failures

**Routing sends conversation text to the selected Jev provider**, even when the generation model is local. The visible-text projection shares a 48,000-character content budget across the latest prompt, system instructions/tools, recent conversation, source-attributed requirement excerpts, failure excerpts, and omission notices. The prompt and system fields are capped at 16,000 and 8,000 characters. Important middle and older constraints are prioritized, but extraction is not a complete task specification. The generation model still receives Pi's normal context.

Image bytes, hidden reasoning, and provider signatures are not sent to Jev. Image requests are restricted to image-capable scoped models, and their unseen details trigger protected routing.

Missing scope, insufficient fitting candidates, missing required Jev authentication, classifier failures, invalid fallback decisions, or exhausted capability recovery stop the request with an error rather than choosing an arbitrary model. Classification has a 10-second deadline and respects cancellation. A sole eligible model/control pair does not require Jev authentication.

## Development

```bash
git clone https://github.com/emo333/jev-llm-router.git
cd jev-llm-router
npm ci --ignore-scripts
npm run check
npm test
```

Load the local package while developing:

```bash
pi --extension . --model jev/auto
```

The subagent display integration test is optional. Set `PI_SUBAGENT_EXTENSION` to the path of an installed subagent extension to include it in `npm test`.

## GitHub releases

The repository must be public for anyone to install without repository access. Review tracked files and Git history before changing its visibility.

After the checks pass, commit and push the changes to `main`. Create a tag matching the version in `package.json`. For the initial `0.1.0` release:

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

Publish a GitHub release from that tag. No build artifacts are required. Users can then pin it with:

```bash
pi install git:github.com/emo333/jev-llm-router@v0.1.0
```

`package.json` sets `private: true` to prevent accidental npm publishing. This does not restrict installation through Git.
