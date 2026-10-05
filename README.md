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

Keys use the exact `provider/model-id`. A model without an entry defaults to its highest supported level. Caps limit Jev's choices and sticky follow-up routes. If a cap is below every level supported by a model, Jev excludes that candidate. The config is read on each route, so direct file edits also apply without reloading Pi. Without a UI, `/jev` prints the current scoped levels without opening the picker.

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

One Jev call estimates each candidate's minimum sufficient thinking level and identifies the strongest candidate for the task. The router takes the lowest supported thinking level covering 96.7% of Jev's estimated distribution, then selects the cheapest qualifying model. If none qualifies, it uses Jev's strongest candidate at its highest supported level. These estimates are advisory, not verified success rates.

Cost ranking uses Pi's catalog prices, an approximate input-token count, and a 4,096-token output estimate. It accounts for catalog pricing tiers, but not cache hits, reasoning-token volume, or subscription billing. Equal estimates preserve scope order.

The chosen pair stays fixed through tool follow-ups and retries while it remains available and scoped. Each new user prompt is reassessed. Router state follows Pi's session branches. Direct requests, such as compaction, reuse the previous pair when possible.

## Context and failures

**Routing sends conversation text to the selected Jev provider**, even when the generation model is local. Jev receives up to 16,000 characters of the latest prompt, 8,000 of system instructions/tool definitions, and 24,000 of recent conversation/tool results. Long entries retain their beginning and end. The generation model still receives Pi's normal context.

Image data and hidden reasoning are not sent to Jev. Requests containing images are restricted to image-capable scoped models, but Jev does not inspect the images themselves.

Missing scope, missing Jev authentication, classifier failures, or invalid fallback decisions stop the request with an error rather than silently choosing an arbitrary model. Routing has a 10-second deadline and respects cancellation.

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
