# Logits × local Codex test report — 2026-10-20

The three Logits models passed the tested text coding workflow. Native GPT remains the default (`gpt-6-astra`, existing ChatGPT login). The main `~/.codex/config.toml` was not edited. This is a functional smoke test, not a model-quality or inference-performance benchmark.

| Model | Direct authenticated API | Streaming Responses bridge | Create/edit/run code | Same-session follow-up | Installed profile + service |
|---|---|---|---|---|---|
| `deepseek-v4.1-flash` | PASS | PASS | PASS | PASS in v3 | PASS |
| `kimi-k3` | PASS | PASS | PASS | PASS in v2 and v3 | PASS |
| `muse-spark-open` | PASS | PASS | PASS | PASS in v3 | PASS |

The follow-up test starts a session, provides a random test tag without writing it to a workspace file, then resumes the same session ID. The second turn edits Python code using the remembered tag and executes assertions. An independent Python process checks the resulting file. Session metadata records `model_provider = logits-local` and the intended model on both turns. Each v3 run made two successful shell calls, exited zero and returned the exact requested markers.

## Failures and warnings retained

1. **Follow-up v1: test harness routing failure, all three failed.** Combining `--ignore-user-config` with `--profile` skipped the profile. Requests went to OpenAI without credentials and received HTTP 401; no tool ran. Removed `--ignore-user-config` when testing profiles. Each isolated test home contains only the intended Logits profile and no GPT credential. The failed logs remain in `codex-followup-2026-10-20-v1/`.
2. **Follow-up v2: DeepSeek exact-output failure.** Code execution, same-session continuation and the independent file assertion passed, but the model added explanatory text around the required final markers. Retained as a strict test failure; no result was relabeled a pass.
3. **Follow-up v2: Muse instruction interpretation failure.** The initial prompt prohibited writing a test tag; the second prompt asked to write it. Muse treated the first instruction as still binding and refused, so the required file assertion failed. v3 explicitly limits the prohibition to the first turn and authorizes writing in the second. All three models passed v3. This is a test clarification, not a change to Muse or proof that instruction-following issues are eliminated.
4. **Installed-profile v1: test harness config-key error, all three failed before network use.** An unnecessarily quoted key in a `-c mcp_servers...enabled=false` override created an invalid MCP entry. Corrected the invocation to use the existing simple key `node_repl`; v2 passed all three. No existing MCP settings were edited.
5. **Initial CLI smoke warning.** Codex logged `state db discrepancy ... falling_back` twice per run while using the main home with ephemeral sessions. All code assertions still passed. Isolated follow-up homes produced no such warning. No database repair was attempted or required for these tests.

Raw command arrays, stdout JSONL, stderr, final text and independent verification results are preserved under `/Users/ryanli/.config/logits-codex/`. No API key is included in this report or the test command arguments.

## Use alongside native GPT

```sh
codex                                  # Existing native GPT login and default
codex --profile logits-deepseek         # deepseek-v4.1-flash
codex --profile logits-kimi             # kimi-k3
codex --profile logits-muse             # muse-spark-open
```

For the tested minimal worker setup, add `--disable apps --disable plugins --disable multi_agent`; the profile disables native web search. These command flags affect that process only. Profile files are `/Users/ryanli/.codex/logits-{deepseek,kimi,muse}.config.toml`. Each defines its own provider and command-backed local bridge authentication. Do not add `--ignore-user-config` to profile-based invocations. The current desktop conversation and default GPT selection are unchanged. Desktop model-picker switching between providers has not been tested.

## Runtime and supported scope

- CLI: 0.163.0-alpha.10.2; LiteLLM: 1.106.0.
- Logits documented Responses-to-Chat-Completions adapter, with its text conversion hook extended to the three catalog model IDs.
- Adapter listens only on `127.0.0.1:4000`; upstream is `https://api.logits.com/v1`.
- macOS LaunchAgent: `com.logits.codex-adapter`; starts on login and is currently running. Credentials stay in private local files (0600) and are read by the launcher, not embedded in TOML.
- Tested with reasoning effort `none`, 32,000-token local context limit, 24,000-token compaction threshold. Muse can still emit upstream reasoning; text and terminal streaming events were received correctly.
- This adapter qualification covers text and shell-based editing. Images, native free-form patch tools, the desktop model picker, native web search, plugins, long-context compaction and prolonged reliability have not been qualified. Automatic fallback to GPT is not configured.

## Evidence

- Direct API: `connectivity-2026-10-20.json`, `connectivity-muse-2026-10-20.json`.
- Streaming: `responses-stream-test-2026-10-20.json`.
- Real coding: `codex-smoke-2026-10-20-v1/summary.json`.
- Follow-up: all `codex-followup-2026-10-20-v{1,2,3}/summary.json` retained.
- Final installed profiles: `installed-profile-smoke-2026-10-20-v2/summary.json`; failed v1 retained.
- Installation: `profile-installation-2026-10-20.json`, `service-installation-2026-10-20.json`.

Paths above are relative to `/Users/ryanli/.config/logits-codex/`. Test workspaces are under `/Users/ryanli/code/logits-codex-smoke/`. Preserve existing run directories; reruns must use a fresh output directory.

## Service control

```sh
launchctl print gui/$(id -u)/com.logits.codex-adapter
launchctl bootout gui/$(id -u)/com.logits.codex-adapter
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.logits.codex-adapter.plist
```

Stopping this adapter affects only the three Logits profiles. Native GPT does not depend on it. To remove the integration, stop the LaunchAgent and remove its plist and the three Logits profile files; keep the main GPT configuration and auth intact.

Sources: [Logits Codex setup](https://logits.com/docs/codex), [OpenAI Docs: configuration profiles](https://learn.chatgpt.com/docs/config-file/config-advanced), [OpenAI Docs: gateway authentication](https://learn.chatgpt.com/docs/enterprise/connect-to-a-gateway).

## Private repository research handoff

The three private-repository research worker processes were NOT started. Automatic approval review rejected transmission of private minivllm source/docs/experiment data to Logits without explicit payload authorization. The concrete dispatch script is prepared; user clarification is pending. Public connectivity and synthetic coding tests do not require sending minivllm source. Existing GPT child workers report usage limits; no successful new GPT worker is claimed.
