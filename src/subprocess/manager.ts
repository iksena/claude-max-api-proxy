/**
 * Claude Code CLI Subprocess Manager
 *
 * Handles spawning, managing, and parsing output from Claude CLI subprocesses.
 * Uses spawn() instead of exec() to prevent shell injection vulnerabilities.
 */

import { spawn, spawnSync, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import fs from "fs/promises";
import { readFileSync, mkdtempSync, rmSync, appendFileSync } from "fs";
import os from "os";
import path from "path";
import type {
  ClaudeCliMessage,
  ClaudeCliAssistant,
  ClaudeCliResult,
  ClaudeCliStreamEvent,
} from "../types/claude-cli.js";
import {
  isAssistantMessage,
  isResultMessage,
  isContentDelta,
  isTextBlockStart,
  isToolUseBlockStart,
  isInputJsonDelta,
  isContentBlockStop,
  isSystemInit,
} from "../types/claude-cli.js";
import type { ClaudeModel } from "../adapter/openai-to-cli.js";

export interface SubprocessOptions {
  model: ClaudeModel;
  sessionId?: string;
  /** Resume an existing persisted session (sessionId) instead of creating a new one */
  resume?: boolean;
  cwd?: string;
  timeout?: number;
}

export interface SubprocessEvents {
  message: (msg: ClaudeCliMessage) => void;
  assistant: (msg: ClaudeCliAssistant) => void;
  result: (result: ClaudeCliResult) => void;
  error: (error: Error) => void;
  close: (code: number | null) => void;
  raw: (line: string) => void;
}

const DEFAULT_TIMEOUT = 900000; // 15 minutes

/**
 * Locked-down (text-only) mode is the DEFAULT in this fork.
 *
 * Upstream runs the Claude Code CLI with --dangerously-skip-permissions and every
 * built-in tool (Bash, Read, Write, Grep, WebFetch, ...) plus whatever MCP
 * connectors the account has, and tells the model to use them. When this proxy
 * is used as an LLM backend for a benchmark, that lets the model read the host
 * filesystem, run commands and browse the web, i.e. it can see ground truth,
 * other jobs, and run extra validators. Locked mode makes the proxy a plain
 * text-completion endpoint: no tools, no MCP, no settings/hooks, no skills, no
 * session persistence, an empty throw-away working directory, prompts passed
 * through verbatim, and fail-closed verification of the CLI's init event.
 *
 * Set CLAUDE_PROXY_ALLOW_TOOLS=1 to restore the upstream behaviour.
 */
export const TOOLS_ALLOWED = process.env.CLAUDE_PROXY_ALLOW_TOOLS === "1";

/** Optional JSONL audit log: one line per completed request. */
const AUDIT_LOG = process.env.CLAUDE_PROXY_AUDIT_LOG;

/**
 * System prompt appended to Claude CLI to map OpenClaw tool names to Claude Code equivalents.
 * OpenClaw's system prompt references tools like `exec`, `read`, `web_search` etc. that
 * don't exist in Claude Code. This mapping tells the model what to use instead.
 */
const OPENCLAW_TOOL_MAPPING_PROMPT = [
  "## Tool Name Mapping",
  "You are running inside Claude Code CLI, not OpenClaw. The system prompt may reference OpenClaw tool names — map them to your actual tools:",
  "",
  "### Direct tool replacements",
  "- `exec` or `process` → use `Bash` (run shell commands)",
  "- `read` → use `Read` (read file contents)",
  "- `write` → use `Write` (write files)",
  "- `edit` → use `Edit` (edit files)",
  "- `grep` → use `Grep` (search file contents)",
  "- `find` or `ls` → use `Glob` or `Bash(ls ...)`",
  "- `web_search` → use `WebSearch`",
  "- `web_fetch` → use `WebFetch`",
  "- `image` → use `Read` (Claude Code can read images)",
  "",
  "### OpenClaw CLI tools (use via Bash)",
  "These OpenClaw tools are available through the `openclaw` CLI. Use `Bash` to run them:",
  '- `memory_search` → `Bash(openclaw memory search "<query>")` — semantic search across memory files',
  "- `memory_get` → `Read` on the memory file directly, OR `Bash(openclaw memory search \"<query>\")` for discovery",
  '- `message` → `Bash(openclaw message send --to <target> "<text>")` — send messages to channels (Telegram, Discord, etc.)',
  "  - Also: `openclaw message read`, `openclaw message broadcast`, `openclaw message react`, `openclaw message poll`",
  "- `cron` → `Bash(openclaw cron list)`, `Bash(openclaw cron add ...)`, `Bash(openclaw cron status)` — manage scheduled jobs",
  "  - Also: `openclaw cron rm`, `openclaw cron enable`, `openclaw cron disable`, `openclaw cron runs`, `openclaw cron run`, `openclaw cron edit`",
  '- `sessions_list` → `Bash(openclaw agent --local --message "list sessions")` or check session files directly',
  '- `sessions_history` → `Bash(openclaw agent --local --message "show history for session <key>")` or check session files',
  "- `nodes` → `Bash(openclaw nodes status)`, `Bash(openclaw nodes describe <node>)`, `Bash(openclaw nodes invoke --node <id> --command <cmd>)`",
  '  - Also: `openclaw nodes run --node <id> "<shell command>"` for running commands on paired nodes',
  "",
  "### Not available via CLI",
  "- `browser` — requires OpenClaw's dedicated browser server (no CLI equivalent)",
  "- `canvas` — requires paired node with canvas capability; use `openclaw nodes invoke` if a node is available",
  "",
  "### Skills",
  "When a skill says to run a bash/python command, use the `Bash` tool directly.",
  "Skills are located in the `skills/` directory relative to your working directory.",
  "To use a skill: `Read` its SKILL.md file first, then follow the instructions using `Bash`.",
  "Run `openclaw skills list --eligible --json` to see all available skills.",
].join("\n");

/**
 * Resolve the real Claude CLI binary to spawn.
 *
 * On Windows, the global `claude` command is an npm shim (`claude.cmd`) that
 * just execs a bundled `claude.exe`. Running the shim requires `shell: true`,
 * which routes our argv through cmd.exe — and cmd.exe treats characters our
 * appended system prompt legitimately contains (`<`, `>`, `(`, `)`, `&`) as
 * redirection/grouping/chaining operators, corrupting the argument list that
 * follows (notably `--session-id`/`--resume`, which silently stop working).
 * Resolving straight to the `.exe` lets us spawn with `shell: false` and
 * skip cmd.exe entirely. Falls back to the shim (with shell:true) if the
 * `.exe` can't be located.
 */
let resolvedClaudeBin: { bin: string; shell: boolean } | null = null;

function resolveClaudeBin(): { bin: string; shell: boolean } {
  if (process.env.CLAUDE_BIN) {
    // Environment overrides are intentionally not cached. This lets callers
    // temporarily select a binary without contaminating later resolutions.
    return { bin: process.env.CLAUDE_BIN, shell: false };
  }

  if (resolvedClaudeBin) return resolvedClaudeBin;

  if (process.platform === "win32") {
    try {
      const where = spawnSync("where.exe", ["claude"], { encoding: "utf8" });
      const shimPath = (where.stdout || "")
        .split(/\r?\n/)
        .map((p) => p.trim())
        .find((p) => p.toLowerCase().endsWith(".cmd"));

      if (shimPath) {
        const shimDir = path.dirname(shimPath);
        const shimContent = readFileSync(shimPath, "utf8");
        const match = shimContent.match(/"%dp0%\\(.+?\.exe)"/i);
        if (match) {
          const exePath = path.join(shimDir, match[1]);
          resolvedClaudeBin = { bin: exePath, shell: false };
          return resolvedClaudeBin;
        }
      }
    } catch {
      // Fall through to shim fallback below
    }

    resolvedClaudeBin = { bin: "claude", shell: true };
    return resolvedClaudeBin;
  }

  resolvedClaudeBin = { bin: "claude", shell: false };
  return resolvedClaudeBin;
}

/**
 * Kill a process and its full descendant tree.
 *
 * `ChildProcess.kill()` only signals the direct child. On Windows this is
 * insufficient because the Claude CLI spawns its own subprocesses (e.g. for
 * Bash tool calls) that aren't part of a job object — killing just the
 * parent leaves them running in the background even after a client
 * disconnect or timeout. `taskkill /T` walks the whole tree instead.
 */
function killProcessTree(
  child: ChildProcess,
  signal: NodeJS.Signals = "SIGTERM"
): boolean {
  const pid = child.pid;
  if (!pid) return false;

  if (process.platform === "win32") {
    const taskkill = process.env.SystemRoot
      ? path.join(process.env.SystemRoot, "System32", "taskkill.exe")
      : "taskkill.exe";
    const result = spawnSync(taskkill, ["/PID", String(pid), "/T", "/F"], {
      stdio: "ignore",
      windowsHide: true,
    });
    if (!result.error && result.status === 0) {
      return true;
    }

    // If taskkill could not be started or rejected the request, still make a
    // best-effort attempt to stop the managed root process. This must not be
    // reported as a successful tree termination: descendants may still be
    // running, and callers must remain able to retry.
    try {
      child.kill(signal);
    } catch {}
    return false;
  }

  try {
    return child.kill(signal);
  } catch {
    // Process may have already exited
    return false;
  }
}

export class ClaudeSubprocess extends EventEmitter {
  private process: ChildProcess | null = null;
  private buffer: string = "";
  private timeoutId: NodeJS.Timeout | null = null;
  private isKilled: boolean = false;
  private tmpCwd: string | null = null;
  private sawTools: boolean = false;
  private lastInit: { tools: unknown[]; mcp: unknown[] } | null = null;

  /**
   * Start the Claude CLI subprocess with the given prompt
   */
  async start(prompt: string, options: SubprocessOptions): Promise<void> {
    const args = this.buildArgs(options);
    const timeout = options.timeout || DEFAULT_TIMEOUT;
    // Locked mode: run in a fresh empty directory so even a misconfigured CLI
    // would have nothing to read.
    if (!TOOLS_ALLOWED) {
      this.tmpCwd = mkdtempSync(path.join(os.tmpdir(), "claude-proxy-"));
    }
    if (process.env.DEBUG_SUBPROCESS) {
      console.error(`[Subprocess] args: ${JSON.stringify(args)}`);
      console.error(`[Subprocess] prompt: ${prompt.slice(0, 200)}`);
    }

    return new Promise((resolve, reject) => {
      try {
        // Use spawn() for security - no shell interpretation
        const { bin, shell } = resolveClaudeBin();
        this.process = spawn(bin, args, {
          cwd: this.tmpCwd || options.cwd || process.cwd(),
          env: Object.fromEntries(
            Object.entries(process.env).filter(([k]) => k !== "CLAUDECODE")
          ),
          stdio: ["pipe", "pipe", "pipe"],
          shell,
        });

        this.armTimeout(timeout);

        // Handle spawn errors (e.g., claude not found)
        this.process.on("error", (err) => {
          this.clearTimeout();
          if (err.message.includes("ENOENT")) {
            reject(
              new Error(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
              )
            );
          } else {
            reject(err);
          }
        });

        // Pass prompt via stdin to avoid E2BIG on large inputs
        this.process.stdin?.write(prompt);
        this.process.stdin?.end();

        if (process.env.DEBUG_SUBPROCESS) {
          console.error(`[Subprocess] Process spawned with PID: ${this.process.pid}`);
        }

        // Parse JSON stream from stdout
        this.process.stdout?.on("data", (chunk: Buffer) => {
          const data = chunk.toString();
          if (process.env.DEBUG_SUBPROCESS) {
            console.error(`[Subprocess] Received ${data.length} bytes of stdout`);
          }
          this.buffer += data;
          this.processBuffer();
        });

        // Capture stderr for debugging
        this.process.stderr?.on("data", (chunk: Buffer) => {
          const errorText = chunk.toString().trim();
          if (errorText) {
            // Don't emit as error unless it's actually an error
            // Claude CLI may write debug info to stderr
            if (process.env.DEBUG_SUBPROCESS) {
              console.error("[Subprocess stderr]:", errorText.slice(0, 200));
            }
          }
        });

        // Handle process close
        this.process.on("close", (code) => {
          if (process.env.DEBUG_SUBPROCESS) {
            console.error(`[Subprocess] Process closed with code: ${code}`);
          }
          this.clearTimeout();
          // Process any remaining buffer
          if (this.buffer.trim()) {
            this.processBuffer();
          }
          if (this.tmpCwd) {
            try { rmSync(this.tmpCwd, { recursive: true, force: true }); } catch { /* ignore */ }
            this.tmpCwd = null;
          }
          this.emit("close", code);
        });

        // Resolve immediately since we're streaming
        resolve();
      } catch (err) {
        this.clearTimeout();
        reject(err);
      }
    });
  }

  /**
   * Build CLI arguments array
   */
  private buildArgs(options: SubprocessOptions): string[] {
    if (!TOOLS_ALLOWED) {
      // Locked, text-only mode (default). See TOOLS_ALLOWED above.
      return [
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model",
        options.model,
        "--tools",
        "", // no built-in tools at all
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}', // ignore every MCP server / claude.ai connector
        "--setting-sources",
        "", // ignore user/project/local settings, hooks, CLAUDE.md
        "--disable-slash-commands", // no skills
        "--no-session-persistence", // never resume or leak context between requests
        // Prompt is passed via stdin (avoids E2BIG on large inputs)
      ];
    }

    const args = [
      "--print", // Non-interactive mode
      "--dangerously-skip-permissions", // Skip permission prompts
      "--output-format",
      "stream-json", // JSON streaming output
      "--verbose", // Required for stream-json
      "--include-partial-messages", // Enable streaming chunks
      "--model",
      options.model, // Model alias (opus/sonnet/haiku)
      "--append-system-prompt",
      OPENCLAW_TOOL_MAPPING_PROMPT,
      // Prompt is passed via stdin (avoids E2BIG on large inputs)
    ];

    if (options.sessionId && options.resume) {
      // Continue a previously persisted session — avoids replaying full history
      args.push("--resume", options.sessionId);
    } else if (options.sessionId) {
      // First turn for this session key — create it under a known ID so we
      // can --resume it on subsequent turns
      args.push("--session-id", options.sessionId);
    } else {
      // No stable session key (e.g. request.user missing) — don't leave
      // orphaned session files behind
      args.push("--no-session-persistence");
    }

    return args;
  }

  /** Kill the CLI and report a policy violation (locked mode only). */
  private failClosed(reason: string): void {
    if (this.isKilled) return;
    if (this.process) {
      this.isKilled = killProcessTree(this.process, "SIGKILL");
    }
    this.emit("error", new Error(`[locked mode] ${reason}`));
  }

  /** Append one audit record per finished request (CLAUDE_PROXY_AUDIT_LOG). */
  private audit(result: ClaudeCliResult): void {
    if (!AUDIT_LOG) return;
    try {
      const usage = result.modelUsage || {};
      appendFileSync(
        AUDIT_LOG,
        JSON.stringify({
          ts: new Date().toISOString(),
          locked: !TOOLS_ALLOWED,
          num_turns: result.num_turns,
          tools_at_init: this.lastInit ? this.lastInit.tools : null,
          mcp_at_init: this.lastInit ? this.lastInit.mcp : null,
          tool_use_seen: this.sawTools,
          model_usage: Object.fromEntries(
            Object.entries(usage).map(([k, v]) => [
              k,
              { in: (v as any).inputTokens, out: (v as any).outputTokens },
            ])
          ),
        }) + "\n"
      );
    } catch {
      /* auditing must never break a request */
    }
  }

  /**
   * Process the buffer and emit parsed messages
   */
  private processBuffer(): void {
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop() || ""; // Keep incomplete line

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        const message: ClaudeCliMessage = JSON.parse(trimmed);

        if (isSystemInit(message)) {
          this.lastInit = {
            tools: message.tools || [],
            mcp: message.mcp_servers || [],
          };
          if (!TOOLS_ALLOWED && (this.lastInit.tools.length > 0 || this.lastInit.mcp.length > 0)) {
            this.failClosed(
              `CLI exposed tools=${JSON.stringify(this.lastInit.tools)} mcp=${JSON.stringify(this.lastInit.mcp)}`
            );
            return;
          }
        }
        if (isToolUseBlockStart(message)) {
          this.sawTools = true;
          if (!TOOLS_ALLOWED) {
            this.failClosed("model attempted a tool call");
            return;
          }
        }
        if (isResultMessage(message)) {
          this.audit(message);
        }

        this.emit("message", message);

        if (isTextBlockStart(message)) {
          // Emit when a new text content block starts (for inserting separators)
          this.emit("text_block_start", message as ClaudeCliStreamEvent);
        }

        if (isToolUseBlockStart(message)) {
          this.emit("tool_use_start", message as ClaudeCliStreamEvent);
        }

        if (isInputJsonDelta(message)) {
          this.emit("input_json_delta", message as ClaudeCliStreamEvent);
        }

        if (isContentBlockStop(message)) {
          this.emit("content_block_stop", message as ClaudeCliStreamEvent);
        }

        if (isContentDelta(message)) {
          // Emit content delta for streaming (text_delta only)
          this.emit("content_delta", message as ClaudeCliStreamEvent);
        } else if (isAssistantMessage(message)) {
          this.emit("assistant", message);
        } else if (isResultMessage(message)) {
          this.emit("result", message);
        }
      } catch {
        // Non-JSON output, emit as raw
        this.emit("raw", trimmed);
      }
    }
  }

  /**
   * Clear the timeout timer
   */
  private clearTimeout(): void {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }

  /**
   * Kill the subprocess
   */
  kill(signal: NodeJS.Signals = "SIGTERM"): void {
    if (!this.isKilled && this.process) {
      this.clearTimeout();
      this.isKilled = killProcessTree(this.process, signal);
    }
  }

  /**
   * Arm the request timeout. Kept narrow so tests can deterministically re-arm
   * the production timeout behavior after their fixture process tree is ready.
   */
  private armTimeout(timeout: number): void {
    this.clearTimeout();
    this.timeoutId = setTimeout(() => {
      if (!this.isKilled) {
        if (this.process) {
          this.isKilled = killProcessTree(this.process, "SIGTERM");
        }
        this.emit("error", new Error(`Request timed out after ${timeout}ms`));
      }
    }, timeout);
  }

  /**
   * Check if the process is still running
   */
  isRunning(): boolean {
    return this.process !== null && !this.isKilled && this.process.exitCode === null;
  }
}

/**
 * Verify that Claude CLI is installed and accessible
 */
export async function verifyClaude(): Promise<{ ok: boolean; error?: string; version?: string }> {
  return new Promise((resolve) => {
    const { bin, shell } = resolveClaudeBin();
    const proc = spawn(bin, ["--version"], { stdio: "pipe", shell });
    let output = "";

    proc.stdout?.on("data", (chunk: Buffer) => {
      output += chunk.toString();
    });

    proc.on("error", () => {
      resolve({
        ok: false,
        error:
          "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
      });
    });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve({ ok: true, version: output.trim() });
      } else {
        resolve({
          ok: false,
          error: "Claude CLI returned non-zero exit code",
        });
      }
    });
  });
}

/**
 * Check if Claude CLI is authenticated
 *
 * Claude Code stores credentials in the OS keychain, not a file.
 * We verify authentication by checking if we can call the CLI successfully.
 * If the CLI is installed, it typically has valid credentials from `claude auth login`.
 */
export async function verifyAuth(): Promise<{ ok: boolean; error?: string }> {
  // If Claude CLI is installed and the user has run `claude auth login`,
  // credentials are stored in the OS keychain and will be used automatically.
  // We can't easily check the keychain, so we'll just return true if the CLI exists.
  // Authentication errors will surface when making actual API calls.
  return { ok: true };
}
