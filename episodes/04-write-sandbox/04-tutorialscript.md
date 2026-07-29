# Episode 4 — Writing Code: `write_file` + the Workspace Sandbox (Tutorial Video Script)

## Overview
**Length:** ~10 minutes  
**Goal:** Show viewers how to give their agent the ability to write files using Deep Agents' built-in `FilesystemBackend`, while reinforcing the workspace sandbox discipline. The episode adds a `resolve_in_workspace()` helper that custom tools in later episodes will use to share the backend's security model.

---

## Scene 1: Hook & Why File Writing Matters (0:00–1:30)

**On-screen:** Terminal showing an agent that can read files but cannot create new ones — it's limited to analysis only. Then with `write_file`, it becomes a productive coding assistant.

> **Host:** "So far, our agent can explore the filesystem and understand code — but it can't CREATE anything. It's like having a brilliant architect who can analyze blueprints but can't actually build."
>
> *(Show agent unable to create files without write_file)*

**Key points:**
- Reading-only agents are limited to analysis tasks.
- Real coding requires writing new files and modifying existing ones.
- `write_file` is built into Deep Agents' FilesystemBackend — no custom code needed for the tool itself.

---

## Scene 2: The Sandbox Story (1:30–4:00)

**On-screen:** Diagram showing the workspace sandbox with virtual_mode=True, then code showing how paths are resolved and confined.

> **Host:** "Before we write any files, let's understand the sandbox. Deep Agents' `FilesystemBackend` has a `virtual_mode=True` parameter that confines all file operations to a designated workspace directory."
>
> ```python
> backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
> return create_deep_agent(
>     model=get_model(), tools=[],  # write_file/read_file/ls come from the backend
>     system_prompt="...", backend=backend, checkpointer=InMemorySaver(),
> )
> ```

**Key points:**
- `virtual_mode=True` is what makes the sandbox work — it intercepts all file operations.
- The agent gets built-in tools: `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`.
- We pass `tools=[]` because these come from the backend automatically.

---

## Scene 3: resolve_in_workspace — Sharing Sandbox Discipline (4:00–6:30)

**On-screen:** Code showing `resolve_in_workspace()` and `PathEscapeError`, then a demo of path escape attempts being blocked.

> **Host:** "While the backend handles its own tools, our CUSTOM tools in later episodes need to share this sandbox discipline. That's why we add `resolve_in_workspace` — it mirrors what the backend does internally."
>
> ```python
> class PathEscapeError(PermissionError):
>     """Raised when a resolved path leaves the workspace sandbox."""
> 
> def resolve_in_workspace(path: str | Path) -> Path:
>     """Resolve `path` relative to CODEIT_WORKDIR, refusing to escape.
>     
>     Blocks `../`, `~`, and absolute paths outside the workspace root.
>     Follows symlinks via .resolve(), then checks the result is still inside.
>     """
>     root = workspace_root()
>     target = (root / path).resolve()
>     try:
>         target.relative_to(root)  # raises ValueError if outside root
>     except ValueError:
>         raise PathEscapeError(
>             f"Path {path!r} resolves outside the workspace ({root}). Refusing."
>         )
>     return target
> ```

**Live demo:** Show path escape attempts being blocked:
```python
resolve_in_workspace("../../../etc/passwd")  # → PathEscapeError!
resolve_in_workspace("/etc/shadow")          # → PathEscapeError!
resolve_in_workspace("main.py")              # → /workspace/main.py (OK)
```

**Key points:**
- `PathEscapeError` is a `PermissionError` subclass — caught by generic handlers but identifiable.
- `.resolve()` follows symlinks, so symlink-based escapes are also blocked.
- This helper will be used by Episodes 5 (shell), 8 (edit wrapper), and 9 (repo map).

---

## Scene 4: Live Demo — Agent Writes Real Code (6:30–8:30)

**On-screen:** Terminal session showing the agent creating a Python file, then verifying it exists on disk.

> **Host:** "Let's see this in action! We'll ask our agent to create a real Python file."
>
> ```bash
> CODEIT_WORKDIR=./workspace python 04-write-sandbox.py \
>     --yolo "Create main.py with a FastAPI app: GET /hello returns {'msg':'hello'}"
> ```

**What viewers will see:**
1. Agent calls `write_file` to create `main.py`.
2. The file is written inside the workspace sandbox.
3. At the end, we verify the file exists on disk and show its size: `(wrote /workspace/main.py — 112 bytes)`

**Key points:**
- This is the "agent wrote real code" moment — viewers can open the file themselves.
- The `--yolo` flag auto-approves (no approval gate yet in this episode).
- File operations are confined to `CODEIT_WORKDIR`.

---

## Scene 5: What virtual_mode Actually Protects Against (8:30–9:30)

**On-screen:** Text explaining what the sandbox protects against and its limitations.

> **Host:** "Let's be clear about what `virtual_mode=True` actually does — and doesn't do."
>
> *(Show protection matrix)*
> - ✅ Blocks path traversal (`../../../etc/passwd`)
> - ✅ Confines all file operations to workspace root
> - ❌ Does NOT prevent the agent from running destructive shell commands (Ep 5)
> - ❌ Does NOT prevent `rm -rf` inside the workspace

**Key points:**
- The sandbox protects against path escapes but not destructive actions within it.
- Episode 6 adds the approval gate for dangerous operations.
- Even with sandboxing, `--yolo` mode is still risky — never run unsandboxed on a real repo.

---

## Scene 6: Wrap-up & What's Next (9:30–10:00)

**On-screen:** Summary showing all write sandbox capabilities and preview of Episode 5.

> **Host:** "In this episode, we've given our agent the ability to create files — but with proper sandboxing discipline."
>
> - **`FilesystemBackend(virtual_mode=True)`** — provides `write_file`, `read_file`, `ls`, etc. as built-in tools.
> - **`resolve_in_workspace()`** — helper for custom tools to share sandbox discipline (used in Episodes 5, 8, 9).
> - **`PathEscapeError`** — raised when paths try to escape the workspace.

**Key takeaways:**
1. `write_file` is built-in via FilesystemBackend — no need to implement it ourselves.
2. The sandbox confines all file operations to CODEIT_WORKDIR using path resolution + relative_to checks.
3. Custom tools must use `resolve_in_workspace()` to maintain the same security discipline.
4. Symlinks are followed by `.resolve()`, so symlink-based escapes are also blocked.
5. Sandboxing protects against path traversal but NOT destructive actions — that's Episode 6.