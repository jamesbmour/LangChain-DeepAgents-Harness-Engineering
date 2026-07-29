# Episode 8 — Surgical Edits: `edit_file` + the Fuzzy Fallback Wrapper (Tutorial Video Script)

## Overview
**Length:** ~13 minutes  
**Goal:** Show viewers how to graduate from whole-file rewrites (`write_file`) to search-replace edits — the industry-standard reliable edit format used by Aider, Codex, and Cline. The episode adds `edit_file_safe` with a three-tier fallback: exact match → fuzzy match (≥90%) → unified diff for self-correction.

---

## Scene 1: Hook & Why Search-Replace Matters (0:00–1:30)

**On-screen:** Terminal showing an agent rewriting an entire file just to change one line, then the same task done with surgical edits.

> **Host:** "In earlier episodes, our agent used `write_file` — it rewrites the ENTIRE file content every time. That works for small files, but what happens when you're editing a 500-line Python module and just need to change one function?"
>
> *(Show whole-file rewrite vs surgical edit side by side)*

**Key points:**
- Whole-file rewrites are wasteful — they send the entire file content back.
- Search-replace is the industry standard used by Aider, Codex, Cline, and Claude Code.
- It's more reliable because it targets specific text rather than rewriting everything.

---

## Scene 2: The edit_file_safe Tool (1:30–4:30)

**On-screen:** Code editor showing the `edit_file_safe` function with its three-tier fallback logic, then a live demo of each tier in action.

> **Host:** "Our solution is `edit_file_safe` — a custom tool that tries exact match first, falls back to fuzzy matching at 90% similarity, and on total failure returns a unified diff so the model can self-correct."
>
> ```python
> @tool
> def edit_file_safe(path: str, search: str, replace: str) -> str:
>     """Edit a file by replacing `search` text with `replace` text (first occurrence).
>     
>     For targeted changes to existing files — never rewrite a whole file.
>     path: file relative to workspace (e.g. 'main.py').
>     search: exact text to find; include enough context to be unique.
>     replace: new text to substitute.
>     
>     On exact match: applies. On near-miss (>=90% similar): applies with a note.
>     On no match: returns a unified diff; re-read the file and retry.
>     """
> ```

**Key points:**
- The docstring explains all three outcomes so the model learns the contract.
- Uses `search`/`replace` format — same as Aider, Codex, Cline.
- Goes through the approval gate from Episode 6 (`interrupt_on={"edit_file_safe": True}`).

---

## Scene 3: Tier 1 — Exact Match (4:30–5:30)

**On-screen:** Code showing the exact match path and a demo of it working on a simple edit.

> **Host:** "Tier one is straightforward — if `search` appears exactly in the file, we replace the first occurrence."
>
> ```python
> if search not in text:
>     # Fall through to fuzzy matching...
> 
> new_text = text.replace(search, replace, 1)  # first occurrence only
> _resolve_in_workspace(path).write_text(new_text, encoding="utf-8")
> return f"Applied exact match edit to {path}."
> ```

**Key points:**
- `replace(..., 1)` ensures we only change the FIRST occurrence.
- Uses `_resolve_in_workspace` (from Episode 4) for sandbox safety — can't escape workspace.
- Returns a confirmation string that appears in the streaming output.

---

## Scene 4: Tier 2 — Fuzzy Match with difflib (5:30–8:00)

**On-screen:** Code showing `_fuzzy_find()` function, then a demo where exact match fails but fuzzy matching succeeds.

> **Host:** "Tier two handles the reality that small models often produce slightly wrong search strings — maybe they miss a character or have different whitespace."
>
> ```python
> def _fuzzy_find(text: str, search: str, threshold: float = 0.90):
>     """Find the substring of `text` most similar to `search`, above `threshold`."""
>     if not search or len(text) > 50_000:
>         return None, 0.0
>     best, best_score, n = None, 0.0, len(search)
>     for i in range(0, max(0, len(text) - n + 1)):
>         score = difflib.SequenceMatcher(None, search, text[i:i + n]).ratio()
>         if score > best_score:
>             best, best_score = text[i:i + n], score
>             if best_score == 1.0:
>                 break
>     return (best, best_score) if best_score >= threshold else (None, best_score)
> ```

**Live demo:** Show a case where the model's search string is slightly off — maybe it has different indentation or missed a character. The fuzzy matcher finds the closest match at 92% similarity and applies it with a note: "Applied fuzzy match (similarity 0.92). Review the result."

**Key points:**
- O(n*m) scan through all possible positions in the text — fine for teaching, Aider uses more sophisticated matching.
- `threshold=0.90` means we need at least 90% similarity to apply.
- Caps file size at 50k chars to prevent performance issues on huge files.

---

## Scene 5: Tier 3 — Unified Diff for Self-Correction (8:00–10:00)

**On-screen:** Code showing the unified diff fallback, then a demo where both exact and fuzzy matching fail.

> **Host:** "Tier three is our safety net — when neither exact nor fuzzy match works, we return a unified diff so the model can see what went wrong and self-correct."
>
> ```python
> # No match — return a unified diff so the model sees the current state.
> diff = difflib.unified_diff(
>     text.splitlines(keepends=True),
>     (text + "\n# --- proposed edit did not apply ---\n").splitlines(keepends=True),
>     fromfile=f"{path} (current)", tofile=f"{path} (NOT applied)", n=3)
> return (f"Could not find the search text in {path} (best similarity {score:.2f}, "
>         f"needed >=0.90). NOT modified. Re-read with read_file, then retry:\n\n"
>         + "".join(diff))
> ```

**Key points:**
- The diff shows the current file state vs what was proposed — but not applied.
- "NOT modified" is critical — we never silently fail or apply wrong edits.
- The model can read this feedback and retry with a corrected search string.

---

## Scene 6: Live Demo — Surgical Edit in Action (10:00–12:00)

**On-screen:** Terminal session showing the agent using `edit_file_safe` to make targeted changes to a file, including handling approval prompts.

> **Host:** "Let's see this working end-to-end! We'll create a Python file with some content and ask the agent to edit it."
>
> ```bash
> CODEIT_WORKDIR=./workspace python 08-surgical-edits.py \
>     --yolo "In big.py, change line 50 to say 'EDITED' instead of 'some content'. Use edit_file_safe."
> ```

**What viewers will see:**
1. Agent reads the file first (using built-in `read_file`).
2. Agent calls `edit_file_safe` with exact search/replace strings.
3. Approval prompt appears — auto-approved via `--yolo`.
4. The edit is applied successfully, confirmed in the output.

---

## Scene 7: Why This Matters for Reliability (12:00–13:00)

**On-screen:** Text summarizing the three-tier approach and its benefits.

> **Host:** "The key insight here isn't just about editing files — it's about building reliable tools that help the model succeed."
>
> - **Exact match** handles 80% of cases where the model gets it right.
> - **Fuzzy matching** catches small mistakes in search strings (whitespace, typos).
> - **Unified diff fallback** gives the model actionable feedback to self-correct.

**Key points:**
- Each tier degrades gracefully — never silently fails or applies wrong edits.
- The approval gate ensures file modifications are reviewed before applying.
- This pattern of "try → fallback → fail-with-feedback" is a general principle for building robust agent tools.

---

## Scene 8: Wrap-up & What's Next (13:00–14:00)

**On-screen:** Summary showing all surgical edit capabilities and preview of Episode 9.

> **Host:** "In this episode, we've graduated from whole-file rewrites to surgical search-replace edits — the same format used by professional coding agents."
>
> - **`edit_file_safe`** with three-tier fallback: exact → fuzzy (≥90%) → unified diff for self-correction.
> - Uses `_resolve_in_workspace` (Ep 4) for sandbox safety.
> - Goes through the approval gate from Episode 6.

**Key takeaways:**
1. Search-replace is more reliable and efficient than whole-file rewrites.
2. The three-tier fallback handles exact matches, near-misses, and total failures gracefully.
3. `difflib.SequenceMatcher` provides a simple fuzzy matching implementation for teaching.
4. Always return actionable feedback (unified diff) when edits fail — never silently fail.