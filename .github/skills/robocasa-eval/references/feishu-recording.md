---
# Reference: recording eval results into the Feishu wiki doc
---

# Recording results in the Feishu doc

Target doc (default): **pi05 test** — <https://qcnsjukqtk5w.feishu.cn/wiki/VoVMwDMVgi0upAkdiRecjz6SnIS>
(相域未来智能科技 wiki). Results are appended as a new section at the very end.

## Preconditions

* The page is login-gated: `fetch_webpage` gets redirected to `accounts.feishu.cn`. It only works if the
  user is logged in **in the VS Code integrated browser** (that browser carries their session). Ask the
  user to log in and say so, then re-open/reload the page. Do not try to log in for them.
* Confirm with the user before writing to the shared doc. Every write is undoable with `Ctrl+Z`, but it
  is a team document.

## Reading the document

Feishu virtualizes long docs — only the blocks near the viewport exist in the DOM. The scroll container
is `.bear-web-x-container` (class also contains `bear-web-x-container catalogue-opened docx-in-wiki`).

Scroll in steps and collect leaf blocks, then order them by absolute offset:

```js
const root = document.querySelector('.bear-web-x-container');
const seen = new Map();
for (let p = 0; p <= root.scrollHeight; p += 400) {
  root.scrollTop = p; await new Promise(r => setTimeout(r, 150));
  const cRect = root.getBoundingClientRect();
  for (const b of root.querySelectorAll('[data-block-id]')) {
    if (b.querySelector('[data-block-id]')) continue;            // leaves only
    const id = b.getAttribute('data-block-id');
    const abs = Math.round(b.getBoundingClientRect().top - cRect.top + root.scrollTop);
    seen.set(id, { abs, text: (b.innerText || '').trim() });
  }
}
```

Block class names encode the type: `docx-heading1-block`, `docx-heading2-block`, `docx-heading3-block`,
`docx-text-block`, `docx-sheet-block`, `docx-back_ref_list-block`. Heading levels are what give the
document its H1/H2/H3 structure — use them to attribute numbers to the right experiment group. The left
outline panel gives the same hierarchy.

**Gotcha:** empty Feishu paragraphs contain `\u200b` (zero-width space), which `String.trim()` does
**not** strip. Normalize before testing emptiness:

```js
const norm = s => String(s).replace(/[\u200b\u200c\u200d\ufeff]/g, '').trim();
```

## Appending a table

1. **Focus the page** — this browser does not implement `Browser.grantPermissions`
   (`Protocol error: Method not found`), and `navigator.clipboard.writeText` fails with
   `NotAllowedError: Document is not focused` unless the page is focused:
   `await page.bringToFront(); await page.mouse.move(700, 400);`

2. **Put the cursor in a fresh trailing paragraph.** Find the last leaf block that is *not*
   `docx-back_ref_list-block` (`"This document hasn't been mentioned by others yet."`), click it,
   press `End`, press `Enter`, then click the new empty block. (Clicking an existing paragraph without
   pressing Enter first would split the user's paragraph.)

3. **Write both clipboard flavours** — `text/html` gives Feishu a real table, `text/plain` is the
   fallback:

   ```js
   await navigator.clipboard.write([new ClipboardItem({
     'text/html':  new Blob([htmlTable],  { type: 'text/html' }),
     'text/plain': new Blob([tsvPlaintext], { type: 'text/plain' })
   })]);
   ```

4. **Paste**: `await page.keyboard.press('Control+v')`, then wait ~5 s.

5. **Verify** by re-scanning `[data-block-id]` blocks at the end of the doc (same scroll routine as
   above). Expect, in order: `docx-heading2-block` (section title) → `docx-text-block` (notes) →
   `docx-heading3-block` per table → `docx-sheet-block` per table.

## Expected (and surprising) behaviour

* A pasted HTML table becomes a **`blockType=sheet` embedded spreadsheet** (`div[data-sheet-element=
  "embeddedSheetContainer"]`, `.spreadsheet-wrap.embed-spreadsheet-wrap`, canvas-rendered) rather than a
  native `docx-table-block`. That is Feishu's automatic conversion for larger tables, not a failure —
  the data is intact and editable. Mention it when reporting to the user.
* Sanity-check the row count via the rendered block height (~45 px/row at default zoom): a 46-row table
  measured ≈ 2080 px.
* A page reload is the strongest check that the content persisted (`Saved to cloud` in the header).

## If a paste goes wrong

Press `Control+Z` (possibly twice: once for the paste, once for the `Enter`) and re-verify. Do not leave
the doc in a half-pasted state; tell the user what happened.

## Sourcing the table content

Build the table from the run's `summary.md` (see `scripts/run_eval.py`) and prepend the setup block:
model path, training config, inference config, and eval parameters (trials, envs, seed, replan,
save-video). One section per eval run:

```
### <exp_name> / checkpoint-<step>            (H3 heading)
模型: <ckpt path>
训练 config: <train config>  推理 config: <serve config>
评测参数: 100 trials × <num_envs> envs, seed=<seed>, replan=<k>, save_video=<bool>
[结果表]
```
