# Obsidian Order Interro-Helper

A local web tool for manually inspecting and comparing prompt transformations.
You type a prompt, tick boxes, and get back every variant side by side — deterministic
encodings instantly, LLM rewrites streamed as they arrive.

It is an **inspection aid**, not an automated attacker. Nothing is sent to a target
system. All outputs are read-only cards you copy from.

---

## Quick start

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

Open `http://localhost:7860`.

Paste your API key and pick a model in the **⚙ Settings** panel — it opens
automatically if no key is configured. The panel also lets you set a separate model
for the Custom Transform field and choose a reasoning effort level for models that
support it (gpt-5.4, o3, o1, etc.).

Alternatively, configure via environment variables before starting (the Settings panel
will reflect these on load):

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_CHAT_MODEL=gpt-5.4          # transforms model, default shown
export OPENAI_REASONING_MODEL=gpt-5.4    # custom transform model, default shown
export OPENAI_CHAT_ENDPOINT=https://...  # change for Azure OpenAI
uvicorn app:app --reload --port 7860
```

Deterministic transforms (encoding, obfuscation, structural) work without an API key.
LLM-based transforms (technique, tactic, translation, custom) require one.

---

## File layout

```
obsidian-order/
├── app.py            FastAPI server: lifecycle, routing, SSE streaming, /config endpoint
├── transforms.py     Transform registry: all 46 transforms + translation + custom
├── requirements.txt  Python dependencies
└── static/
    └── index.html    Single-file frontend: no build step, no CDN
```

---

## Architecture

```
Browser
  │
  ├── GET  /            → index.html (FileResponse)
  ├── GET  /status      → {"llm_available": bool}
  ├── GET  /transforms  → [{id, label, group, requires_llm}, ...]
  │
  └── POST /transform   → SSE stream
        body: {prompt, selected, languages, context}
        events: {id, label, group, result, error}  ← one per transform
                {done: true}                        ← sentinel

        asyncio.Queue + asyncio.create_task for each transform
        → all fire in parallel; results stream as each completes
```

The server never accumulates results before sending. Deterministic transforms (no LLM)
complete in milliseconds and appear immediately. LLM transforms arrive as they finish.
A failed transform emits `{"error": "..."}` on its card without blocking the rest.

---

## `transforms.py` — the registry

### `TransformDef`

```python
@dataclass
class TransformDef:
    id: str           # URL-safe identifier, used as checkbox value and SSE id
    label: str        # Human-readable name shown in the UI
    group: str        # "encoding" | "obfuscation" | "structural" | "technique" | "tactic"
    requires_llm: bool
    _fn: Any          # async callable: (prompt: str, llm, context: str) -> str

    async def apply_async(self, prompt, llm=None, context="") -> str: ...
    def to_dict(self) -> dict: ...  # serialised for GET /transforms
```

`_fn` is always an async function with the signature `(prompt, llm, context) -> str`.
Deterministic transforms ignore `llm` and `context`. LLM transforms use both.

### Adding a deterministic transform

```python
from pyrit.prompt_converter import MyNewConverter

_reg(
    "my-transform",          # id
    "My Transform",          # label
    "structural",            # group
    False,                   # requires_llm
    lambda p, l, ctx="": _conv(MyNewConverter, p, l, kwarg=value),
)
```

`_conv` instantiates the converter class with any kwargs, calls `convert_async`, and
returns `.output_text`. The `l` (llm) and `ctx` args are accepted but ignored.

### Adding an LLM-based transform

```python
_reg(
    "my-llm-transform",
    "My LLM Transform",
    "technique",
    True,
    lambda p, l, ctx="": _llm_rewrite(
        "Rewrite the following as ... Return only the rewritten prompt.",
        p, l, ctx,
    ),
)
```

`_llm_rewrite` sets a system prompt on the shared LLM target, sends the user message,
and returns the response text. It generates a fresh `uuid4()` conversation ID for
every call so parallel invocations never share state.

### Why `_llm_rewrite` calls the target directly

`LLMGenericTextConverter` takes `system_prompt_template: SeedPrompt`, not a plain
string. Passing `system_prompt=str` silently lands in `**kwargs` and is never applied,
so the model receives the prompt with no system context. The direct call pattern
(`set_system_prompt` → `send_prompt_async`) bypasses this and matches how
`GuardedAgentTarget._classify_async` works elsewhere in this repo.

### Context injection

When the user fills the **Context** field, every LLM transform prepends:

```
Target context: <context text>

<technique system prompt>
```

Deterministic transforms receive `context` in their lambda signature but ignore it.
Translations include context in the same way — the translation system prompt becomes
`"Translate to {language}."` with the context prepended.

### Translations

Languages are not in `TRANSFORMS` because they are dynamic (user-selected at runtime).
They are handled separately by `apply_translation(language, prompt, llm, context)`,
which calls `_llm_rewrite` with a fixed translation system prompt. The app assigns
them IDs of the form `lang-{language}` (e.g. `lang-French`).

### Lambda capture in loops

The technique and tactic registrations use a closure to capture the loop variable:

```python
for tid, system_prompt in _TECHNIQUE_PROMPTS.items():
    _reg(tid, ..., (lambda sp: lambda p, l, ctx="": _llm_rewrite(sp, p, l, ctx))(system_prompt))
```

The outer `lambda sp:` immediately calls itself with the current value of
`system_prompt`, binding it to `sp` in the inner lambda's closure. Without this,
all entries would share the last value of the loop variable.

---

## `app.py` — the server

### Startup (`lifespan`)

```python
@asynccontextmanager
async def lifespan(app):
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)
    app.state.llm = OpenAIChatTarget(...) if OPENAI_API_KEY else None
    app.state.llm_available = bool(OPENAI_API_KEY)
    yield
```

`initialize_pyrit_async(IN_MEMORY)` must be called exactly once before any PyRIT
converter or target is used. `IN_MEMORY` means nothing is written to disk; the DB
is discarded when the server stops. The single `OpenAIChatTarget` instance is shared
across all requests — this is safe because every LLM call uses a unique
`conversation_id`, so no state bleeds between concurrent requests.

### SSE streaming (`POST /transform`)

```python
async def event_stream():
    queue = asyncio.Queue()
    tasks = [create_task(run_one_transform(tid, queue)) for tid in selected]
           + [create_task(run_one_language(lang, queue)) for lang in languages]
    for _ in range(total):
        item = await queue.get()
        yield f"data: {json.dumps(item)}\n\n"
    yield 'data: {"done": true}\n\n'
    await asyncio.gather(*tasks, return_exceptions=True)
```

All transforms are launched as concurrent tasks. Each pushes exactly one item to
the shared queue when it finishes (success or error). The generator reads from the
queue and yields events, so results arrive at the browser in completion order, not
submission order. The final `asyncio.gather` ensures all tasks are awaited and
their exceptions are captured before the generator closes.

The response uses two headers:
- `Cache-Control: no-cache` — prevents buffering by intermediate proxies
- `X-Accel-Buffering: no` — disables Nginx buffering if deployed behind a reverse proxy

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for LLM transforms. Without it, only deterministic transforms work. |
| `OPENAI_CHAT_ENDPOINT` | `https://api.openai.com/v1` | Override for Azure OpenAI or local proxies. |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Model used for all LLM rewrites. |

---

## `static/index.html` — the frontend

Single self-contained file. No npm, no bundler, no CDN dependencies.

### Initialisation (`init`)

On load, two requests fire in sequence:
1. `GET /status` → updates the LLM badge (green / red) in the header
2. `GET /transforms` → populates `allTransforms[]`, then calls `renderControls()` to
   build the checkbox grid and `renderLangChips()` to populate language toggles

### Checkbox grid (`renderControls`)

Built entirely from the `/transforms` JSON. Groups are rendered in the fixed order
defined by `GROUP_ORDER`. Each group gets a column box with an "all" toggle that
inverts the current selection state. Items with `requires_llm: true` show a small
yellow dot as a visual hint.

To change which groups appear or their display order, update `GROUP_ORDER` and
`GROUP_LABELS` at the top of the script block.

### SSE reading

The browser uses `fetch` + `ReadableStream` rather than `EventSource` because
`EventSource` only supports GET requests. The reader loop accumulates chunks in a
string buffer, splits on newlines, and parses complete `data: ...` lines. Incomplete
lines at the end of a chunk are held in the buffer until the next chunk arrives.

```javascript
buf += decoder.decode(value, { stream: true });
const lines = buf.split('\n');
buf = lines.pop(); // hold the incomplete trailing line
```

### Card lifecycle

1. **On submit**: placeholder cards (with skeleton shimmer animation) are created
   immediately for every selected transform, prepended to the results grid, and
   stored in `cardMap[id]`.
2. **On SSE event**: `fillCard(cardMap[data.id], data)` replaces the skeleton with
   the result text (or an error message), adds a coloured left border, and shows
   the copy button.
3. **Copy button**: calls `navigator.clipboard.writeText` on the card's result text.
   Briefly shows "Copied!" then resets.

New results are **prepended** so the most recent run appears at the top without
clearing previous results. The "clear" button empties the grid entirely.

### CSS design tokens

All colours are CSS custom properties on `:root`. To retheme, change the variables
at the top of the `<style>` block. Group tag colours are hardcoded in `.tag-*`
classes — update these if you add a new group.

---

## Custom Transform field

The **Custom Transform** textarea (red section, above the checkbox grid) lets you write
free-form multi-step transformation instructions in plain English. The LLM applies them
exactly and returns only the result — no explanation.

Example instructions:

```
First wrap the payload in a plausible story about a confused customer.
Then encode only the instructions about the payload in ROT13, with inline
instructions to decode. Split the payload using Python list syntax.
Translate the non-story part into Spanish.
```

### How it works

`apply_custom` in `transforms.py` wraps your instructions in a fixed system prompt:

```python
_CUSTOM_WRAPPER = (
    "You are a prompt transformation engine. "
    "Apply the following transformation instructions to the user's input exactly as described. "
    "Return ONLY the transformed result — no explanation, no preamble, no commentary.\n\n"
    "Transformation instructions:\n"
    "{instructions}"
)
```

The wrapper explicitly forbids the LLM from explaining or commenting — it can only
transform. This prevents the model from hedging or adding meta-commentary about what
it did.

If a **Context** value is set, it is prepended to the system prompt in the same way as
for built-in techniques:

```
Target context: <context text>

You are a prompt transformation engine. Apply the following…
```

### In `app.py`

`TransformRequest` has a `custom_prompt: str = ""` field. When it is non-empty, a
`run_one_custom` task is added to the SSE task pool alongside the regular transforms.
The result card has `id="custom"`, `group="custom"`, and `label="Custom Transform"`.

### In `index.html`

- The custom textarea (`id="custom-prompt"`) sits above the checkbox grid.
- `generate()` reads it as `customPrompt`; if non-empty it adds a placeholder card with
  `id="custom"` and includes `custom_prompt` in the POST body.
- The guard `if (!selected.length && !languages.length && !customPrompt)` means you
  can run a custom transform even with zero checkboxes selected.
- `GROUP_COLORS.custom` and `.tag-custom` give the card a distinct red colour.

---

## Extending the tool

### Add a new transform group

1. Add entries to `TRANSFORMS` in `transforms.py` with the new group name.
2. Add the group to `GROUP_ORDER` and `GROUP_LABELS` in `index.html`.
3. Add a colour to `GROUP_COLORS` and a `.tag-{group}` CSS class.

### Change the LLM model

Set `OPENAI_CHAT_MODEL` before starting the server, or edit the default in `app.py`:

```python
model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
```

To use an Azure OpenAI endpoint, set `OPENAI_CHAT_ENDPOINT` to your Azure base URL.

### Add reasoning effort

To add reasoning tokens to LLM rewrites (slower but higher quality framing):

```python
app.state.llm = OpenAIChatTarget(
    ...
    extra_body={"reasoning": {"effort": "low"}},
)
```

`low` is a reasonable default for rewriting tasks. The gain is marginal for simple
text transformations; `medium` or `high` are better reserved for attack planners.

### Run without an API key

All deterministic transforms (encoding, obfuscation, structural) work with no API
key. The LLM badge shows red and technique/tactic/translation cards will emit an
error, but they won't block the deterministic results from appearing.
