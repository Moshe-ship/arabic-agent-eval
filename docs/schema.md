# Schema

The canonical source of truth is `arabic_agent_eval/dataset.py` (Python dataclasses). The JSONL files under `data/` are derived artifacts; regenerate with `aae export` or `python scripts/export_jsonl.py`.

## `EvalItem`

Every evaluation row in `data/*.jsonl` conforms to this shape:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier, prefixed by category (e.g. `simple_001`, `dialect_007`). |
| `category` | `str` | One of: `simple_function_calling`, `parameter_extraction`, `multi_step`, `dialect_handling`, `tool_selection`, `error_recovery`. |
| `instruction` | `str` | Natural-language user instruction in Arabic. |
| `dialect` | `str` | One of: `msa`, `gulf`, `egyptian`, `levantine`, `maghrebi`. |
| `available_functions` | `list[str]` | Function names exposed to the agent for this item. Must all be declared in `data/functions.json`. |
| `expected_calls` | `list[ExpectedCall]` | Ordered list of tool calls the agent should produce. |
| `difficulty` | `str` | One of: `easy`, `medium`, `hard`. |
| `error_response` | `dict \| null` | Present only for `error_recovery` items; the simulated tool error the agent must surface. |

## `ExpectedCall`

| Field | Type | Description |
|---|---|---|
| `function` | `str` | Target function name. The literal `*` acts as wildcard for multi-step items where the intermediate call's arguments depend on runtime state. |
| `arguments` | `dict[str, Any]` | Expected argument key/value pairs. Argument values may be `*` (wildcard) to accept any concrete value. Arabic string values are compared after normalization (see [grading.md](grading.md)). |

## Function registry

`data/functions.json` contains the 22 tool definitions exposed to agents. Each follows the OpenAI function-calling schema with Arabic metadata:

```json
{
  "name": "book_hotel",
  "name_ar": "حجز فندق",
  "description": "Book a hotel room",
  "description_ar": "حجز غرفة في فندق",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name"},
      "check_in": {"type": "string", "description": "Check-in date"},
      "check_out": {"type": "string", "description": "Check-out date"},
      "guests": {"type": "integer", "description": "Number of guests"}
    },
    "required": ["city", "check_in", "check_out"]
  }
}
```

The `name` and `description` fields are English for cross-model interoperability; `name_ar` and `description_ar` are the human-readable Arabic counterparts the evaluator surfaces to the agent (via `to_openai_tools`). Arabic descriptions are the primary signal to the model during evaluation.

## Category → weight mapping

Overall score is a weighted average across categories:

| Category | Weight |
|---|---:|
| `simple_function_calling` | 0.20 |
| `parameter_extraction` | 0.20 |
| `multi_step` | 0.20 |
| `dialect_handling` | 0.15 |
| `tool_selection` | 0.15 |
| `error_recovery` | 0.10 |

Weights sum to 1.0. Change via `CATEGORIES` in `arabic_agent_eval/dataset.py`.

## Versioning

The dataset ships as `v0.1.0 — reference set`. Expansion is tracked in GitHub issues; the Python literal source is the canonical versioning surface.
