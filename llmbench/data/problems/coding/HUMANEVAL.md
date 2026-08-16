# HumanEval, converted into this project's problem format

The 164 `humaneval_NNN/` directories beside this file are OpenAI's **HumanEval**
benchmark, converted once and committed. Source:
<https://github.com/openai/human-eval>, file `data/HumanEval.jsonl.gz`, retrieved
2026-08-16.

The prompts, the reference solutions and the tests are OpenAI's. What this project
added is the packaging: the split into three files, and a wrapper that hands the
candidate to HumanEval's own `check`.

## Why it is committed rather than downloaded

Bundled data is reached only through `resources.py` and ships inside the wheel. A
benchmark that needs a download is a benchmark that fails on a train, and a
downloaded corpus is one silent upstream edit away from making last month's results
incomparable. Committing the converted output also means the problem set is readable
here and diffable when it changes. The converter itself is not shipped; it ran once.

## How each problem maps

| HumanEval | here |
|---|---|
| `task_id` | `source:` in `problem.yaml`, and the directory name |
| `prompt` | `prompt:` in `problem.yaml`, verbatim |
| `entry_point` | `entrypoint:` in `problem.yaml` |
| `prompt` + `canonical_solution` | `solution.py` — the complete reference module |
| `test` | `tests.py`, unmodified, plus a four-line pytest wrapper |

Every one of the 164 reference solutions was executed against its converted tests
through this project's own harness before being committed. All 164 pass.

## One difference from HumanEval's own protocol, stated plainly

HumanEval asks a model to **complete** the prompt, so whatever the prompt already
defines is always present in the graded program. This bench asks the model for a
complete solution given the prompt as the problem statement.

For 161 problems that is the same thing. For three of them — `humaneval_032`,
`humaneval_038` and `humaneval_050` — the prompt defines a *helper* (`poly`,
`encode_cyclic`, `encode_shift`) that HumanEval's `check` calls directly, so a model
that writes only the entry point will fail them where a completion-style harness
would not. The helper is visible in the prompt and the instruction does say to write
a complete solution, so this is harder rather than unfair — but it is a difference,
and a score from here is not interchangeable with a published HumanEval number.

That is true of this bench generally: it compares configurations you control, not
models against a public leaderboard.

## Licence

HumanEval is MIT licensed. The full notice, as published:

```
The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## A word on what running these costs

This is 41× more model-generated code executed than the four problems that were here
before. The security posture has not changed and was never a sandbox — see the
README — but the exposure has. `execute: false` grades extraction only, and running
the bench inside a container is the recommendation for anything adversarial.
