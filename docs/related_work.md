# Related work

This benchmark sits in a three-way intersection: function-calling evaluation, Arabic NLP, and agent-level (not model-level) testing. No prior artifact occupies that intersection. We position against the closest works below.

## vs Berkeley Function-Calling Leaderboard (BFCL)

BFCL ([gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html)) is the de-facto English function-calling benchmark. It covers simple, multiple, parallel, and nested function calls with thousands of items and an AST-based grader. It is **monolingual English** and has no dialect dimension. Arabic Agent Eval is not a replacement — it is an orthogonal axis: smaller scale, specific to Arabic, with dialect splits (MSA, Gulf, Egyptian, Levantine, Maghrebi) that BFCL does not provide. A natural extension is to run our grading axes on any BFCL item translated into Arabic (future work).

## vs MADAR / Arabic dialect resources

MADAR (Bouamor et al. 2018) and related resources (ArabicDialectNLI, ADI corpus, NADI shared tasks) focus on dialect *classification* — can the model tell Gulf from Egyptian? Our benchmark assumes a correct dialect label and tests whether dialectal framing **disrupts tool-call behavior**. We consume dialect labels as an input axis; they produce dialect labels as an output axis. Complementary, non-overlapping.

## vs Habibi corpus and Arabic morphological resources

Habibi (Habibullah et al.), CAMeL Tools morphological analyzers, MADAMIRA, and Farasa all operate at the **corpus / word** level: tokenization, root extraction, POS tagging, lemmatization. Arabic Agent Eval operates at the **tool-argument level** — a layer above. Morphological resources are a natural dependency, not a competitor. The [MTG project](https://github.com/Moshe-ship/mtg) bridges the two by adding morphological type guards over tool-call parameters.

## vs Arabic LLM benchmarks (OALL, HELM-Arabic, ABBL, Silma)

Existing Arabic leaderboards measure **model-level capabilities**: reading comprehension, translation, summarization, cultural knowledge. They do not evaluate **agent-level behavior** — tool selection, argument extraction, multi-step planning. Arabic Agent Eval is the first artifact at the agent tier. It is compatible with any model on those leaderboards and can be run on an existing Arabic LLM ranking to add an agent-behavior column without duplicating existing work.

## vs Hermes-Function-Calling (NousResearch)

NousResearch's Hermes-Function-Calling repo ships eval suites in `eval/` and `template_tests/` — all English. The arXiv 2601.05101 paper ("Arabic Prompts with English Tools") explicitly calls for native Arabic FC benchmarks with dialect splits. This benchmark is designed to land upstream at `NousResearch/Hermes-Function-Calling/eval/arabic/` as the missing piece.

## What this benchmark is NOT

- Not a replacement for BFCL or any English FC benchmark.
- Not a general Arabic LLM capability benchmark — it tests *agent* behavior specifically.
- Not complete coverage of Arabic dialects (see `baselines.md` for current per-dialect item counts).
- Not a training set — items are held out for evaluation only.
- Not a corpus of Arabic morphology or lexicography.
