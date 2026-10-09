# Study → Quizzes (Phase 13)

Notebook tab order: **Sources · Chat · Notes · Study · Memory · Search**.
Study is a container for study tools (`STUDY_TOOLS` in
`web/routes/study.py`). Phase 13 adds Quizzes. Audio overviews can be added
later as another entry, with no change to the navigation.

## Rule: grounded in the selected material only

Questions are written only from the material the user selects:

- **Sources:** all ready PDFs are checked by default; the user can uncheck any.
- **My notes:** off by default.
- **Chat history:** off by default.

Model "world knowledge" is never asked for and is rejected when it shows (see
validation). Nothing from unselected sources is read: every chunk query is
restricted to the selected source ids, this notebook and `status = 'ready'`.
The FAISS index covers the whole notebook, so its hits are filtered to the
selected chunk ids before use.

## Pipeline (`study/generate.py`)

```
selected material
  → evidence pool          (study/evidence.py)
  → bounded batches        (sized to the model's context)
  → Qwen, JSON schema      (study/prompts.py, grammar-constrained by llama.cpp)
  → server-side validation (study/validate.py)
  → bounded repair requests for shortfalls (at most 2)
  → save (one transaction) (study/store.py)
```

Stages shown while it runs: Retrieving evidence… · Starting Local AI… ·
Generating questions (request n of m)… · Validating quiz… · Saving…
No percentages are shown. Cancel works at every stage: it closes the
streaming response, so llama-server stops generating.

**Evidence pool**

- *No focus:* passages are spread evenly through each selected source and
  interleaved across sources. MiniLM is **not** started.
- *With a focus:* FTS5 and FAISS results (MiniLM, in its worker process) are
  combined with RRF inside the selection. Near-duplicates are dropped, and a
  page gives at most two passages.
- Very short chunks and chunks that are mostly symbols (for example tables of
  contents) are skipped, as long as enough other text remains.

**Context and requests**

- The context size comes from the Local AI profile (managed llama-server) or
  from llama-server's `GET /props` (otherwise 4096 is assumed).
- Questions per request depend on the context: 5 at ≥ 6000 tokens, 4 at
  ≥ 4000, 3 at ≥ 3000, else 2.
- The evidence sent per request is capped at 9,000 characters.
- Example: 5 questions at an 8k context take **one** request.
- Servers that reject `response_format` get one plain retry.

**Concurrency:** quizzes use the same `GenerationGuard` lease as Chat.

- While a quiz generates, Chat says "Local AI is busy generating a quiz", and
  a second quiz is refused.
- While Chat answers, a quiz is refused with a busy message.
- Both use the same endpoint and the same managed llama-server (autostarted
  as for Chat), so a second server is never started.
- Long quiz runs refresh the lease between requests, so it is not reclaimed
  as stale.

**Local AI not configured:** "Local AI is not configured." with a
**Set up local AI** link. Existing quizzes keep working: taking them and
reviewing them need neither Qwen nor MiniLM.

## Structured output and validation

The model must reply `{"questions": [...]}`. Each question has: `evidence`
(labels), `type`, `question`, `options`, `correct` (letters), `explanation`,
`hint` and `concepts`. llama.cpp enforces the shape with a grammar, and every
question is validated again on the server.

**Rejected:**

- wrong type, or a type the user did not request;
- single answer without exactly 4 options and 1 correct;
- multiple answers without 4 options and 2–3 correct;
- True/False whose options are not exactly True/False;
- duplicate options;
- missing or empty explanation;
- missing evidence, or evidence labels that were not offered;
- evidence from unselected sources;
- HTML or script, or over-long text;
- "according to the passage" or "all/none of the above";
- duplicate or near-duplicate questions;
- numbers (two or more digits) in the question or explanation that the
  cited evidence does not contain;
- **numbers in a correct answer (or a True statement)** that the evidence does
  not state *for the same thing*. Values are normalised deterministically:
  - digits lose thousands separators and trailing decimal zeros;
  - the words zero…twenty become digits, so "three" ↔ "3";
  - ordinals, "dozen" and compound words are not interpreted, which rejects
    rather than guesses.

  A number with its own noun ("three stages", "17 minutes") needs the same
  value next to the same word in the evidence. A bare number ("4") must be
  the only value the evidence attaches to a word of the question. So a stray
  "tram line 4" never supports "4 stages", and an ambiguous match is
  rejected. A statement marked **False** that changes a number is accepted
  only when the evidence gives a different number for the same word ("four
  stages" vs "exactly three stages", "ended in 2050" vs "ended in 2041"). This
  is positional matching, not reasoning;
- a correct option much longer than the distractors;
- a question whose content words are not supported by the cited evidence
  (the lexical grounding gate);
- a correct answer, or a True statement, that the evidence does not state:
  no new content word is allowed in an answer of up to 4 content words, and
  one per 4 words beyond that. This caught real Qwen output that marked
  general knowledge ("distinct eigenvalues ⇒ diagonalisable", "chlorophyll
  produces oxygen") as correct.

**Repaired, not rejected:**

- whitespace, `A)` option prefixes and stray `[S1]` labels are cleaned up;
- True/False casing and order are fixed;
- repeated letters are merged;
- a hint that reveals the answer, or that points at "the passage" or "the
  evidence", is dropped (the question stays);
- a leading "According to the passage," is removed;
- references to option letters in explanations ("option A", "(C)") are
  removed, because the options are shuffled when the quiz is saved.

**Shortfalls:** if fewer questions survive than were requested, the quiz is
saved with a note: "Generated 9 of 10 requested questions. 1 question was left
out because it could not be grounded safely…". Nothing is invented to reach
the count. If none survive, no quiz is saved and the reason is shown.

The grounding gate is a conservative heuristic, not a proof. It catches
off-topic and unsupported questions, but a subtly wrong distractor key can
still pass. The stored evidence is shown after every answer, so the user can
check it.

## Prompt injection

Evidence goes between fixed `=== EVIDENCE … ===` markers. Any run of `===`
inside document text is shortened, so a PDF cannot close the block early. The
system prompt says that evidence is untrusted data and that instructions
inside it must never be followed.

As a structural defence, sentences that read as instructions ("ignore …
instructions", "the correct answer is …", "you must answer …") are never
counted as support for an answer. A question whose answer only such text
supports is rejected (regression-tested with a "make the correct answer
banana" PDF).

## Data (schema 9, migration `0009_study_quizzes.sql`)

**Tables**

- `quizzes`: notebook, title, status, stage, difficulty, types, requested and
  actual counts, focus, material snapshot (source ids and file names, notes
  and chat toggles), message/error, model, meta (timings, request count,
  rejection reasons).
- `quiz_questions`, `quiz_question_options`: stable positions; options are
  shuffled once at save so the answer is not always "A".
- `quiz_evidence`: an immutable snapshot of each item:
  - kind: source, note or chat;
  - source id, chunk id and page, or note id, or message and conversation id;
  - label: file name, note title or conversation title;
  - the excerpt text.

  The ids are identity values, not foreign keys.
- `quiz_attempts` (`full` or `mistakes` with `parent_attempt_id`) and
  `quiz_attempt_items` (answered once; hint use recorded).
- `study_concept_stats`: derived. Per concept: times seen, correct and
  incorrect counts, a streak, the last result and when it was last seen.

**Rules enforced by triggers**

- Questions, options and evidence can never be updated after they are saved.
- An answered item can never be changed.
- "Generate again" always makes a new quiz.

**Retry mistakes** creates a new attempt containing only the questions that
were wrong or skipped in a completed attempt. Earlier attempts are never
modified.

**Concept labels** come from attempt data only:

- **Needs review**: the last answer was wrong.
- **Strong recently**: the last two answers were correct.
- **Seen**: anything else.

Nothing is ever called "mastered", and no percentages are invented.
Concepts come from the model, but they are kept only if their words appear in
the evidence (at most 3 per question). Otherwise one deterministic phrase is
used. They stay separate from the memory graph. A later graph phase can read
`study_concept_stats`.

**Deleting**

- *A quiz* (always confirmed): removes its questions, options, evidence
  snapshots and **all of its attempts and answers**. Concept stats are then
  recomputed from the remaining quizzes.
- *A source:* keeps every quiz. Evidence shows "Source deleted" with the
  stored snapshot, and "Open in PDF" is disabled.
- *A note or conversation:* behaves the same way.
- *A notebook:* deletes its quizzes.

**Open in PDF** opens the Reader at the cited chunk:
`…/read?chunk=<id>&attempt=<id>`. The chunk is highlighted, and "Back to quiz"
returns to the review. If the chunk is gone but the source remains, the link
opens the stored page.

## What wakes which component

| Action | MiniLM | Qwen / llama-server |
|---|---|---|
| Study home, history, quiz page, create form | no | no |
| Taking a quiz, hints, results, review, retry | no | no |
| Generate, no focus | no | yes |
| Generate, with focus | yes (worker process) | yes |

The web process never imports torch or sentence-transformers.

## Backup / export (archive format version 2)

`export_format_version` changed from 1 to 2 because older LiteNotebook builds
reject unknown tables. A version-1 reader therefore cannot be given quiz
tables, and it now gets the clear "created by a newer LiteNotebook version"
message instead.

- **Version 2 archives** add `quizzes`, `quiz_questions`,
  `quiz_question_options`, `quiz_evidence`, `quiz_attempts`,
  `quiz_attempt_items` and `study_concept_stats`.
- **Version 1 (Phase 10–12) archives still import.** The Study tables are
  simply absent. `tests/fixtures/phase12_notebook_v1.zip` is a real archive
  made by the Phase 12 code and is imported in the tests.

**On import**

- Every quiz id is remapped.
- Evidence and material snapshots point at the new source, chunk, note,
  message and conversation ids. Items that were not exported (for example a
  source deleted before the export) become `NULL`, and their text snapshots
  stay.
- Answers must belong to their attempt's quiz.
- JSON columns are validated.
- The integrity check verifies that concept stats match the answers.
- A quiz exported mid-generation is imported as failed.
