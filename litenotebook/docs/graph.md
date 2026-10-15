# Knowledge graph (Memory → Graph)

Since 0.9.3 the Memory tab has an interactive knowledge graph with two views:

- **Global**: the meaningful structure of the whole notebook, bounded.
- **Local**: one node and its neighbourhood, depth 1, 2 or 3.

It is a **view** over data LiteNotebook already has: graph memory
(`memory_nodes` / `memory_edges`, built deterministically from chats and
notes), sources, Reader annotations and quizzes. The graph never writes to
the database, is never authoritative, and needs **no AI**. Opening it,
filtering, searching and inspecting never start MiniLM, llama.cpp or speech.
Rebuilding memory (`memory.graph.rebuild`) works exactly as before; the graph
simply shows the result.

## What the graph does *not* mean

- **Graph connectivity is not the same as conceptual importance.** A large
  node is connected to many items in *this notebook*. It is not more
  important, more correct or better understood.
- Concepts are extracted deterministically from words in your questions,
  answers, corrections and notes (`memory/concepts.py`).
  They are word and phrase matches, not a model of meaning. Two concepts close
  together are connected through the same items, not related in meaning.
- Clusters you can see are layout effects of shared connections. They are not
  semantic categories.
- Study markers reflect your actual quiz answers only. Nothing is ever marked
  "mastered".

## Node types

| Type | Shape | Shown by default | Opens |
|---|---|---|---|
| Concept | circle | yes | its local graph; *Memory details* |
| Note | square | yes | the note editor |
| Source | tall rectangle | yes | the Reader (*Read PDF*) or the source page |
| Conversation | diamond | yes | the chat |
| Correction | triangle | yes | the corrected answer in its chat |
| Question | downward triangle | no | the message in its chat |
| Answer | hexagon | no | the message in its chat |
| Evidence (a cited or linked passage) | small square | no | the Reader at that page with the passage highlighted |
| Annotation | small circle | no | the Reader at that annotation |
| Quiz (study activity) | star | no | the quiz; *Review latest attempt* |

Each type has its own shape and a legend, so colour is never the only cue.
Colours come from CSS variables (`--g-*` in `app.css`) and suit the app's
dark theme.

Markers:

- **Needs review**: an amber ring around a concept whose most recent quiz
  answers were wrong (the Study tab's rule: a negative streak).
- **Seen** / **Strong recently**: shown in the inspector, taken from
  `study_concept_stats` when the concept's key matches exactly.
- **Corrected / superseded**: an answer you corrected is drawn faded with a
  dashed outline, and `corrects` links are dashed. The inspector says "this
  answer was corrected — prefer the correction". A conversation that contains
  a correction says so. History is never hidden: tick *Answers* to see the
  original answer next to its correction.
- **Useful** / **Dead end**: from your ratings. Dead ends are faded like
  superseded answers.

## Edges

Every edge comes from stored data:

| Edge | From |
|---|---|
| `contains`, `answered_by`, `followed_by`, `mentions`, `supported_by`, `corrects` | graph memory of chats |
| `mentions`, `supported by` (notes) | graph memory of notes (`note_mentions`, `note_supported_by`) |
| `passage of` | an evidence passage and the source it comes from |
| `annotates` | an annotation and its source |
| `linked to` | a note and an annotation linked to it |
| `tests`, `references` | a quiz, the concepts its questions test and the sources its evidence cites |

**Hidden types are contracted, not cut out.** When questions, answers or
evidence are hidden (the default), their edges are re-attached to the visible
object that owns them: questions, answers and corrections to their
conversation, evidence to its source. For example, "conversation —mentions→
concept" is the stored "question —mentions→ concept" seen from the
conversation. Each such edge records how many stored edges it stands for
(thicker = more) and which hidden types it passed through ("via"). No edge is
invented to make the graph denser.

## Global graph

Default types: concepts, notes, sources, conversations and corrections.

Filters (**Filters** under the toolbar):

- Node types (any of the ten above).
- *Hide nodes without connections*.
- *Minimum connections*: a degree threshold for every node.
- *Concepts*: *Auto*, *All*, *Shared by 2+*, *Shared by 3+*. A concept
  mentioned by one item only adds a leaf, not structure. In *Auto* mode, once
  the view has more than 40 concepts, only concepts shared by at least two
  items are shown. The status line always says how many concepts were hidden.
- *Show labels*. Labels are placed most important first; a label that would
  overlap another is skipped until you zoom in or hover the node.

Filter choices (types, toggles, depth) are remembered in this browser's
`localStorage`. Only these view settings are stored there, never labels or
content. Pinned node positions are not stored.

## Local graph

Starts from one node: the selected node, a search result, *Show local graph*
in the inspector, a double-click, the **L** key, or *Open in graph* on a memory
item. Depth 1–3, breadth first in a fixed order (type, connections, label), at
most 150 nodes. The status line reports when the neighbourhood was larger.
The URL (`?mode=local&node=…&depth=…`) can be bookmarked or reloaded.

## Inspector

Selecting a node shows its type, markers, a short excerpt (at most 240
characters, never a whole document), facts such as pages, latest quiz score or
how often a concept was quizzed, and buttons that open the real object (see
the table above). It lists the node's connections grouped by type (eight of
each, with a count). Choosing one selects it, or opens its local graph if it
is not in the current view. Connections use the contracted view too, so a
conversation lists its concepts and cited passages, and a passage lists its
source.

From a cited passage, *Open passage in PDF* opens the Reader at that page with
the passage highlighted (the same `?citation=` link the chat uses).

## Search

Type in the search field (or press **/**) to highlight matching nodes in the
view. A list of matches across the whole notebook appears, with matches
already in the view first. **Enter** selects the first match. A match outside
the view opens its local graph. Search is a case-insensitive substring match
on labels: no MiniLM, no model.

## Mouse and keyboard

- Drag empty space to pan, scroll to zoom, drag a node to move it. A dragged
  node stays pinned until **Reset layout**.
- Click to select, double-click for the local graph. Hovering highlights a
  node's neighbours and dims the rest.

With the graph focused (Tab to it):

| Key | Action |
|---|---|
| ← ↑ → ↓ | move to the connected node in that direction |
| Enter | open the selected node (its first action) |
| Esc | clear the selection and search highlight |
| + / − | zoom |
| 0 | fit the graph |
| L | local graph of the selected node |

`/` focuses the search field, as on other pages. All keys are listed in the
app's shortcut help (**?**).

## Accessibility

The canvas is not the only way in:

- The graph region is focusable, with a description of its keys. The status
  line (`role="status"`) announces counts, truncation and the selected node.
- The inspector is plain HTML with buttons and links.
- **List view of this graph** (below the graph) lists every node in the
  current view with its relationships (relation, direction and the node at the
  other end) as buttons. It works with a keyboard or a screen reader and
  selects nodes in the graph and the inspector.
- Every control has a label. Mode buttons expose `aria-pressed`.
- Without JavaScript, the page points to the memory list view.
- **Reduced motion.** When the system or browser asks for reduced motion
  (`prefers-reduced-motion: reduce`), the layout is computed out of sight in
  short chunks (at most 12 ms each, yielding to input in between) and only the
  settled graph is drawn: no animated settling. Dragging a node moves only that
  node; its neighbours do not spring after it. Pan, zoom, selection and every
  other control work as usual. The preference is followed live if it changes.

## Bounds and performance

| Limit | Value |
|---|---|
| Global graph nodes (default / maximum on request) | 400 / 1,500 |
| Global graph links (default / maximum) | 1,600 / 6,000, strongest kept |
| Local graph | depth ≤ 3, ≤ 150 nodes, ≤ 600 links |
| Memory nodes read per request | 50,000 (the status line says when more exist) |
| Label / inspector excerpt | 80 / 240 characters |

When a cap applies, the most connected nodes are kept. Ties are broken by
type, label and id, so the result is always the same, and the status line says
"Showing the most connected 400 of 1,200 nodes."

Measured on this machine, headless Chromium:

| | Small notebook (37 nodes) | Synthetic: 1,200 nodes, 4,200 links |
|---|---|---|
| `view.json` | 4.9 ms, 8 KB | 18.6 ms, 100 KB (400 shown) |
| `local.json` depth 3 | 7.0 ms | 18.2 ms (150 nodes) |
| `node.json` / `find.json` | 4.4 / 4.2 ms | 15.7 / 15.1 ms |
| First draw → settled layout | 0.2 → 1.7 s | 0.3 → 1.7 s (400 shown) |
| Frame time while laying out and hovering | 16.7 ms | 16.7 ms median and p95 |
| JS heap | 4–7 MB | 5.6 MB (400), 33.5 MB (1,200) |

The browser stayed at 60 fps even with all 1,200 nodes. The 400-node default
is for readability, not speed. `?max_nodes=` on the page URL raises it, up to
the 1,500 server cap.

## How it is built

- Server: `memory/visual.py` builds the view, contraction, filters, ranking,
  local BFS, the inspector payload and search. Routes are in
  `web/routes/memory.py`: `/memory/graph` (page), `/graph/view.json`,
  `/graph/local.json`, `/graph/node.json`, `/graph/find.json`. Every route is
  scoped by the notebook in the URL, and node ids (`m:…`, `s:…`, `a:…`,
  `z:…`) are checked against that notebook. Payloads hold ids, types, labels,
  degrees, flags and small metadata only, never chunk text or note bodies.
- Browser: `static/graph.js`, vanilla JavaScript with no library and no
  build step. A canvas draws the graph with a small
  force-directed layout: Barnes–Hut repulsion, springs, a gentle pull to the
  centre, deterministic sunflower-spiral start positions. All text goes
  through `textContent`; no labels are ever rendered as HTML.
- Considered and not chosen: a vendored `d3-force` (ISC licence, maintained,
  about 30 KB). The custom layout is a few hundred lines, covers what the view
  needs, and adds no third-party code or licence to track. SVG was used by
  the old explorer; canvas scales better for hundreds of nodes. Accessibility
  comes from the HTML list view and inspector, not the drawing. No WebGL.
- The old radial explorer was replaced by the Local graph. Its inspector page
  (*Memory details*) and the `graph.json` subgraph endpoint remain.

## Concept cleanup

Phase 18 only added conservative filtering to concept extraction: LaTeX
commands (`\frac`, `\mathbf`), LaTeX environment names (`\begin{pmatrix}`),
URLs and e-mail addresses are run breaks, so they never become concepts.
Existing graph memory changes only when it is rebuilt or a chat or note is
re-synced. Extraction otherwise works as before; richer, model-based
extraction is post-v1 work.

## Data, backup and schema

Nothing new is stored in the notebook: the schema stays **10** and the export
format stays **3**. Only view preferences live in the browser's
`localStorage`.
