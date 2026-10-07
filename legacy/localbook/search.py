import sqlite3
import sys
from pathlib import Path

DB = Path("/run/media/faraz/DATA/LocalBook/indexes/linear_equations.db")

if len(sys.argv) < 2:
    print('Usage: python app/search.py "search query"')
    raise SystemExit(1)

query = " ".join(sys.argv[1:])

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

# Convert ordinary words into an FTS OR query.
# This is more forgiving of noisy OCR than requiring every word.
words = [w.strip() for w in query.split() if w.strip()]
fts_query = " OR ".join(f'"{w}"' for w in words)

rows = con.execute("""
    SELECT
        p.pdf,
        p.page,
        p.text,
        bm25(pages_fts) AS score
    FROM pages_fts
    JOIN pages p ON p.id = pages_fts.rowid
    WHERE pages_fts MATCH ?
    ORDER BY score
    LIMIT 10
""", (fts_query,)).fetchall()

if not rows:
    print(f'\nNo results for: "{query}"')
    raise SystemExit()

print(f'\nSearch: "{query}"')
print(f"Results: {len(rows)}\n")

for i, row in enumerate(rows, 1):
    text = " ".join((row["text"] or "").split())

    if len(text) > 500:
        text = text[:500] + "..."

    print("=" * 80)
    print(f"[{i}] {row['pdf']}")
    print(f"    Page: {row['page']}")
    print(f"    BM25: {row['score']:.4f}")
    print()
    print(text)
    print()

con.close()
