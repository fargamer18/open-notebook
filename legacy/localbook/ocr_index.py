from pathlib import Path
import pymupdf
import subprocess
import sqlite3
import tempfile
import hashlib

SOURCE = Path.home() / "Gate" / "System of Linear Equations"
DB = Path("/run/media/faraz/DATA/LocalBook/indexes/linear_equations.db")

DB.parent.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(DB)

con.execute("""
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    pdf TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT,
    hash TEXT UNIQUE
)
""")

con.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts
USING fts5(
    text,
    content='pages',
    content_rowid='id'
)
""")

pdfs = sorted(SOURCE.glob("*.pdf"))

print(f"Found {len(pdfs)} PDFs")

for pdf_num, pdf in enumerate(pdfs, 1):

    doc = pymupdf.open(pdf)

    print()
    print("=" * 70)
    print(f"[{pdf_num}/{len(pdfs)}] {pdf.name}")
    print(f"{len(doc)} pages")
    print("=" * 70)

    for page_num, page in enumerate(doc):

        identifier = f"{pdf}:{page_num}"
        digest = hashlib.sha256(identifier.encode()).hexdigest()

        existing = con.execute(
            "SELECT id FROM pages WHERE hash=?",
            (digest,)
        ).fetchone()

        if existing:
            print(
                f"\r  page {page_num+1}/{len(doc)} [cached]",
                end="",
                flush=True
            )
            continue

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5),
                alpha=False
            )

            pix.save(tmp.name)

            result = subprocess.run(
                [
                    "tesseract",
                    tmp.name,
                    "stdout",
                    "-l", "eng",
                    "--psm", "6"
                ],
                capture_output=True,
                text=True
            )

            text = result.stdout.strip()

        cur = con.execute(
            """
            INSERT INTO pages(pdf,page,text,hash)
            VALUES(?,?,?,?)
            """,
            (
                pdf.name,
                page_num + 1,
                text,
                digest
            )
        )

        rowid = cur.lastrowid

        con.execute(
            "INSERT INTO pages_fts(rowid,text) VALUES (?,?)",
            (rowid, text)
        )

        con.commit()

        print(
            f"\r  page {page_num+1}/{len(doc)} "
            f"[{len(text):,} chars]",
            end="",
            flush=True
        )

    print("\n  done")

con.close()

print()
print("Index created:")
print(DB)
