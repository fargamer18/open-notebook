from pathlib import Path
import fitz

ROOT = Path.home() / "Gate" / "System of Linear Equations"

pdfs = sorted(ROOT.glob("*.pdf"))

print(f"\nFound {len(pdfs)} PDFs\n")

for pdf_path in pdfs:
    try:
        doc = fitz.open(pdf_path)

        chars = 0
        nonempty_pages = 0
        sample = ""

        for page in doc:
            text = page.get_text("text").strip()
            chars += len(text)

            if text:
                nonempty_pages += 1

                if not sample:
                    sample = " ".join(text.split())[:100]

        ratio = nonempty_pages / len(doc) if len(doc) else 0

        if ratio >= 0.8:
            status = "GOOD"
        elif ratio >= 0.3:
            status = "PARTIAL"
        else:
            status = "OCR NEEDED"

        print("=" * 80)
        print(f"FILE:   {pdf_path.name}")
        print(f"PAGES:  {len(doc)}")
        print(f"TEXT:   {chars:,} chars")
        print(f"PAGES WITH TEXT: {nonempty_pages}/{len(doc)}")
        print(f"STATUS: {status}")
        print(f"SAMPLE: {sample or '[NO TEXT FOUND]'}")

        doc.close()

    except Exception as e:
        print("=" * 80)
        print(f"FILE:  {pdf_path.name}")
        print(f"ERROR: {e}")

