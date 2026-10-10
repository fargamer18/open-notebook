"""One process-wide lock around every PyMuPDF call.

PyMuPDF does not support running on multiple threads ("doing so may cause
incorrect behaviour or even crash Python itself" — PyMuPDF documentation),
and LiteNotebook calls it from request threads (Reader pages, text geometry)
while ingestion runs in a background thread. Every PyMuPDF call holds this
lock. Long jobs take it per page, so a large ingestion never blocks the Reader
for more than one page's work; slow non-PyMuPDF work (OCR) runs outside it.
"""

from __future__ import annotations

import threading

PDF_LOCK = threading.RLock()
