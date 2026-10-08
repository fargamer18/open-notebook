"""Operating-system differences, kept in one small place.

LiteNotebook targets Linux and Windows. Only behaviour that genuinely differs
between them lives here (file locking, directory fsync, binary file flags,
renaming directories that may still be in use); everything else uses
``pathlib`` and the standard library directly.
"""
