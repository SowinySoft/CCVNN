import os

# Extend package search path so 'watcher.<module>' resolves to 'watcher/src/<module>.py'
_src_dir = os.path.join(os.path.dirname(__file__), "src")
if os.path.exists(_src_dir) and _src_dir not in __path__:
    __path__.append(_src_dir)