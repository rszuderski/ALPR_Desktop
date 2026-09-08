"""Image names and counts shared by campaign resource readers.

Membership depends on directory entries, not image contents. Validate each visited
directory before reusing a scan, including nested folders changed outside the app.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from threading import RLock


@dataclass(frozen=True)
class ImageDirectorySnapshot:
    count: int
    token: str


class ImageDirectoryIndex:
    def __init__(self, capacity=32):
        self.capacity = capacity
        self._cache = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def _stamp(path):
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    def snapshot(self, root, extensions, *, recursive=False):
        root = os.path.abspath(os.fspath(root))
        extensions = frozenset(str(ext).lower() for ext in extensions)
        key = os.path.normcase(root), recursive, extensions
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                stamps, result = cached
                try:
                    valid = all(self._stamp(path) == stamp for path, stamp in stamps)
                except OSError:
                    valid = False
                if valid:
                    self._cache.move_to_end(key)
                    return result
                del self._cache[key]

        names, stamps = [], []
        pending = [root]
        cacheable = True
        while pending:
            directory = pending.pop()
            try:
                stamps.append((directory, self._stamp(directory)))
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            if entry.is_symlink():
                                # A link's target may change outside the scanned tree.
                                cacheable = False
                            if entry.is_dir(follow_symlinks=False):
                                if recursive:
                                    pending.append(entry.path)
                            elif os.path.splitext(entry.name)[1].lower() in extensions and entry.is_file():
                                names.append(entry.name.strip().lower())
                        except OSError:
                            cacheable = False
            except OSError:
                cacheable = False
        unique_names = sorted(set(name for name in names if name))
        token = ""
        if unique_names:
            digest = hashlib.sha1("\n".join(unique_names).encode("utf-8")).hexdigest()[:20]
            token = f"iset_{len(unique_names):05d}_{digest}"
        result = ImageDirectorySnapshot(len(names), token)
        try:
            cacheable = cacheable and all(self._stamp(path) == stamp for path, stamp in stamps)
        except OSError:
            cacheable = False
        if cacheable:
            with self._lock:
                self._cache[key] = (tuple(stamps), result)
                self._cache.move_to_end(key)
                while len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
        return result


IMAGE_DIRECTORIES = ImageDirectoryIndex()
