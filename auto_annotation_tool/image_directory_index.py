"""Image names and counts shared by campaign resource readers.

Membership depends on directory entries, not image contents. Validate each visited
directory before reusing a scan, including nested folders changed outside the app.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from threading import RLock


class _DirectoryWatch:
    """Windows can retain directory timestamps after unlink; watch membership."""
    def __init__(self, path, recursive):
        import ctypes
        from ctypes import wintypes
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.FindFirstChangeNotificationW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL, wintypes.DWORD]
        self.api.FindFirstChangeNotificationW.restype = wintypes.HANDLE
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.FindCloseChangeNotification.argtypes = [wintypes.HANDLE]
        self.handle = self.api.FindFirstChangeNotificationW(path, recursive, 3)
        if self.handle == wintypes.HANDLE(-1).value:
            self.handle = None
            raise OSError("Directory notifications unavailable")

    def changed(self):
        return not self.handle or self.api.WaitForSingleObject(self.handle, 0) != 258

    def close(self):
        if self.handle:
            self.api.FindCloseChangeNotification(self.handle)
            self.handle = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()


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
                stamps, result, watch = cached
                try:
                    valid = (watch is None or not watch.changed()) and all(self._stamp(path) == stamp for path, stamp in stamps)
                except OSError:
                    valid = False
                if valid:
                    self._cache.move_to_end(key)
                    return result
                del self._cache[key]
                if watch is not None:
                    watch.close()

        names, stamps = [], []
        pending = [root]
        cacheable = True
        watch = None
        if os.name == "nt":
            try:
                watch = _DirectoryWatch(root, recursive)
            except OSError:
                cacheable = False
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
            cacheable = cacheable and (watch is None or not watch.changed()) and all(self._stamp(path) == stamp for path, stamp in stamps)
        except OSError:
            cacheable = False
        if cacheable:
            with self._lock:
                previous = self._cache.pop(key, None)
                if previous is not None and previous[2] is not None:
                    previous[2].close()
                self._cache[key] = (tuple(stamps), result, watch)
                self._cache.move_to_end(key)
                while len(self._cache) > self.capacity:
                    _, removed = self._cache.popitem(last=False)
                    if removed[2] is not None:
                        removed[2].close()
        elif watch is not None:
            watch.close()
        return result


IMAGE_DIRECTORIES = ImageDirectoryIndex()
