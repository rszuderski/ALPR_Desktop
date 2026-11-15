"""Bounded image cache and coalesced prefetch, with no Tk or annotation writes."""
from collections import deque
from concurrent.futures import Future
import os
from pathlib import Path
import threading

from PIL import Image, ImageOps


def path_key(path):
    return os.path.normcase(os.path.abspath(str(path)))


def image_key(path):
    stat = Path(path).stat()
    return path_key(path), stat.st_mtime_ns, stat.st_size


def decode_image(path):
    # Pillow supports Windows Unicode paths. Match the EXIF orientation used by
    # the previous OpenCV decoder so cached pixels and polygon coordinates agree.
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        return oriented if oriented.mode == "RGB" else oriented.convert("RGB")


class PreviewImages:
    def __init__(self, cache=None, *, max_items=24, max_bytes=128 * 1024 * 1024, decoder=decode_image):
        self.cache = cache if isinstance(cache, dict) else {}
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.decoder = decoder
        self.lock = threading.RLock()
        self.inflight = {}
        self.pending = deque()
        self.worker = None

    @staticmethod
    def _size(image):
        return image.width * image.height * len(image.getbands())

    def load(self, path):
        key = image_key(path)
        with self.lock:
            if key in self.cache:
                image = self.cache.pop(key)
                self.cache[key] = image
                return image
            future = self.inflight.get(key)
            owner = future is None
            if owner:
                future = Future()
                self.inflight[key] = future
        if not owner:
            return future.result()
        try:
            image = self.decoder(Path(path))
            with self.lock:
                for stale in list(self.cache):
                    if stale[0] == key[0]:
                        self.cache.pop(stale)
                if self._size(image) <= self.max_bytes:
                    self.cache[key] = image
                    while len(self.cache) > self.max_items or sum(self._size(v) for v in self.cache.values()) > self.max_bytes:
                        self.cache.pop(next(iter(self.cache)))
                future.set_result(image)
            return image
        except Exception as exc:
            future.set_exception(exc)
            raise
        finally:
            with self.lock:
                self.inflight.pop(key, None)

    def prefetch(self, paths):
        with self.lock:
            cached_paths = {key[0] for key in self.cache}
            # A new direction/run replaces queued work; only an in-flight decode
            # is allowed to finish. Cached images never start another thread.
            self.pending = deque(dict.fromkeys(Path(p) for p in paths if path_key(p) not in cached_paths))
            if not self.pending or self.worker is not None:
                return
            self.worker = threading.Thread(target=self._prefetch, name="z2-image-prefetch", daemon=True)
            try:
                self.worker.start()
            except RuntimeError:
                self.worker = None

    def _prefetch(self):
        while True:
            with self.lock:
                if not self.pending:
                    self.worker = None
                    return
                path = self.pending.popleft()
            try:
                self.load(path)
            except Exception:
                continue


def preview_images(host):
    loader = getattr(host, "_preview_image_loader", None)
    if loader is None:
        loader = PreviewImages(getattr(host, "_preview_render_image_cache", None))
        host._preview_image_loader = loader
        host._preview_render_image_cache = loader.cache
    return loader
