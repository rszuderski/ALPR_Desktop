"""One parsed preview file per Z3 host, shared by read-only UI queries."""
from __future__ import annotations

import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def path_key(path) -> str:
    return os.path.normcase(os.path.abspath(str(path))) if path else ""


def file_signature(path) -> tuple:
    stat = Path(path).stat()
    return path_key(path), stat.st_mtime_ns, stat.st_size


def read_preview_metadata(host, path, *, prefer_live: bool = True) -> dict:
    """Readers must copy records before normalizing them. Only the editor owns mutations."""
    path = Path(path)
    live = getattr(host, "preview_metadata", None)
    if prefer_live and isinstance(live, dict) and path_key(getattr(host, "_loaded_meta_path", None)) == path_key(path):
        return live
    signature = file_signature(path)
    cached = getattr(host, "_preview_metadata_file_cache", None)
    if isinstance(cached, tuple) and cached[0] == signature:
        return cached[1]
    with path.open("r", encoding="utf-8-sig") as stream:
        payload, fragments = decode_preview_records(stream.read())
    host._preview_metadata_file_cache = (signature, payload, fragments)
    return payload


def decode_preview_records(text: str) -> tuple[dict, dict[str, str]]:
    """Parse once and retain immutable JSON fragments for unchanged records.

    Autosave can then encode only edited plates and assemble the complete file
    in a worker, without copying or serializing all raw YOLO proposals in Tk.
    """
    whitespace = re.compile(r"[ \t\r\n]*")
    decoder = json.JSONDecoder()
    index = whitespace.match(text).end()
    if text[index:index + 1] != "{":
        json.loads(text)  # Preserve validation of malformed non-object input.
        return {}, {}
    index = whitespace.match(text, index + 1).end()
    payload, fragments = {}, {}
    if text[index:index + 1] == "}":
        index += 1
    else:
        while True:
            key, index = decoder.raw_decode(text, index)
            if not isinstance(key, str):
                raise ValueError("Klucz metadata.json musi być tekstem")
            index = whitespace.match(text, index).end()
            if text[index:index + 1] != ":":
                raise ValueError("Brak dwukropka w metadata.json")
            start = whitespace.match(text, index + 1).end()
            value, index = decoder.raw_decode(text, start)
            payload[key] = value
            fragments[key] = text[start:index]
            index = whitespace.match(text, index).end()
            token = text[index:index + 1]
            if token == "}":
                index += 1
                break
            if token != ",":
                raise ValueError("Nieprawidłowy separator w metadata.json")
            index = whitespace.match(text, index + 1).end()
    if text[index:].strip():
        raise ValueError("Dane za końcem metadata.json")
    return payload, fragments


class PreviewMetadataAutosave:
    """Serialize edited rows in Tk; write immutable snapshots in submission order."""
    def __init__(self, path: Path, metadata: dict, fragments: dict[str, str]):
        self.path = Path(path)
        self.metadata = metadata
        self.fragments = dict(fragments)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pz2-metadata-save")
        self.future = None

    def submit(self, dirty_ids):
        for key in dirty_ids:
            if key in self.metadata:
                self.fragments[key] = json.dumps(self.metadata[key], ensure_ascii=False, separators=(",", ":"))
        # Import/deletion can change the set of records while the editor is open.
        for key in self.metadata:
            if key not in self.fragments:
                self.fragments[key] = json.dumps(self.metadata[key], ensure_ascii=False, separators=(",", ":"))
        snapshot = tuple((json.dumps(key, ensure_ascii=False), self.fragments[key]) for key in self.metadata)
        self.future = self.executor.submit(self._write, snapshot)
        return self.future

    def _write(self, snapshot):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=self.path.parent,
                                             prefix=self.path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write("{\n")
                for index, (key, value) in enumerate(snapshot):
                    if index:
                        stream.write(",\n")
                    stream.write(key)
                    stream.write(":")
                    stream.write(value)
                stream.write("\n}\n")
            temporary.replace(self.path)
            return self.path.stat().st_mtime
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def flush(self):
        return self.future.result() if self.future is not None else None

    def close(self):
        try:
            return self.flush()
        finally:
            self.executor.shutdown(wait=True)


def mark_preview_metadata_changed(host) -> None:
    host._preview_metadata_revision = int(getattr(host, "_preview_metadata_revision", 0) or 0) + 1
    host._preview_counts_cache = None
    host._preview_contextual_counts_cache = None
    host._preview_export_readiness_cache = None


def readiness_key(host, strategies, sources, minimum) -> tuple | None:
    # Hosts without the editor's revision protocol (e.g. batch export) compute
    # fresh results. UI edits/load/import/undo invalidate this revision.
    if not isinstance(getattr(host, "_preview_metadata_revision", None), int):
        return None
    paths = host._get_gold_export_meta_candidates()
    loaded_path = path_key(getattr(host, "_loaded_meta_path", None))
    signatures = []
    for path in paths:
        if path_key(path) == loaded_path and isinstance(getattr(host, "preview_metadata", None), dict):
            signatures.append((loaded_path, id(host.preview_metadata), host._preview_metadata_revision))
        else:
            signatures.append(file_signature(path))
    return tuple(signatures), tuple(sorted(strategies)), tuple(sorted(sources)), minimum
