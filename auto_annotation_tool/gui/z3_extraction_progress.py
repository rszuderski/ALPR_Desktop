"""Latest-value progress mailbox: workers never wait for a Tk progress update."""

from threading import Lock
import tkinter as tk


class ExtractionProgress:
    def __init__(self, frame, present, is_current, *, interval_ms=100):
        self.frame = frame
        self.present = present
        self.is_current = is_current
        self.interval_ms = interval_ms
        self._lock = Lock()
        self._latest = None
        self._closed = False
        self._job = None
        self._destroy_binding = frame.bind("<Destroy>", self._destroy, add="+")
        self._job = frame.after(0, self._poll)

    def publish(self, processed, total, plates, phase="crop"):
        with self._lock:
            if not self._closed:
                self._latest = (processed, total, plates, phase)

    def close(self):
        # May be called by the worker. Tk cleanup occurs in the next UI poll.
        with self._lock:
            self._closed = True
            self._latest = None

    def _poll(self):
        self._job = None
        with self._lock:
            closed, value = self._closed, self._latest
            self._latest = None
        if closed or not self.is_current():
            self.close()
            self._cleanup()
            return
        try:
            if value is not None:
                self.present(*value)
        finally:
            self._job = self.frame.after(self.interval_ms, self._poll)

    def _cleanup(self):
        if self._job is not None:
            try:
                self.frame.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        if self._destroy_binding is not None:
            try:
                self.frame.unbind("<Destroy>", self._destroy_binding)
            except tk.TclError:
                pass
            self._destroy_binding = None

    def _destroy(self, event):
        if event.widget is self.frame:
            self.close()
            self._cleanup()
