import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from auto_annotation_tool.model_download import DownloadCancelled, DownloadProgress, download_model
from auto_annotation_tool.gui.model_download_dialog import transfer_caption
from auto_annotation_tool.gui import z2_annotation_startup as startup


class Response(io.BytesIO):
    def __init__(self, data, size=None):
        super().__init__(data)
        self.headers = {} if size is None else {"Content-Length": str(size)}


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.target = Path(directory.name) / "vehicle.pt"
        self.cancel = threading.Event()
        self.events = []

    def download(self, response, progress=None):
        opener = Mock(return_value=response)
        result = download_model("https://example.test/model.pt", self.target, self.cancel,
                                progress or self.events.append, opener=opener)
        return result, opener

    def test_atomic_complete_file_and_actual_bytes(self):
        data = b"test-weights" * 20000
        def update(state):
            if not state.complete:
                self.assertFalse(self.target.exists())
            self.events.append(state)
        result, _ = self.download(Response(data, len(data)), update)
        self.assertEqual(result.read_bytes(), data)
        self.assertEqual(self.events[-1].percent, 100)
        self.assertEqual(self.events[-1].received, len(data))
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])

    def test_incomplete_download_is_not_a_model(self):
        with self.assertRaises(OSError):
            self.download(Response(b"partial", 1000))
        self.assertEqual(list(self.target.parent.iterdir()), [])
        self.assertFalse(any(event.complete for event in self.events))

    def test_cancel_cleans_partial_file(self):
        response = Response(b"test" * 100000, 400000)
        original_read = response.read
        def read(size):
            chunk = original_read(size)
            self.cancel.set()
            return chunk
        response.read = read
        with self.assertRaises(DownloadCancelled):
            self.download(response)
        self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_existing_model_skips_network(self):
        self.target.write_bytes(b"weights")
        _, opener = self.download(Response(b"other"))
        opener.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b"weights")

    def test_unknown_size_has_no_fake_percent(self):
        self.assertIsNone(DownloadProgress("url", 12345).percent)
        self.assertIn("rozmiar niepodany", transfer_caption(DownloadProgress("url", 12345)))
        self.download(Response(b"model"))
        self.assertEqual(self.events[-1].percent, 100)

    def test_network_failure_does_not_leave_weight_file(self):
        with self.assertRaises(OSError):
            download_model("https://example.test/model.pt", self.target, self.cancel,
                           self.events.append, opener=Mock(side_effect=OSError("offline")))
        self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_consent_required_before_splash_or_network(self):
        owner = Mock()
        items = [{"label": "YOLO", "target_path": self.target, "asset_name": "vehicle.pt"}]
        with patch.object(startup.messagebox, "askyesno", return_value=False), \
             patch("auto_annotation_tool.gui.model_download_dialog.download_models_with_splash") as splash:
            self.assertFalse(startup._confirm_and_download_missing_models(owner, items))
            splash.assert_not_called()
            owner._set_annotation_process_log_visibility.assert_not_called()

    def test_accepted_download_does_not_open_console(self):
        owner = Mock()
        items = [{"label": "YOLO", "target_path": self.target, "asset_name": "vehicle.pt"}]
        with patch.object(startup.messagebox, "askyesno", return_value=True), \
             patch("auto_annotation_tool.gui.model_download_dialog.download_models_with_splash", return_value=True) as splash:
            self.assertTrue(startup._confirm_and_download_missing_models(owner, items))
            splash.assert_called_once_with(owner, items)
            owner._set_annotation_process_log_visibility.assert_not_called()
