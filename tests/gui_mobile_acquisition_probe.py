"""Native acquisition workflow probe with isolated synthetic input and screenshots."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ctypes
import io
import json
import time
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageGrab

from test_mobile_acquisition import make_crop
from auto_annotation_tool.gui.z3_mobile_acquisition import CropAcquisitionWindow
from auto_annotation_tool.mobile_acquisition import CropReview, file_sha256, read_crop_session


def main():
    output = Path(__file__).resolve().parents[1] / "output" / "mobile_acquisition_probe" / str(time.time_ns())
    output.mkdir(parents=True)
    crop = make_crop("wi1234a")
    text = "WI1234A"
    crop["image_width"], crop["image_height"] = 420, 100
    crop["characters"] = [dict(label=char, confidence=.9, left=(22 + index * 53) / 420,
                               top=.16, right=(65 + index * 53) / 420, bottom=.83)
                          for index, char in enumerate(text)]
    plate = Image.new("RGB", (420, 100), "white")
    drawing = ImageDraw.Draw(plate)
    drawing.rectangle((2, 2, 417, 97), outline="black", width=3)
    font = ImageFont.truetype("arialbd.ttf", 58)
    for index, char in enumerate(text):
        drawing.text((22 + index * 53, 14), char, fill="black", font=font)
    image_bytes = io.BytesIO()
    plate.save(image_bytes, format="JPEG")
    source = output / "android-acquisition.zip"
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.json", json.dumps({"schema": "alpr_crop_session_v1", "session_id": "phone-session-1",
                           "started_at_ms": 1700000000000, "crops": [crop]}))
        archive.writestr(crop["image"], image_bytes.getvalue())
    before = file_sha256(source)
    errors, exported = [], []
    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda *args: errors.append(str(args))
    host = SimpleNamespace(frame=root, _get_step3_chars_root_dir=lambda: output / "chars",
                           _persist_step3_extract_state=lambda: None,
                           _reset_extract_source_inputs=lambda: None,
                           _open_detection_subtab_with_preview=lambda path, **kwargs: exported.append(path) or True)
    window = CropAcquisitionWindow(host)
    message_errors = []

    def pump():
        root.update()
        time.sleep(.015)

    def wait_for(predicate):
        deadline = time.monotonic() + 12
        while not predicate() and time.monotonic() < deadline:
            pump()
        assert predicate(), "GUI operation timed out"

    def capture(name):
        window.lift()
        for _ in range(15):
            pump()
        ancestor = ctypes.windll.user32.GetAncestor
        ancestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        ancestor.restype = ctypes.c_void_p
        hwnd = ancestor(window.winfo_id(), 2)
        ImageGrab.grab(window=hwnd).save(output / name)

    try:
        with patch("auto_annotation_tool.gui.z3_mobile_acquisition.filedialog.askopenfilename", return_value=str(source)), \
             patch("auto_annotation_tool.gui.z3_mobile_acquisition.messagebox.showerror", side_effect=lambda *args, **kwargs: message_errors.append(args)):
            window.open_file()
            wait_for(lambda: window.current is not None and not window.busy)
            assert window.current.key == "WI1234A"
            assert window.gt_var.get() == ""
            assert window.export_button.instate(["disabled"])
            assert window.source_image.size == (420, 100)
            assert len(window.canvas.find_all()) == 8
            window.gt_var.set("wi1234a")
            window.decide("draft")
            assert window.export_button.instate(["disabled"])
            window.decide("confirmed")
            assert not window.export_button.instate(["disabled"])
            capture("gallery-1040.png")
            window.geometry("760x540")
            capture("gallery-760.png")
            assert window.canvas.winfo_width() > 350
            for button in window.decision_buttons:
                assert button.winfo_rootx() + button.winfo_width() <= window.winfo_rootx() + window.winfo_width()
            window.export()
            wait_for(lambda: bool(exported) and not window.busy)
            metadata = json.loads((exported[0] / "metadata.json").read_text(encoding="utf-8"))
            assert metadata["plate_000001"]["source_expected_text"] == "WI1234A"
            sidecar = window.review.path
            window.gt_var.set("wi1234b")
            window.close()
            reopened = CropReview(read_crop_session(source), sidecar)
            assert reopened.decisions["plate_000001"] == {"gt": "WI1234B", "status": "draft"}
            assert file_sha256(source) == before
            assert not errors, errors
            assert not message_errors, message_errors
            print(f"PASS: archive validation, preview, draft/confirm, export, reopen, two window sizes; screenshots: {output}")
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
