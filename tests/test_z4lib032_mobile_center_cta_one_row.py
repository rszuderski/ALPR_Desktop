"""Z4LIB032: stopka Centrum eksportu ma tylko jeden rzad CTA; test bez GUI i bez zapisu."""
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "auto_annotation_tool" / "gui" / "z4_model_export.py"


class MobileExportCenterFooterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8-sig")
        start = cls.source.index("# Z4LIB001: browse existing mobile artifacts")
        end = cls.source.index("    selected_iid = initial_iid", start)
        cls.footer = cls.source[start:end]

    def test_all_three_ctas_are_in_row_zero(self):
        self.assertRegex(
            self.footer,
            r'text="Biblioteka modeli mobilnych"[\s\S]*?'
            r'\.grid\(row=0, column=0, sticky="w",'
        )
        self.assertIn('export_button.grid(row=0, column=1, sticky="e"', self.footer)
        self.assertIn('close_button.grid(row=0, column=2, sticky="e"', self.footer)
        self.assertNotIn("columnspan=3", self.footer)
        self.assertNotIn(".grid(row=1,", self.footer)

    def test_existing_actions_are_unchanged(self):
        self.assertIn("command=_open_z4_mobile_artifact_library", self.footer)
        self.assertIn("command=open_selected_mobile_export_executor", self.footer)
        self.assertIn("command=dialog.destroy", self.footer)
        self.assertIn("state=tk.DISABLED", self.footer)
        self.assertIn("required_width = actions.winfo_reqwidth() + 32", self.footer)


if __name__ == "__main__":
    unittest.main()
