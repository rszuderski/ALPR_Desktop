"""Z4LIB033: stale widoczne narzedzia w sekcji kontroli eksportu."""
from pathlib import Path
import ast
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'auto_annotation_tool/gui/z4_model_export.py'

class Section4ActionVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = SOURCE.read_text(encoding='utf-8-sig')
        start = cls.raw.index('        checks_shell = tk.Frame(')
        end = cls.raw.index('        status_shell, exec_status_label, exec_progress_row', start)
        cls.section = cls.raw[start:end]

    def test_01_controls_outside_scroll_area_and_full_width_checks_tab(self):
        self.assertIn('verify_section.grid(row=3, column=0, sticky="ew", pady=(8, 0))', self.section)
        self.assertIn('executor_requirements_table["shell"].grid(row=2, column=0, sticky="nsew")', self.section)
        self.assertIn('executor_install_table["shell"].grid(row=2, column=0, sticky="nsew")', self.section)
        self.assertIn('checks_shell.grid_rowconfigure(2, weight=1, minsize=160)', self.section)
        self.assertNotIn('verify_section.grid(row=2', self.section)
        self.assertIn('checks_parent', self.section)

    def test_02_both_tables_have_compact_scrollable_viewport(self):
        self.assertEqual(self.section.count('height=260,'), 2)
        self.assertNotIn('height=340,', self.section)
        self.assertIn('executor_requirements_table["shell"].grid_remove()', self.section)
        self.assertIn('executor_install_table["shell"].grid_remove()', self.section)
        self.assertIn('def _show_executor_table(mode: str)', self.section)

    def test_03_both_buttons_keep_original_callbacks(self):
        self.assertIn('text="Sprawdź precyzję TFLite"', self.section)
        self.assertIn('command=_inspect_last_or_selected_tflite', self.section)
        self.assertIn('text="Raporty z telefonu"', self.section)
        self.assertIn('command=lambda: open_mobile_report_browser(self, parent=executor_modal)', self.section)

    def test_04_preflight_action_and_syntax_unaffected(self):
        self.assertIn('text="Sprawdź gotowość eksportu"', self.raw)
        self.assertIn('def run_executor_preflight(', self.raw)
        ast.parse(self.raw, filename=str(SOURCE))

if __name__ == '__main__':
    unittest.main()
