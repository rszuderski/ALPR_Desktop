from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_t02_list_counter_names_review_scope_and_pool_explicitly():
    source = (
        ROOT / "auto_annotation_tool" / "gui" / "z2_preview_workflow.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def _refresh_preview_list_summary")
    end = source.index("def _parse_cvat_preview_annotations", start)
    segment = source[start:end]

    assert 'total_label_text = "Zakres kontroli"' in segment
    assert 'f"{visible_count}/{summary_total_images} · "' in segment
    assert 'f"Pula O: {len(annotations)}"' in segment
    assert 'total_label_text = "Zdjęcia"' in segment


def test_t02_canvas_overlay_says_control_scope_and_pool_o():
    source = (
        ROOT / "auto_annotation_tool" / "gui" / "z2_canvas_overlays.py"
    ).read_text(encoding="utf-8-sig")

    start = source.index("def _get_preview_legend_context")
    end = source.index("def _fit_preview_text_to_width", start)
    segment = source[start:end]

    assert 't02_review_context = bool(' in segment
    assert 'f"Kontrola: {current_image_no}/{display_total}"' in segment
    assert 'f"Pula O: {total_images}"' in segment
    assert 'else f"Zdjęcie: {current_image_no}/{display_total}"' in segment
