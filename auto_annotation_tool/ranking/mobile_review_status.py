"""Read-only captions: resolved images do not imply session approval."""


def approval_caption(inventory, review_status, revision):
    if inventory.get('operator_review_progress'):
        from .mobile_operator_progress import presentation
        source = inventory['operator_review_progress']
        p = presentation(source)
        caption = f"Weryfikacja operatora: {p['status']} · {p['progress']}"
        if source.get('unit_kind') == 'MT_ROI' and source.get('mt_assessed') is not None:
            caption += f" · osobno MT/ROI: {source['mt_assessed']}/{source['mt_total']}"
        return caption
    total = inventory.get('total', 0)
    if inventory.get('scene_scope'):
        scene=inventory['scene_scope']
        return (f"Pełne sceny: {scene['n_assessed']}/{scene['n_scene_events']} ocenione · N/A źródła: {scene['n_missing']}"
                f" · MT/ROI z oceną: {inventory.get('operator_assessed',0)}/{total} · TV E2E: N/A")
    if inventory.get('experiment_family') == 'F5A':
        assessed = inventory.get('operator_assessed', inventory.get('assessed', 0))
        na = inventory.get('source_na', 0) + inventory.get('no_reference', 0)
        caption = f'MT/ROI: zapisane oceny {assessed}/{total} · N/A: {na}'
        if review_status == 'COMPLETED':
            suffix = f' · zatwierdzona rewizja r{revision}'
            return caption + suffix + ('; zakres częściowy' if assessed < total else '')
        return caption + f" · pozostałe decyzje ramek: {inventory.get('decisions', 0)}"
    ready = inventory.get('inspected', inventory.get('assessed', 0))
    remaining = inventory.get('remaining', total)
    prefix = f'{ready}/{total} sprawdzone'
    scope=inventory.get('source_na',0)+inventory.get('no_reference',0)
    if scope:prefix+=f' · zakres N/A: {scope}'
    if total and not remaining:
        if inventory.get('approval_remaining',0):
            return prefix + ' — uzupełnij GTZ/dane do zatwierdzenia'
        if review_status == 'COMPLETED':
            return prefix + f' — zatwierdzona rewizja r{revision}'
        return prefix + ' — oczekuje na zatwierdzenie'
    return prefix + f' — do uzupełnienia: {remaining}'
