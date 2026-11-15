"""Windows-style row selection and in-memory analysis basket operations."""


def toggle_checks(browser, iids):
    """Set a mixed group on; set an entirely checked group off, once per source.

    Hidden basket items stay intact. Source identity guards are the same as for
    a single checkbox; copies cannot become additional measurements.
    """
    from .mobile_analysis import selection_identity
    checked = browser._comparison_checked
    reports = browser.report_by_iid
    ordered = getattr(browser, '_ordered_reports', list(reports.values()))
    occupied = set()
    for report in ordered:
        if browser._comparison_key(report) not in checked:
            continue
        try:
            sid, sha = selection_identity(report, browser._matching_cached_bundle(report))
            occupied.update((('session', sid), ('sha', sha)))
        except ValueError:
            pass
    targets = set()
    skipped = []
    for iid in iids:
        report = reports.get(iid)
        if report is None:
            continue
        key = browser._comparison_key(report)
        if key in targets:
            continue
        # Removing a stale selection must remain possible.
        if key in checked:
            targets.add(key)
            continue
        try:
            sid, sha = selection_identity(report, browser._matching_cached_bundle(report))
        except ValueError:
            skipped.append('brak potwierdzonej tożsamości sesji/SHA')
            continue
        ids = {('session', sid), ('sha', sha)}
        if occupied & ids:
            skipped.append('kopia tej samej sesji lub SHA')
            continue
        targets.add(key)
        occupied.update(ids)
    if targets:
        if targets <= checked:
            checked.difference_update(targets)
        else:
            checked.update(targets)
    for iid, report in reports.items():
        if browser._comparison_key(report) in checked:
            symbol = '☑'
        else:
            try:
                selection_identity(report, browser._matching_cached_bundle(report))
                symbol = '☐'
            except ValueError:
                symbol = '—'
        browser.report_table.tree.set(iid, 'Wybor', symbol)
    browser._update_comparison_basket_status()
    if skipped:
        browser._set_status(f'Koszyk: {len(checked)}. Pominięte wiersze: {len(skipped)} — '
                            + '; '.join(dict.fromkeys(skipped)) + '.', 'warning')
    return 'break'


class ReportListSelection:
    """Keep a range anchor distinct from the row used by the single preview."""

    def __init__(self, browser):
        self.browser = browser
        self.tree = browser.report_table.tree
        self.anchor = None
        self.tree.configure(selectmode='extended')
        self.tree.bind('<Button-1>', self.click)
        self.tree.bind('<space>', self.space)
        for key in ('a', 'A'):
            self.tree.bind('<Control-'+key+'>', self.select_all)
        for key in ('Up', 'Down', 'Home', 'End'):
            self.tree.bind('<'+key+'>', lambda event, k=key: self.navigate(event, k))

    def choose(self, iid, *, control=False, shift=False):
        rows = self.tree.get_children('')
        focus = self.tree.focus()
        if shift:
            anchor = self.anchor if self.anchor in rows else focus if focus in rows else iid
            self.anchor = anchor
            start, end = sorted((rows.index(anchor), rows.index(iid)))
            group = rows[start:end+1]
            self.tree.focus(iid)
            if control:
                self.tree.selection_add(group)
            else:
                self.tree.selection_set(group)
        else:
            self.anchor = iid
            self.tree.focus(iid)
            if control:
                self.tree.selection_toggle(iid)
            else:
                self.tree.selection_set(iid)
        self.tree.see(iid)

    def click(self, event):
        if self.tree.identify_region(event.x, event.y) != 'cell':
            return None  # retain native column resizing and heading commands
        iid = self.tree.identify_row(event.y)
        if not iid:
            return None
        control, shift = bool(event.state & 4), bool(event.state & 1)
        self.tree.focus_set()
        if self.tree.identify_column(event.x) == '#1' and not (control or shift):
            return self.browser._toggle_comparison_checkbox(event)
        self.choose(iid, control=control, shift=shift)
        return 'break'

    def space(self, _event=None):
        selected = set(self.tree.selection())
        # Process in visible order, never by stale indices or report captions.
        return toggle_checks(self.browser, [iid for iid in self.tree.get_children('') if iid in selected])

    def select_all(self, _event=None):
        rows = self.tree.get_children('')
        if rows:
            if self.tree.focus() not in rows:
                self.tree.focus(rows[0])
            self.anchor = self.tree.focus()
            self.tree.selection_set(rows)
        return 'break'

    def navigate(self, event, key):
        rows = self.tree.get_children('')
        if not rows:
            return 'break'
        focus = self.tree.focus()
        index = rows.index(focus) if focus in rows else 0
        index = (0 if key == 'Home' else len(rows)-1 if key == 'End'
                 else max(0, min(len(rows)-1, index + (1 if key == 'Down' else -1))))
        control, shift = bool(event.state & 4), bool(event.state & 1)
        if control and not shift:
            self.tree.focus(rows[index]); self.tree.see(rows[index])
        else:
            self.choose(rows[index], control=control, shift=shift)
        return 'break'
