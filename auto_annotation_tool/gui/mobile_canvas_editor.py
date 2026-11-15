"""P2B2 canvas-first review controller; Z2-style drawing and inline GT.

The existing review window remains responsible for image loading, worker tasks,
zoom, session identity and exports. This controller owns interaction on the
verified source Bitmap only. Geometry persists in source pixels, never in Tk.
"""
from __future__ import annotations

import copy
import math
import sys
import time
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, font as tkfont

from ..ranking import mobile_canvas_review as core
from ..ranking.mobile_source_frame_roi import quad_contains_point
from ..ranking.mobile_pose_metrics import validate_mt_quad
from ..ranking.mobile_canvas_presentation import build_inventory, subject_groups
from .mobile_canvas_ui import palette, label as ui_label, MetricCard, Disclosure
from .mobile_canvas_results_panel import CanvasResultsPanel

TAG = "p2b2_canvas_overlay"


def overlay_label_position(bounds, size, viewport, obstacles, *, above=False):
    """Find free screen space without placing text over any MT/GT polygon box."""
    l, t, r, b = bounds
    w, h = size
    vw, vh = viewport
    if w + 16 > vw or h + 16 > vh:
        return None
    x = (l + r - w) / 2 if above else l
    preferred = [(x, t-h-16), (x, b+16)] if above else [(x, b+16), (x, t-h-16)]
    preferred += [(r+16, t), (l-w-16, t), (8, 8), (vw-w-8, 8),
                  (8, vh-h-8), (vw-w-8, vh-h-8)]
    preferred += [(x, 8 + i * (vh-h-16) / 8) for i in range(9)]
    for px, py in preferred:
        px, py = max(8, min(vw-w-8, px)), max(8, min(vh-h-8, py))
        if not any(px < q[2]+8 and px+w > q[0]-8 and py < q[3]+8 and py+h > q[1]-8
                   for q in obstacles):
            return px, py
    return None


def reconciliation_story(scene, tally):
    """Human-facing explanation; does not infer physical plate count from NMS.

    The trace describes the *phone's* post-filter output. Counting all physical
    plates in the ROI remains an independent operator task.
    """
    raw = int(tally.get("mt", 0))
    suppressed = int(tally.get("raw_suppressed", 0))
    # Pending historical scenes still store both objects as active: counts()
    # cannot report the post-filter total until the explicit migration.
    pending = bool(scene.get("_post_nms_pending"))
    final = raw - (1 if pending else suppressed)
    if raw > 2 and (pending or suppressed):
        aid=scene.get('_post_nms_pending') or next((o['attempt_id'] for o in scene['objects'] if o.get('post_nms_suppressed')),None)
        parent=next((p for p in scene['parents'] if p['attempt_id']==aid),None)
        label=f"MT {int(parent['detection_index'])+1}" if parent else 'Dodatkowy kandydat'
        from ..ranking.mobile_candidate_stacks import physical_count_view
        physical_n,known=physical_count_view(scene)
        physical_text=str(physical_n) if known else 'do ustalenia'
        return dict(title='CO ZAPISAŁ TELEFON?',
            body=f'Wskazania: {raw} surowe → {final} końcowe.\n{label}: odrzucony przez filtr; bez osobnej tablicy i bez końcowego błędu.\nTablice na zdjęciu: {physical_text}.',
            action=f'Uwzględnij wynik telefonu: {raw} → {final}' if pending else None,
            state='pending' if pending else 'resolved')
    if pending:
        return {
            "title": "DLACZEGO PROGRAM PROSI O POPRAWIENIE OCENY?",
            "body": (
                f"1. TELEFON: na jednej klatce wskazał {raw} możliwe obszary "
                f"tablicy. Po usunięciu nakładających się wskazań "
                f"przekazał {final} wynik do rozpoznawania.\n\n"
                "2. WCZEŚNIEJSZA OCENA: powstała zanim uwzględniono ten "
                "końcowy wynik. Oba wskazania policzono jako dwie tablice. "
                "Stąd niezgodność w liczniku.\n\n"
                "3. PROPONOWANA ZMIANA: liczyć tylko wynik, którego użył "
                "telefon. Drugie wskazanie, jego GT i poprawki zostaną "
                "zachowane w historii — nie uznamy go za fałszywy alarm.\n\n"
                "4. TWOJE ZADANIE PO ZMIANIE: obejrzeć cały zaznaczony "
                "obszar i samodzielnie policzyć rzeczywiste tablice. "
                "Dodatkową tablicę, której telefon nie wykrył, dodaj "
                "klawiszem D. Sprawdź ramkę i numer GT; dopiero potem "
                "potwierdź sprawdzenie obrazu."
            ),
            "action": "Popraw wcześniejszą ocenę: 2 → 1 wynik",
            "state": "pending",
        }
    if suppressed:
        return {
            "title": "WYNIK TELEFONU UWZGLĘDNIONY — CO DALEJ?",
            "body": (
                f"Telefon zapisał {raw} wstępne wskazania, ale do "
                f"rozpoznawania przekazał {final}. "
                f"Pozostałe {suppressed} zachowano w historii i nie "
                "liczymy ich jako osobnych tablic ani fałszywych alarmów.\n\n"
                "TERAZ TWOJA DECYZJA: obejrzyj cały analizowany obszar. "
                "Czy wszystkie rzeczywiste tablice są oznaczone? "
                "Jeśli brakuje tablicy, dodaj ją (D). Sprawdź narożniki "
                "i numer GT. Dopiero potem zaznacz sprawdzenie obrazu. "
                "Sam filtr telefonu nie ustala liczby tablic w rzeczywistości."
            ),
            "action": None,
            "state": "resolved",
        }
    return None


def linked_measurement_rows(session, subject_ids, available_group_ids):
    """Return rows that merely repeat the *same executed MT invocation*.

    A subject becomes an informational child only when it has exactly one
    inspectable invocation, and that invocation already belongs to the first
    subject in the ordered observation register. Never group by text, GT,
    spatial proximity, detection count, or approximate timestamps.

    The return value maps existing Treeview subject iids to (owner_iid, gid).
    It does NOT alter subjects, invocation IDs, source images or review states.
    """
    subjects = list(subject_ids.items())
    inspected = {
        key: [gid for gid in subject_groups(session, key) if gid in available_group_ids]
        for _, key in subjects
    }
    linked = {}
    for iid, key in subjects:
        gids = inspected[key]
        if len(gids) != 1:
            continue  # Multiple regions or missing evidence require own row.
        gid = gids[0]
        group = session.mt_invocations.get(gid)
        if group is None or key not in group.subject_keys:
            continue  # Inconsistent links must remain visible for manual review.
        owner = next((owner_iid for owner_iid, owner_key in subjects
                      if owner_key in group.subject_keys and gid in inspected[owner_key]), None)
        if owner is not None and iid != owner:
            linked[iid] = (owner, gid)
    return linked


class CanvasReviewEditor:
    def __init__(self, host, right, review, viewer, *, reference_source=None):
        self.reference_source = reference_source
        self.h = host
        self.c = host.canvas
        self.selected = ""
        self._bad_mt_selection = None
        self.pointer_canvas = None
        self.selection_key_down = False
        self.selection_release_after = None
        self._s_hover_hit = None  # (group_id, object_id, canvas_xy) from real Motion
        self._win_s_poll_id = None
        self._win_s_down = False
        self._win_user32 = None
        self.fit_key_down = False
        self.fit_release_after = None
        self.shortcut_tag = None
        self.shortcut_original_tags = {}
        self.pointer_canvas_at = None
        self.tree_branch = ""
        self.mode = "select"
        self.points = []
        self.draw_reference_target = None
        self.draw_target = None
        self.pointer = None
        self.drag = None
        self._click_candidate = None  # click selects; drag pans unless a corner is grabbed
        self.show_roi_overlay = True
        self.plate_fit_saved = None
        self.plate_fit_active = False
        self.plate_fit_gid = ""
        self.plate_fit_previous = None  # (zoom, center) before A; view-only
        self.hover = ""
        self.skey = ""
        self.cached = None
        self.cache_key = None
        self.badges = {}
        self.card_positions = {}
        self.entry = None
        self.entry_var = None
        self.entry_target = None
        self.entry_original = ""
        self.entry_after = None
        self.refresh_after = None
        self.task_inventory_cache = {}
        self.paint_scheduled = False
        self.paint_after = None
        self.in_refresh = False
        self.busy = False
        self.undo_stack = {}
        self.redo_stack = {}
        # Original photographs use the same gestures, geometry renderer and GT
        # entry, with a source/save adapter and no mobile review panels.
        if reference_source is not None:
            reference_source.install(self, viewer)
            return
        # Keep old widgets alive for report readers, but remove their form UI.
        self.old_right = list(right.winfo_children())
        for widget in self.old_right:
            widget.grid_remove()
        # Give the inspector meaningful space without stealing the whole image.
        # Use the actual screen width, not the 238 px legacy Canvas override.
        screen_width = host.window.winfo_screenwidth()
        self.sidebar_width = (398 if screen_width >= 1500 else
                              358 if screen_width >= 1250 else 318)
        # The operator's left panel lists images only. Technical records,
        # candidate boxes and source IDs belong to the audit/report view.
        self.left_width = 310
        review.columnconfigure(0, weight=0, minsize=self.left_width)
        review.columnconfigure(1, weight=1, minsize=430 if screen_width >= 1250 else 350)
        review.columnconfigure(2, weight=0, minsize=self.sidebar_width + 12)
        host.window.minsize(min(1150, max(900, screen_width - 60)), 670)
        host.review_sidebar_scroll.configure(width=self.sidebar_width)
        self.p = palette(host.window, host.bg)
        from .mobile_canvas_collision_warning import CollisionWarning
        self.collision_warning = CollisionWarning(self)
        from .mobile_canvas_corner_pulse import CornerPulse
        self.corner_pulse = CornerPulse(self)
        from .mobile_canvas_shift_preview import ShiftPreview
        self.shift_preview = ShiftPreview(self)
        self._badge_font = tkfont.Font(root=host.window, family="Segoe UI", size=9, weight="bold")
        self.tree_targets = {}
        self.linked_tree_rows = {}  # hidden aliases, used only for navigation
        self.pending_tree_target = None
        self.inventory_key = None
        self.inventory_data = None
        self.tree_refreshing = False
        self._active_tree_iid = ""
        self._build_panel(right)
        review.bind("<Configure>", self._fit_review_columns, add="+")
        self._configure_tree()
        # The former binding pass ran BEFORE this panel existed.
        host._bind_sidebar_wheel(self.panel)
        host.review_sidebar_scroll.configure(yscrollincrement=22)
        # Keep toolbar objects alive for existing configure() callers, but
        # let Canvas occupy all their space. V retains the source-view action.
        for row in (0, 1, 2, 5):
            for widget in viewer.grid_slaves(row=row):
                widget.grid_remove()
        host.target_nav.grid_remove()
        host.image_help.configure(text="Turkusowa ramka: wynik telefonu • pomarańczowo-żółta: korekta operatora • czerwone ×: błędne wskazanie • półprzezroczysty obszar: miejsce analizy")
        self.bindings()
        from .mobile_canvas_clean_view import install
        install(self,viewer)

    def _fit_review_columns(self, event):
        """Keep the inspector and its scrollbar inside a narrow window."""
        width = event.width
        left = min(self.left_width, max(250, int(width * .21)))
        right = min(self.sidebar_width, max(280, int(width * .27)))
        layout = (left, right)
        if getattr(self, "_review_columns", None) == layout:
            return
        self._review_columns = layout
        event.widget.columnconfigure(0, minsize=left)
        event.widget.columnconfigure(1, minsize=300)
        event.widget.columnconfigure(2, minsize=right + 12)
        # A minsize alone cannot shrink a Treeview's requested column width.
        self.h.subject_tree.column("#0", width=left - 38, minwidth=200)
        scrollbar_width=getattr(self,'_sidebar_bar_width',None)
        self.h.review_sidebar_scroll.configure(width=max(1,right+12-scrollbar_width) if scrollbar_width is not None else right)
        self.approve.configure(wraplength=right - 35)
        self.exclude.configure(wraplength=right - 35)
        for item in (self.status, self.exclude_hint, self.tally, self.selection,
                     self.selection_hint, self.metrics, self.reconciliation_body):
            item.configure(wraplength=right - 42)

    def label(self, text, bold=False, small=False):
        return tk.Label(self.panel, text=text, anchor="w", justify="left",
            wraplength=self.sidebar_width - 44,
            bg=self.h.bg, fg=self.h.fg, font=("Segoe UI", 9 if small else 10, "bold" if bold else "normal"))

    @staticmethod
    def make_action_button(parent, text, command, *, primary=False, compact=False, **kwargs):
        """A real button, not a flat heading; works with native Windows Tk themes.

        Applies only to review actions. No global ttk/theme override and no
        bindings or data writes. Tk Button owns hover/pressed/disabled states.
        """
        window = parent.winfo_toplevel()
        p = palette(window, window.cget("bg"))
        base = p["blue"] if primary else p["surface"]

        def luminance(color):
            channels = (value / 65535.0 for value in window.winfo_rgb(color))
            linear = [value / 12.92 if value <= .04045 else
                      ((value + .055) / 1.055) ** 2.4 for value in channels]
            return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]

        def blend(color_a, color_b, amount):
            aa = window.winfo_rgb(color_a)
            bb = window.winfo_rgb(color_b)
            values = [round((1-amount)*a/257 + amount*b/257) for a,b in zip(aa,bb)]
            return "#{:02x}{:02x}{:02x}".format(*values)

        # WCAG contrast prefers white below this background luminance.
        foreground = ("#ffffff" if luminance(base) < .179 else "#111827") if primary else p["fg"]
        if primary:
            hover = blend(base, "#ffffff" if luminance(base) < .45 else "#000000", .16)
        else:
            hover = blend(base, p["blue"], .15)
        return tk.Button(parent, text=text, command=command,
            bg=base, fg=foreground, activebackground=hover,
            activeforeground=foreground, disabledforeground=p["muted"],
            font=("Segoe UI", 9 if compact else 10, "bold"),
            padx=8 if compact else 13, pady=4 if compact else 8,
            relief="raised", overrelief="ridge", bd=2,
            highlightthickness=1, highlightbackground=p["border"],
            highlightcolor=p["blue"], cursor="hand2", takefocus=True,
            justify="center", **kwargs)

    def bindings(self):
        self.c.bind("<ButtonPress-1>", self.press)
        self.c.bind("<B1-Motion>", self.motion)
        self.c.bind("<ButtonRelease-1>", self.release)
        self.c.bind("<ButtonPress-3>", self.right_press)
        self.c.bind("<B3-Motion>", self.pan_motion)
        self.c.bind("<ButtonRelease-3>", self.pan_release)
        self.c.unbind("<Control-ButtonPress-3>")
        self.c.bind("<Motion>", self.hover_motion)
        self.c.bind("<Double-Button-1>", self.double_press)
        for key in ('Up','Down'):
            self.c.bind('<KeyPress-'+key+'>',self.stack_arrow_hotkey)
        for k in ("d","D"):
            self.c.bind("<KeyPress-"+k+">", self.draw_toggle)
        # Register shortcuts on the actual review canvas and its early bindtag.
        # Z2 also registers a guarded global fallback for focusable widgets.
        # Only our own window may react; editing text remains untouched.
        self.c.configure(takefocus=1)
        self.install_canvas_shortcuts()
        for k in ("n", "N"):
            self.c.bind("<KeyPress-" + k + ">", self.cycle_hotkey)
        for k in ("t", "T"):
            self.c.bind("<KeyPress-" + k + ">", self.roi_hotkey)
        for k in ("f","F"):
            self.c.bind("<KeyPress-"+k+">", lambda e:self.fit())
        for k in ("v", "V"):
            self.c.bind("<KeyPress-"+k+">", self.source_view_hotkey)
        for k in ("q","Q"):
            self.c.bind("<KeyPress-"+k+">", lambda e:self.navigate(-1))
        for k in ("e","E"):
            self.c.bind("<KeyPress-"+k+">", lambda e:self.navigate(1))
        self.c.bind("<space>", self.confirm)
        self.c.bind("<KeyRelease-space>", self.release_frame_space)
        self.c.bind("<FocusOut>", self.release_frame_space, add='+')
        self.c.bind("<Control-s>", self.confirm)
        self.c.bind("<Control-S>", self.confirm)
        self.c.bind("<Control-z>", lambda e:self.undo())
        self.c.bind("<Control-Z>", lambda e:self.undo())
        self.c.bind("<Control-y>", lambda e:self.undo(redo=True))
        self.c.bind("<Control-Y>", lambda e:self.undo(redo=True))
        for k in ("r", "R"):
            self.c.bind("<KeyPress-" + k + ">", self.remove_hotkey)
        self.c.bind("<Shift-R>", self.remove_selected)
        self.c.bind("<Shift-r>", self.remove_selected)
        self.c.bind("<Leave>", self.leave_canvas)
        # Delete/Backspace stay safe; removal requires selection mode + R.
        self.c.bind("<Delete>", self.no_delete)
        self.c.bind("<BackSpace>", self.back_point)
        self.c.bind("<Escape>", self.escape)
        self.h.window.bind("<Escape>", self.escape)
        self.c.bind("<Enter>", self.focus_canvas)
        # Windows fallback: Tk sometimes fails to deliver S to the review widget
        # while the mouse is on canvas. Poll only the S key, only while THIS
        # window is foreground, and only while the cursor is on the image.
        self._start_windows_s_watch()

    def focus_canvas(self, event=None):
        if self.entry is None:
            self.c.focus_set()

    def source_view_hotkey(self, event=None):
        if event is None or not getattr(event, "state", 0) & 4:
            self.h._toggle_source_frame_view()
        return "break"

    def _start_windows_s_watch(self):
        if sys.platform != "win32" or getattr(self, "_win_s_poll_id", None) is not None:
            return
        self._win_s_poll_id = self.c.after(35, self._windows_s_watch_tick)

    def _windows_s_state(self):
        """(review_window_is_foreground, S_is_down). Never inspect other keys.

        Tk's focus and OS foreground are separate on Windows. A physical-key
        fallback must not affect a different application or a GT text field.
        """
        if sys.platform != "win32":
            return False, False
        try:
            import ctypes
            from ctypes import wintypes
            api = getattr(self, "_win_user32", None)
            if api is None:
                api = ctypes.WinDLL("user32", use_last_error=True)
                api.GetForegroundWindow.restype = wintypes.HWND
                api.GetAncestor.argtypes = (wintypes.HWND, ctypes.c_uint)
                api.GetAncestor.restype = wintypes.HWND
                api.GetAsyncKeyState.argtypes = (ctypes.c_int,)
                api.GetAsyncKeyState.restype = ctypes.c_short
                self._win_user32 = api
            active = api.GetForegroundWindow()
            parent = int(self.h.window.winfo_id())
            # GA_ROOT=2 collapses Tk's native top-level wrapper on Windows.
            same = bool(active and parent and
                        api.GetAncestor(active, 2) == api.GetAncestor(parent, 2))
            return same, bool(api.GetAsyncKeyState(0x53) & 0x8000) if same else False
        except (AttributeError, OSError, TypeError, ValueError, tk.TclError):
            return False, False

    def _windows_s_watch_tick(self, *, schedule=True):
        """Use a physical press edge if no Tk S event arrived.

        We never react outside the active review window, in GT input fields,
        or with the mouse outside the canvas. Existing Tk bindings remain the
        primary path; `selection_key_down` prevents a duplicate toggle.
        """
        self._win_s_poll_id = None
        try:
            if getattr(self.h, "_closed", False):
                return
            active, down = self._windows_s_state()
            was_down = getattr(self, "_win_s_down", False)
            if active and down and not was_down:
                try:
                    focus = self.h.window.focus_get()
                except (AttributeError, tk.TclError):
                    focus = None
                if (not self._is_typing_shortcut(focus)
                        and self._canvas_hotkey_pointer(None) is not None):
                    self.selection_key_press(SimpleNamespace(widget=self.c, state=0))
            elif not down and was_down:
                self.reset_selection_key()
            self._win_s_down = bool(active and down)
        finally:
            if schedule and not getattr(self.h, "_closed", False):
                self._win_s_poll_id = self.c.after(35, self._windows_s_watch_tick)

    def install_canvas_shortcuts(self):
        """Use an early Tk bindtag: Treeview class bindings may stop key events.

        This tag is scoped to the review Toplevel, leaves GT text fields and
        other windows alone and prevents one keypress being handled twice.
        """
        if getattr(self, "shortcut_tag", None):
            self._tag_canvas_shortcut_widgets()
            return
        tag = "ALPRCanvasKey_" + str(id(self))
        self.shortcut_tag = tag
        self.shortcut_original_tags = {}
        for key in ("s", "S"):
            self.c.bind_class(tag, "<KeyPress-" + key + ">", self.selection_key_press)
            self.c.bind_class(tag, "<KeyRelease-" + key + ">", self.selection_key_release)
        for key in ("a", "A"):
            self.c.bind_class(tag, "<KeyPress-" + key + ">", self.fit_key_press)
            self.c.bind_class(tag, "<KeyRelease-" + key + ">", self.fit_key_release)
        for key, handler in (("r", self.remove_hotkey), ("R", self.remove_hotkey),
                             ("t", self.roi_hotkey), ("T", self.roi_hotkey),
                             ("n", self.cycle_hotkey), ("N", self.cycle_hotkey),
                             ("z", self.stack_hotkey), ("Z", self.stack_hotkey),
                             ("x", self.stack_hotkey), ("X", self.stack_hotkey),
                             ("Up", self.stack_arrow_hotkey), ("Down", self.stack_arrow_hotkey)):
            self.c.bind_class(tag, "<KeyPress-" + key + ">", handler)
        self.c.bind_class(tag, '<KeyPress-space>', self.confirm)
        self.c.bind_class(tag, '<KeyRelease-space>', self.release_frame_space)
        self.c.bind_class(tag, '<FocusOut>', self.release_frame_space, add='+')
        for key in ('z','Z','y','Y'):
            self.c.bind_class(tag, '<Control-KeyPress-'+key+'>', self.history_hotkey)
        self._tag_canvas_shortcut_widgets()
        # A newly created focusable widget (e.g. a button in an expanded
        # panel) gets the same tag when it first receives focus.
        self.h.window.bind("<FocusIn>", self._tag_canvas_focus_widget, add="+")
        # Fallback borrowed from working Z2: events dispatched by a nested
        # ttk widget or a later-created focus owner still reach this window.
        # The first bindtag wins when available; it returns "break", so no
        # physical key press is handled twice.
        self._install_shortcut_fallback()

    def _event_in_review_window(self, event):
        try:
            return event.widget.winfo_toplevel() is self.h.window
        except (AttributeError, tk.TclError):
            return False

    def _install_shortcut_fallback(self):
        """Bind guarded S/A fallback through Tk `all`, as used by Z2.

        Preserve existing global handlers (Z2 and other modules); dispose()
        removes only our Tcl callback scripts, never the whole sequence.
        """
        self.shortcut_global_bindings = []
        for key in ("s", "S", "a", "A"):
            press = self.selection_key_press if key.lower() == "s" else self.fit_key_press
            release = self.selection_key_release if key.lower() == "s" else self.fit_key_release
            for phase, handler in (("KeyPress", press), ("KeyRelease", release)):
                sequence = "<" + phase + "-" + key + ">"
                command_id = self.c.bind_all(sequence, handler, add="+")
                self.shortcut_global_bindings.append((sequence, command_id))

    def _remove_shortcut_fallback(self):
        # Tkinter's unbind_all would erase unrelated Z2 bindings. Remove only
        # scripts installed by this editor, identified by their callback IDs.
        for sequence, command_id in getattr(self, "shortcut_global_bindings", ()):
            try:
                script = str(self.c.tk.call("bind", "all", sequence))
                lines = script.splitlines(keepends=True)
                cleaned = "".join(line for line in lines if command_id not in line)
                if cleaned != script:
                    self.c.tk.call("bind", "all", sequence, cleaned)
                # bind_all registers Tcl callbacks on Tk's ROOT widget.
                # Remove them from its destruction registry as well, otherwise
                # Tk.destroy() attempts to delete a command twice.
                owner = self.c._root()
                owner.deletecommand(command_id)
                callbacks = getattr(owner, "_tclCommands", None)
                if callbacks is not None and command_id in callbacks:
                    callbacks.remove(command_id)
            except (AttributeError, tk.TclError):
                pass
        self.shortcut_global_bindings = []

    def _tag_canvas_focus_widget(self, event=None):
        widget = getattr(event, "widget", None)
        if widget is not None:
            self._tag_canvas_widget(widget)

    def _tag_canvas_widget(self, widget):
        tag = getattr(self, "shortcut_tag", None)
        if not tag:
            return
        try:
            if widget.winfo_toplevel() is not self.h.window:
                return
            tags = widget.bindtags()
            if tag not in tags:
                self.shortcut_original_tags[widget] = tags
                widget.bindtags((tag,) + tags)
        except (AttributeError, tk.TclError):
            return

    def _tag_canvas_shortcut_widgets(self):
        todo = [self.h.window]
        seen = set()
        while todo:
            widget = todo.pop()
            if id(widget) in seen:
                continue
            seen.add(id(widget))
            self._tag_canvas_widget(widget)
            try:
                todo.extend(widget.winfo_children())
            except (AttributeError, tk.TclError):
                pass

    def _is_typing_shortcut(self, widget):
        """Never hijack S/A while typing an operator GT or a reason."""
        return isinstance(widget, (tk.Entry, tk.Text, tk.Spinbox,
                                   ttk.Entry, ttk.Combobox, ttk.Spinbox))

    def fit_key_press(self, event):
        if getattr(event, "widget", None) is not self.c and not self._event_in_review_window(event):
            return None
        if self._is_typing_shortcut(getattr(event, "widget", None)):
            return None
        # A cycles the current image's plates also from the image list.
        # S remains a true hover-only selection shortcut.
        scene = self._read()
        if not scene:
            return None
        token = getattr(self, "fit_release_after", None)
        if token:
            self.c.after_cancel(token)
            self.fit_release_after = None
        if getattr(self, "fit_key_down", False):
            return "break"
        self.fit_key_down = True
        return self.fit_plate_toggle(event)

    def fit_key_release(self, event=None):
        if event is not None and getattr(event, "widget", None) is not self.c and not self._event_in_review_window(event):
            return None
        token = getattr(self, "fit_release_after", None)
        if token:
            self.c.after_cancel(token)
        self.fit_release_after = self.c.after_idle(self.reset_fit_key)
        return "break"

    def reset_fit_key(self, event=None):
        token = getattr(self, "fit_release_after", None)
        if token:
            try:
                self.c.after_cancel(token)
            except (AttributeError, tk.TclError):
                pass
        self.fit_release_after = None
        self.fit_key_down = False

    def _canvas_action_key_allowed(self, event):
        """Canvas-only hotkeys, independent of keyboard focus within review.

        The S/A early bindtag catches keypresses even with Treeview focused.
        R/T/N share that dispatch but never steal typing in GT fields.
        Windows Mod1 (Tk state=8) is not Ctrl and must be permitted.
        """
        if event is None:
            return True
        widget = getattr(event, "widget", None)
        if widget is not self.c and not self._event_in_review_window(event):
            return False
        if self._is_typing_shortcut(widget) or getattr(event, "state", 0) & 4:
            return False
        return self._canvas_hotkey_pointer(event) is not None

    def remove_hotkey(self, event=None):
        if not self._canvas_action_key_allowed(event):
            return None
        return self.remove_selected(event)

    def roi_hotkey(self, event=None):
        if not self._canvas_action_key_allowed(event):
            return None
        return self.toggle_roi_overlay(event)

    def cycle_hotkey(self, event=None):
        if not self._canvas_action_key_allowed(event):
            return None
        return self.cycle(event)

    def stack_hotkey(self, event):
        if (not self._event_in_review_window(event)
                or self._is_typing_shortcut(event.widget)
                or getattr(event, 'state', 0) & 4):
            return None
        from .mobile_canvas_navigator import cycle_layer,scope
        return cycle_layer(self,back=event.keysym.lower()=='z') if scope(self,event.widget) else None

    def history_hotkey(self,event):
        if not self._event_in_review_window(event) or self._is_typing_shortcut(event.widget):return None
        return self.undo(redo=event.keysym.lower()=='y')

    def stack_arrow_hotkey(self,event):
        # Image-list arrows and Entry navigation retain their own meaning.
        from .mobile_canvas_navigator import cycle_result,scope
        if not scope(self,event.widget) or getattr(event,'state',0)&5:return None
        return cycle_result(self,back=event.keysym=='Up')

    def no_delete(self, event=None):
        self.message("Najazd + S zaznacza. R usuwa ramkę ręczną lub cofa korektę MT. T pokazuje/ukrywa ROI. PPM jednym kliknięciem zmienia flagę błędu.")
        return "break"

    def message(self, text):
        if getattr(self, 'reference_source', None) is not None:
            if self.reference_source.error:
                return  # an ordinary hint must not cover a failed save
            if text.startswith('Edytujesz GTZ'):
                text='Wpisz numer lub fragment · Enter zapisuje · Esc cofa wpis · PPM: opcje numeru i geometrii'
        self.h.status_var.set(text)

    def _read(self):
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.scene()
        h = self.h
        review = getattr(h.session, 'review', None)
        key = (id(h.session), getattr(h.session, 'session_id', None),
               getattr(review, 'source_archive_sha256', None), h.current_invocation,
               getattr(review, 'review_revision', None))
        if self.cache_key != key:
            self.cache_key = key
            try:
                self.cached = core.scene(h.session, h.current_invocation, include_history=False) if h.session and h.current_invocation else None
                self.read_error = ""
            except (ValueError, KeyError, TypeError) as e:
                self.cached = None
                self.read_error = str(e)
        return self.cached

    def sync(self):
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.sync()
        from ..ranking.mobile_canvas_presentation import execution_state, focused_assessment, operator_context, review_progress
        h = self.h
        for widget in self.old_right:
            widget.grid_remove()
        h.target_nav.grid_remove()
        try:
            h.mt_visual_overlay.place_forget()
        except tk.TclError:
            pass
        context = (id(h.session), getattr(getattr(h.session, 'review', None), 'source_archive_sha256', None), h.current_invocation)
        previous_context = getattr(self, '_selection_context', None)
        if previous_context != context:
            if getattr(self,'manual_gt_first',False):self.inf_visible.set(False)
            self._pending_frame_toggles = []
            self.collision_warning.clear()
            self.corner_pulse.clear()
            self.shift_preview.clear()
            self.gt_focus_id=''
            self.destroy_entry()
            if previous_context and previous_context[:2] != context[:2]:
                self.pending_tree_target = None
            self._selection_context = context
            self._geometry_layer_state = None
            self._idle_badge = None
            self.skey = h.current_invocation
            self.selected = self.hover = ""
            self._bad_mt_selection = None
            self.pointer_canvas = None
            self.points = []
            self.draw_reference_target = None
            self.draw_target = None
            self.drag = None
            self._click_candidate = None
            self.mode = "select"
            self.plate_fit_previous = None
            self.plate_fit_active = False
            self.plate_fit_gid = ""
            self.history_selected = ""
            self.model_location_target = ''
        if self.pending_tree_target and self.pending_tree_target[0] == h.current_invocation:
            self.selected = self.pending_tree_target[1]
            self.mode = "edit" if self.selected else "select"
            self.pending_tree_target = None
        s = self._read()
        if getattr(self,'_bad_mt_selection',None)!=(id(h.session),h.current_invocation,self.selected):
            self._bad_mt_selection=None
        if hasattr(self,'clean_bar'):
            from .mobile_canvas_navigator import ensure_selection
            ensure_selection(self)
            from .mobile_canvas_clean_view import prune
            prune(self)
        obj = next((o for o in (s or {}).get("objects", ()) if o["object_id"] == self.selected), None)
        if self.selected and obj is None:
            self.selected = self.hover = ""
            obj = None
            if self.mode == "edit": self.mode = "select"
        available = bool(s and h._image is not None and h._image_entry == s["image_entry"]
            and tuple(h._image_source_size or ()) == tuple(s["image_size"]))
        enabled = available and not h._saving and not self.busy
        # The previously reviewed 2->1 scene must first be reconciled; checking
        # the image must not imply approval of unresolved historical GT.
        self.approve.configure(state="normal" if enabled and not hasattr(self, 'clean_bar')
                               and not (s and s.get("_post_nms_pending")) else "disabled")
        self.exclude.configure(state="normal" if enabled else "disabled")
        self.reason_entry.configure(state="normal" if enabled else "disabled")
        self.crop.configure(state="disabled")
        execution = execution_state(h.session, h.current_invocation,
                                    image_entry=h._image_entry,
                                    image_size=h._image_source_size)
        self._canvas_execution = execution
        self._guard_gt_entry()
        text_edit_id = (self.entry_target[1] if self.entry is not None
                        and self.entry_target[0] == h.current_invocation else None)
        operator = operator_context(h.session, h.current_invocation, s, self.selected, execution)
        model = focused_assessment(s, self.selected, execution, text_edit_id=text_edit_id, allow_auto=False)
        model['operator'] = operator
        history = self._history() if not s else None
        if history:
            from ..ranking.mobile_mt_history import history_assessment
            if not any(d['attempt_id']==getattr(self,'history_selected','') for d in history['detections']):
                self.history_selected = ''
            model = history_assessment(model,history,getattr(self,'history_selected',''))
        from .mobile_canvas_stacks import present
        self.assessment.render(present(self,model))
        from ..ranking.mobile_canvas_resolution import missing_mt_geometry
        missing_pose = missing_mt_geometry(s)
        self.decision_panel.refresh(s, self.selected, enabled, execution)
        # Decisions belong to the selected Canvas object (Ctrl+PPM). Keep the
        # legacy adapter/API for stored evidence, without a second sidebar form.
        self.decision_panel.pack_forget()
        self.decision_panel.fp.pack_forget()  # PPM already owns this candidate action.
        from .mobile_canvas_stacks import enhanced
        if enhanced(self):
            if not any(option[0]!=h.current_invocation for option in getattr(self.decision_panel,'physical_options',())):
                self.decision_panel.physical_same.pack_forget()
                self.decision_panel.physical_pick.pack_forget()
            # Explicit GT=0 remains an image-level action, never an MT action.
            if self.selected:self.decision_panel.negative.pack_forget()
        if not s:
            self.checked_var.set(False); self.excluded_var.set(False)
            self.reconciliation_box.pack_forget()
            self.nms_action.pack_forget()
            self.reason_host.pack_forget()
            self.status.configure(text="Tylko podgląd", bg=self.p["amber_bg"], fg=self.p["amber"])
            self.tally.configure(text=history['reason'] if history else getattr(self, "read_error", "") or "Wybierz obraz z pełną klatką i ROI.")
            self.selection.configure(text="Inspekcja historii" if history else "Brak aktywnej ramki")
            self.selection_hint.configure(text="Kliknięcie nie tworzy GT ani rewizji." if history else "Edycja wymaga zgodnych współrzędnych źródłowych.")
            for card in self.tally_cards.values(): card.set("—")
            return
        from ..ranking.mobile_canvas_partial_review import progress as frame_progress
        self.checked_var.set(frame_progress(s)['complete'])
        self.excluded_var.set(bool(s.get("exclusion_reason")))
        # The explanatory decision panel is synced after reading final/raw counts.
        reason = s.get("exclusion_reason", "")
        default = "Operator oznaczył obraz jako niemożliwy do oceny"
        if h.window.focus_get() is not self.reason_entry or self.reason_gid != self.skey:
            self.reason_var.set("" if reason == default else reason)
        self.reason_gid = self.skey
        if reason:
            self.reason_host.pack(fill="x", pady=(4, 0))
        else:
            self.reason_host.pack_forget()
        issues = core._content_issues(s)
        progress = review_progress(s)
        tone, title = progress['tone'], progress['title']
        if not available: title += "\nPodgląd wejścia MT"
        self.status.configure(text=title, bg=self.p[tone+"_bg"], fg=self.p[tone])
        tally = core.counts(s)
        for key, card in self.tally_cards.items():
            card.set(str(tally[key]) if execution["state"] == "MT_EXECUTED" else "—")
        from ..ranking.mobile_candidate_stacks import physical_count_view
        physical_n,known=physical_count_view(s)
        self.tally_cards['expected'].title.configure(text='GT fizyczne')
        self.tally_cards['expected'].set(str(physical_n) if known else 'do ustalenia')
        self.tally_cards["missed"].title.configure(text="Ręczne bez MT" if missing_pose else "Pominięte tablice")
        raw_count=tally['mt']; final_count=raw_count-(1 if s.get('_post_nms_pending') else tally['raw_suppressed'])
        self.tally_cards['mt'].title.configure(text='Wskazania: surowe / końcowe')
        self.tally_cards['mt'].set(f'{raw_count} / {final_count}' if not s.get('_post_nms_unknown') else f'{raw_count} / ?')
        self.approve.configure(text='Obraz zweryfikowany')
        self._sync_reconciliation_story(s, tally, enabled)
        notes = []
        if tally["operator_errors"]:
            notes.append(f"Błędy operatora: {tally['operator_errors']} · Po zaznaczeniu R usuwa próbne ramki.")
        if reason:
            notes.append("Sprawdzony; poza mianownikami oceny.")
        elif s["checked"] and issues:
            notes.append("Liczba rzeczywistych tablic wymaga decyzji. Ten obraz nie jest liczony do podsumowania."
                         if missing_pose else "Do uzupełnienia: " + "; ".join(issues))
        elif not s["checked"]:
            notes.append("Liczby robocze — obraz niepotwierdzony.")
        self.tally.configure(text="\n".join(notes))
        parent = next((p for p in s["parents"] if obj and p["parent_id"] == obj["parent_id"]), None)
        if obj:
            title = f"MT {int(parent['detection_index'])+1} · nakładka" if parent else "Ramka ręczna"
            if obj["error"]: title = "× BŁĄD · " + title
            self.selection.configure(text=title, fg=self.p["red"] if obj["error"] else self.p["fg"])
            self.selection_hint.configure(text=("R: cofnij korektę do wyniku MT; GT pozostaje." if parent else "R: usuń ręcznie dodaną ramkę.")
                                          + "\nLPM: przeciągnij narożnik. S: odznacz.")
            self.crop.configure(state="normal" if parent and parent["mz_ran"] else "disabled")
        else:
            self.selection.configure(text="Wybierz ramkę", fg=self.p["fg"])
            self.selection_hint.configure(text="Kliknij ramkę lub najedź i naciśnij S.\nNarożniki pojawią się po zaznaczeniu.")
        self.sync_tree_selection()

    def available(self):
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.available()
        s = self._read()
        if self.h._saving or self.busy:
            self.message("Trwa zapis/odświeżanie wyników; bieżąca edycja pozostaje zachowana.")
            return False
        choices = getattr(self.h, '_image_rows', ())
        index = self.h.image_choice.current() if hasattr(self.h, 'image_choice') else -1
        if 0 <= index < len(choices) and choices[index][1].get('context_only'):
            self.message('Wycinek TV jest kontekstem. Edycja GT pozostaje na pełnej klatce kamery.')
            return False
        if not s or self.h._image is None or self.h._image_entry != s["image_entry"] or tuple(self.h._image_source_size or ()) != tuple(s["image_size"]):
            self.message("Rysowanie jest możliwe na zweryfikowanej pełnej klatce z ROI, nie na cropie ani innym obrazie.")
            return False
        return True

    def open_resolution(self):
        """Compatibility entry point: reveal context inline, never open a dialog."""
        if not self.prepare_navigation() or not self.available():
            return None
        self.sync()
        return self.decision_panel

    def show_model_location(self, object_id):
        """Inspect source location without selecting/editing a reference or saving."""
        from ..ranking.mobile_canvas_resolution import model_location
        s = self._read()
        try:
            model_location(self.h.session,self.h.current_invocation,s,core._object(s,object_id))
        except (ValueError,TypeError,KeyError) as exc:
            self.message(str(exc)); return
        self.model_location_target = object_id
        self.overlay()

    def to_canvas(self, point, *, image_size=None):
        s = self._read()
        h = self.h
        size = image_size or s['image_size']
        sx = h._image.width / size[0]
        sy = h._image.height / size[1]
        cx,cy = h._view_center
        z = h._view_pixel_scale
        return ((point[0]*sx-cx)*z+self.c.winfo_width()/2,
                (point[1]*sy-cy)*z+self.c.winfo_height()/2)

    def to_source(self, x, y):
        s = self._read()
        h = self.h
        cx,cy = h._view_center
        z = max(h._view_pixel_scale,1e-12)
        return ((cx+(x-self.c.winfo_width()/2)/z)*s["image_size"][0]/h._image.width,
                (cy+(y-self.c.winfo_height()/2)/z)*s["image_size"][1]/h._image.height)

    def _current_quad(self, obj):
        if self.drag and self.drag["oid"] == obj["object_id"]:
            return self.drag["quad"]
        if getattr(self,'reference_source',None):
            return self.reference_source.localization_outline(obj)
        return obj.get("quad")

    def paint(self, width, height):
        s = self._read()
        if not s or self.h._image_entry != s["image_entry"]:
            if self.mode=='draw':
                for item in self.c.find_all():
                    if 'evidence_image' not in self.c.gettags(item):self.c.delete(item)
                return True
            if not s and self._paint_history():
                return True
            # Old evidence remains visible, without legacy counting/drawing CTA.
            if self.h._is_source_frame_view():
                self.h._paint_source_frame_overlays(width,height)
            self.h.image_help.configure(text="Podgląd dowodu. Edycja wymaga potwierdzonej pełnej klatki + ROI.")
            return True
        if tuple(self.h._image_source_size or ()) != tuple(s["image_size"]):
            self.h.image_help.configure(text="Rozmiar obrazu nie odpowiada metadanym — edycja zablokowana.")
            return True
        self.overlay()
        return True

    def _history(self):
        if getattr(self, 'reference_source', None) is not None:
            return None
        from ..ranking.mobile_mt_history import history_evidence
        h=self.h
        return history_evidence(h.session,h.current_invocation,
            image_entry=h._image_entry,image_size=h._image_source_size)

    def _history_to_canvas(self, point):
        h=self.h;w,hh=h._image_source_size;cx,cy=h._view_center;z=h._view_pixel_scale
        return ((point[0]*h._image.width/w-cx)*z+self.c.winfo_width()/2,
                (point[1]*h._image.height/hh-cy)*z+self.c.winfo_height()/2)

    def _paint_history(self):
        self.c.delete('mt_history_inspection')
        history=self._history()
        if not history:
            return False
        if self.h._image is None or self.h._view_center is None:
            return True
        tags=('mt_history_inspection',)
        if history['view_bound'] and history['roi'] and self.show_roi_overlay:
            l,t,r,b=history['roi'];a=self._history_to_canvas((l,t));b=self._history_to_canvas((r,b))
            self.c.create_rectangle(*a,*b,outline='#FFD16C',dash=(7,4),width=1,tags=tags)
        for item in history['detections']:
            points=[self._history_to_canvas(p) for p in item['image_quad']]
            selected=item['attempt_id']==getattr(self,'history_selected','')
            self.c.create_polygon(*[v for p in points for v in p],fill='',outline='#36DDFB',
                width=2.8 if selected else 1.5,tags=tags+('mt_history_frame',))
            x,y=min(points,key=lambda p:(p[1],p[0]))
            label=self.c.create_text(max(6,min(self.c.winfo_width()-210,x)),max(8,y-23),anchor='nw',
                text=f"MT {item['detection_index']+1} · tylko inspekcja",fill='#E3FAFF',
                font=('Segoe UI',9,'bold'),tags=tags)
            box=self.c.bbox(label)
            if box:
                bg=self.c.create_rectangle(box[0]-3,box[1]-2,box[2]+3,box[3]+2,fill='#13313D',outline='#36DDFB',tags=tags)
                self.c.tag_lower(bg,label)
        self.h.image_help.configure(text=history['image_status']+' · '+history['reason'])
        return True

    def _inspect_history(self,event):
        history=self._history()
        if not history or not history['view_bound'] or self.h._image is None or self.h._view_center is None:
            return
        selected=''
        for item in history['detections']:
            if quad_contains_point([self._history_to_canvas(p) for p in item['image_quad']],event.x,event.y):
                selected=item['attempt_id'];break
        self.history_selected=selected
        self.selected=self.hover=''
        self.c.focus_set()
        self.h._view_drag_at=(event.x,event.y,self.h._view_center)
        self.sync();self._paint_history()
        self.message(history['reason'])

    def request_overlay(self):
        if self.paint_scheduled or self.h._closed:
            return
        self.paint_scheduled = True
        def repaint():
            self.paint_after = None
            self.paint_scheduled = False
            if not self.h._closed:
                self.overlay()
        self.paint_after = self.c.after_idle(repaint)

    def overlay(self):
        from ..ranking.mobile_canvas_presentation import execution_state, geometry_assessment, plate_review_state
        if hasattr(self,'task_locator'):self.task_locator.sync()
        if self.drag and hasattr(self,'navigator'):
            from .mobile_canvas_corner_drag import paint as paint_drag
            if paint_drag(self):return
        self.c.delete(TAG)
        if self.mode=='draw':
            # Drawing owns the overlay, including remnants from older renderers.
            for item in self.c.find_all():
                if 'evidence_image' not in self.c.gettags(item):self.c.delete(item)
        s = self._read()
        if hasattr(self,'clean_bar'):
            from .mobile_canvas_clean_view import paint
            paint(self)
            from .mobile_canvas_scene_frames import paint_sources, paint_tv_bbox
            paint_sources(self)
            paint_tv_bbox(self)
            self.collision_warning.paint()
            if s:
                if self.drag:self._render_drag_geometry(s)
                elif not getattr(self.h, '_viewport_paint', False):self._render_contexts(self.assessment.model)
            self.corner_pulse.paint()
            if hasattr(self,'task_locator'):self.task_locator.paint()
            return
        if not s:
            self._paint_history()
            return
        if not s or self.h._image is None or self.h._image_entry != s["image_entry"] or self.h._view_center is None:
            return
        from .mobile_canvas_stacks import paint
        if self.mode!='draw' and paint(self):return
        # Image loading can paint before sync(). Never borrow the execution
        # state of the previously selected image for provenance or GT badges.
        reference = getattr(self, 'reference_source', None)
        execution = None if reference else execution_state(self.h.session, self.h.current_invocation,
            image_entry=self.h._image_entry, image_size=self.h._image_source_size)
        width,height = self.c.winfo_width(),self.c.winfo_height()
        self.badges = {}
        self.card_positions = {}
        self.card_widths = {}
        self.card_heights = {}
        self.card_header_heights = {}
        self.stack_badges = {}

        def draw_contour(points, *, color, width, tags=(), dash=()):
            # A SINGLE fine contour. The old dark halo looked like a second box.
            flat = [v for p in points for v in p]
            self.c.create_polygon(*flat, fill="", outline=color, width=width, dash=dash,
                                  tags=tags)

        l,t,r,b = s["roi"]
        x1,y1 = self.to_canvas((l,t)); x2,y2 = self.to_canvas((r,b))
        if self.show_roi_overlay:
            # Outline only: the old gray12 stipple put a dotted grid over the photo.
            self.c.create_rectangle(x1,y1,x2,y2, fill="", outline="#FFD16C",
                                    width=1.1, tags=(TAG,"analyzed_roi"))
        parents = {p["parent_id"]:p for p in s["parents"]}
        from .mobile_canvas_scene_frames import paint_tv_bbox
        if not reference:paint_tv_bbox(self)
        suppressed_parent_ids = {o["parent_id"] for o in s["objects"]
                                 if o.get("post_nms_suppressed", False)}
        # Exclude RAW candidates rejected by NMS only from the visual overlay;
        # preserve all original detections and their evidence for auditing.
        for parent in s["parents"]:
            if not parent["quad"] or parent["parent_id"] in suppressed_parent_ids:
                continue
            # Keep the immutable prediction visible beside the editable GT.
            pts = [self.to_canvas(p) for p in parent["quad"]]
            draw_contour(pts, color="#00E5FF", width=1.5,
                         tags=(TAG,"mt_original","mt_accepted",parent["parent_id"]))
        protected = self._overlay_plate_bounds(s)
        target_id = getattr(self, 'model_location_target', '')
        if target_id:
            from ..ranking.mobile_canvas_resolution import model_location
            try:
                target = core._object(s,target_id)
                location = model_location(self.h.session,self.h.current_invocation,s,target)
            except (ValueError,KeyError,TypeError):
                location = None
            if location:
                parent = parents[target['parent_id']]
                if location['kind'] == 'BBOX':
                    l,t,r,b = location['bbox']
                    x1,y1 = self.to_canvas((l,t)); x2,y2 = self.to_canvas((r,b))
                    self.c.create_rectangle(x1,y1,x2,y2,outline='#00E5FF',width=2,dash=(6,3),
                        tags=(TAG,'model_location_bbox',target_id))
                else:
                    points = [self.to_canvas(p) for p in location['quad']]
                    x1,y1 = min(p[0] for p in points),min(p[1] for p in points)
                    x2,y2 = max(p[0] for p in points),max(p[1] for p in points)
                    draw_contour(points,color='#00E5FF',width=2,tags=(TAG,'model_location_pose',target_id))
                caption = f"Model — wskazanie {int(parent['detection_index'])+1}"
                self.c.create_text(max(8,x1),max(8,y1-5),anchor='sw',text=caption,fill='#00E5FF',
                    font=('Segoe UI',9,'bold'),tags=(TAG,'model_location_label',target_id))
                protected.append((x1,y1,x2,y2))
        placed=[]
        for index,obj in enumerate(s["objects"],1):
            if obj.get("post_nms_suppressed", False):
                continue  # RAW evidence already drawn above, never a second GT badge.
            quad = self._current_quad(obj)
            if not quad:
                if reference:reference.paint_presence_only(obj, index)
                continue
            pts = [self.to_canvas(p) for p in quad]
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            if max(xs)<0 or min(xs)>width or max(ys)<0 or min(ys)>height:
                continue
            oid = obj["object_id"]
            if obj["error"] or obj.get('duplicate_of'):
                color, contour_width = "#FF3B5C", 1.8
            elif oid == self.selected:
                color, contour_width = "#FFD400", 2.0
            else:
                color, contour_width = "#FF8A00", 1.5
            style=reference.geometry_style(obj) if reference else None
            if style:color=style['color']
            contour_tag='photo_approximate_outline' if reference and obj['quad'] is None else 'operator_quad'
            draw_contour(pts, color=color, width=contour_width,
                         tags=(TAG,contour_tag,oid),dash=style['dash'] if style else ())
            if obj["error"]:
                for a,b in ((0,2),(1,3)):
                    self.c.create_line(*pts[a],*pts[b],fill="#FF3B5C",width=1.5,tags=(TAG,"error_flag",oid))
            if self.mode=="edit" and oid == self.selected:
                for j,(x,y) in enumerate(pts):
                    self.c.create_oval(x-6,y-6,x+6,y+6,fill="#FFFFFF",outline=color,width=1.4,
                        tags=(TAG,"corner_handle",oid))
            parent = parents.get(obj["parent_id"])
            live = (quad if self.drag and self.drag.get("oid") == oid
                    and quad != self.drag.get("before") else None)
            state = (self._gt_policy(oid) if reference else geometry_assessment(s, obj, execution, live_quad=live) if live is not None
                     else plate_review_state(s, obj, execution))
            editing = self.entry is not None and self.entry_target == (self.h.current_invocation, oid)
            number = self.entry_var.get() if editing else obj.get("gt_text", "")
            caption = "Numer: " + (number or ("wpisz numer" if state["gt_editable"] else "wyjaśnienie w panelu"))
            if not number and parent and parent["mz_status"] == "NOT_RUN" and state["gt_editable"]:
                caption = "Numer niewymagany · brak odczytu"
            mz_label = f"Tablica {index} · {style['label']}" if reference else state["canvas_frame"]
            if reference:
                number=reference.display_text(obj) if not editing else number
                caption=reference.text_caption(obj, number)
            from .mobile_canvas_stacks import view as stack_view,members as stack_members
            sv=stack_view(self);pid=(sv or {}).get('links',{}).get(oid,{}).get('plate_id')
            if parent:
                mz_label = mz_label.replace("MT ·", f"MT {int(parent['detection_index'])+1} ·", 1)
            if pid in (sv or {}).get('plates',{}):
                number_in_view=list(sv['plates']).index(pid)+1
                count_in_view=len(stack_members(sv,pid))
                active_label=f' · {number_in_view}.1/{count_in_view}' if oid==self.selected and count_in_view else ''
                mz_label=f'Tablica {number_in_view} · {count_in_view} MT{active_label} · '+mz_label
            if obj["error"]:
                mz_label = "× " + mz_label
            font = getattr(self, "_badge_font", None)
            measure = font.measure if font else lambda text: len(text) * 7
            line_height = font.metrics("linespace") if font else 17
            cw = min(max(150, measure(caption)+14, measure(mz_label)+14, 194 if editing else 0), max(40, min(300, width-16)))
            header_h = line_height * max(1, math.ceil(measure(mz_label) / max(1, cw-14))) + 6
            gt_lines = (1 if measure(caption) <= cw-14 else
                        1 + math.ceil(measure(number or caption) / max(1, cw-14)))
            gt_h = line_height * gt_lines + 6
            entry_h = line_height + 8 if editing else 0
            ch = header_h + 2 + gt_h + entry_h
            position = overlay_label_position((min(xs), min(ys), max(xs), max(ys)),
                (cw, ch), (width, height), protected + placed)
            if position is None:
                # Keep the complete GT at the viewport edge even at extreme
                # zoom. Prefer the corner obscuring the least outline area.
                candidates = [(8,8), (max(8,width-cw-8),8),
                              (8,max(8,height-ch-8)), (max(8,width-cw-8),max(8,height-ch-8))]
                def overlap_area(point):
                    x,y = point
                    return sum(max(0, min(x+cw,r)-max(x,l)) * max(0, min(y+ch,b)-max(y,t))
                               for l,t,r,b in protected + placed)
                position = min(candidates, key=overlap_area)
            xx, yy = position
            placed.append((xx,yy,xx+cw,yy+ch))
            self.card_positions[oid]=(xx,yy)
            self.card_widths[oid]=cw
            self.card_heights[oid]=ch
            self.card_header_heights[oid]=header_h
            self.stack_badges[oid]=(xx,yy,xx+cw,yy+header_h+gt_h+2,pid)
            center=(sum(xs)/4,sum(ys)/4)
            tx=max(xx,min(xx+cw,center[0])); ty=max(yy,min(yy+ch,center[1]))
            self.c.create_line(tx,ty,*center,fill="#DDE7EF",width=1.0,tags=(TAG,"badge_tether",oid))
            self.c.create_rectangle(xx,yy,xx+cw,yy+header_h,fill="#253025" if reference else "#0F2230",outline=color if reference else "#00E5FF",width=1.0,tags=(TAG,"mz_badge",oid))
            self.c.create_text(xx+6,yy+3,anchor="nw",text=mz_label,fill="#F2FBFF",font=("Segoe UI",9,"bold"),width=cw-12,tags=(TAG,"mz_badge","stack_badge_text",oid))
            gt_top = yy+header_h+2
            self.c.create_rectangle(xx,gt_top,xx+cw,gt_top+gt_h,fill="#302618",outline=color,width=1.0,tags=(TAG,"gt_badge",oid))
            self.c.create_text(xx+6,gt_top+3,anchor="nw",text=caption,fill=color,font=("Segoe UI",9,"bold"),width=cw-12,tags=(TAG,"gt_badge",oid))
            self.badges[oid]=(xx,gt_top,xx+cw,gt_top+gt_h)
            if self.entry is not None and self.entry_target==(self.h.current_invocation,oid):
                self.entry.place(x=xx+1,y=gt_top+gt_h,width=cw-2,height=entry_h)
                self.entry.lift()
        if reference:reference.paint_location_draft()
        if self.points:
            pts=[self.to_canvas(p) for p in self.points]
            if self.pointer:
                pts.append(self.to_canvas(self.pointer))
            if len(pts)>=2:
                self.c.create_line(*[v for p in pts for v in p],fill="#FFF2A8",width=2.6,tags=(TAG,"drawing_line"))
            for x,y in pts[:len(self.points)]:
                self.c.create_oval(x-5,y-5,x+5,y+5,fill="#FFD400",outline="#FFFFFF",width=1.4,tags=(TAG,"drawing_corner"))
        if not reference:self._paint_live_overlap_hud(s, execution)
        self.c.tag_raise("corner_handle")
        mode="D: wskaż róg %d/4"%(len(self.points)+1) if self.mode=="draw" else "S: odznacz/wybierz · LPM: narożnik · R: usuń/cofnij · PPM: błąd 1 klik" if self.mode=="edit" else "Najedź + S: zaznacz · D: dodaj"
        self.h.image_help.configure(text=("A: następna tablica + dopasowanie  •  F: cały obraz  •  R: usuń/cofnij ramkę  •  T: pokaż/ukryj ROI  •  "
               "N: następna tablica  •  Spacja: zatwierdź ramki tablicy  •  PPM: błąd  •  Ctrl+PPM: decyzje GT  •  "
               "turkus: MT  •  żółty: GT/korekta"))
        self.c.configure(cursor="crosshair" if self.mode=="draw" else "arrow")
        if reference:reference.caption()

    def _overlay_plate_bounds(self, scene):
        suppressed = {o["parent_id"] for o in scene["objects"] if o.get("post_nms_suppressed")}
        quads = [self._current_quad(o) for o in scene["objects"] if not o.get("post_nms_suppressed")]
        quads += [p["quad"] for p in scene["parents"] if p["parent_id"] not in suppressed]
        bounds = []
        for quad in quads:
            if quad:
                points = [self.to_canvas(p) for p in quad]
                bounds.append((min(x for x, y in points), min(y for x, y in points),
                               max(x for x, y in points), max(y for x, y in points)))
        return bounds

    def _paint_live_overlap_hud(self, scene, execution):
        """On-canvas badge next to corrected quad; per-drag assisted metrics.

        Compare original MT with live operator quad (not independent truth).
        Display only; never edit either polygon or persist intermediate points.
        """
        if self.mode != "edit" or not self.selected:
            return
        obj = next((o for o in scene["objects"] if o["object_id"] == self.selected
                    and not o.get("post_nms_suppressed", False)), None)
        if obj is None:
            return
        quad = self._current_quad(obj)
        if not quad:
            return
        from ..ranking.mobile_canvas_presentation import geometry_assessment, focused_assessment, operator_context
        live = (quad if self.drag and self.drag.get("oid") == obj["object_id"]
                and quad != self.drag.get("before") else None)
        model = geometry_assessment(scene, obj, execution, live_quad=live)
        mt = model["mt_label"] or "RAMKA OPERATORA"
        status = model["geometry_status"]
        if status in {"ASSESSED", "LIVE"}:
            lines = (f"{mt} ↔ REFERENCJA · ASSISTED", model["geometry"],
                     "Korekta na żywo · niezapisana" if status == "LIVE" else
                     "Oryginał MT zaakceptowany" if model["frame_state"] == "MT_ORIGINAL" else "Referencja po korekcie operatora")
        elif status == "FP":
            lines = (f"{mt} — błędne wykrycie (FP)", model["geometry"], "Brak referencji rzeczywistej tablicy")
        elif status == "UNCONFIRMED":
            lines = (f"{mt} · referencja niepotwierdzona", model["geometry"], "Sprawdź ramkę i potwierdź obraz")
        elif status == 'NOT_AVAILABLE':
            lines = ('Wskazanie powiązane z tablicą', 'Brak narożników modelu — obrys bez oceny')
        else:
            lines = (mt, model["geometry"])
        # Live and saved previews use the same pair and qualification as the
        # panel, including FP and unconfirmed copies. No intermediate writes.
        if self.assessment.model.get("focus_id") == obj["object_id"]:
            panel = focused_assessment(scene, self.selected, execution,
                text_edit_id=self.entry_target[1] if self.entry is not None else None, live_quad=live)
            panel['operator'] = operator_context(self.h.session,self.h.current_invocation,scene,self.selected,execution)
            if status == "LIVE":
                panel.update(summary_note="Podgląd niezapisanej korekty — jeszcze nie liczony do podsumowania", summary_tone="muted")
            self.assessment.render(panel)
        cw = int(self.c.winfo_width())
        ch = int(self.c.winfo_height())
        if cw < 180 or ch < 100:
            return
        pts = [self.to_canvas(p) for p in quad]
        px = sum(x for x, _ in pts) / len(pts)
        top = min(y for _, y in pts)
        badge_w = min(338, cw - 16)
        badge_h = 69 if len(lines) == 3 else 51
        bounds = (min(x for x,y in pts), top, max(x for x,y in pts), max(y for x,y in pts))
        cards = [(x, y, x+getattr(self, "card_widths", {}).get(oid, 194), y+getattr(self, "card_heights", {}).get(oid,48))
                 for oid, (x,y) in self.card_positions.items()]
        position = overlay_label_position(bounds, (badge_w, badge_h), (cw, ch),
                                          self._overlay_plate_bounds(scene) + cards, above=True)
        if position is None:
            return
        left, badge_y = position
        # A small connector identifies which quad the values describe.
        if top > badge_y+badge_h+1:
            self.c.create_line(px, badge_y+badge_h, px, min(top, ch),
                               fill="#F9CE55", width=1.4,
                               tags=(TAG,"overlap_hud","overlap_hud_tether"))
        self.c.create_rectangle(left, badge_y, left+badge_w, badge_y+badge_h,
                                fill="#081925", outline="#FF3B5C" if status == "FP" else "#FFD400", width=1.5,
                                tags=(TAG,"overlap_hud","overlap_hud_badge"))
        for i, line in enumerate(lines):
            self.c.create_text(left+9, badge_y+12+i*21, text=line, anchor="w",
                               font=("Segoe UI", 9, "bold" if i==0 else "normal"),
                               fill="#FFFFFF" if i==0 else "#E7F2FF",
                               tags=(TAG,"overlap_hud","overlap_hud_text"))

    def hit(self, x, y, badge=True, *, prefer_stack=False):
        if hasattr(self,'clean_bar'):
            from .mobile_canvas_clean_view import hit
            return hit(self,x,y,prefer_stack=prefer_stack)
        from .mobile_canvas_stacks import enhanced,hit as stack_hit
        if self.mode!='draw' and enhanced(self):return stack_hit(self,x,y)
        s = self._read()
        if not s or self.h._image_entry != s["image_entry"] or self.h._view_center is None:
            return None
        objects = [o for o in s["objects"] if not o.get("post_nms_suppressed", False)]
        objects.sort(key=lambda o:o['object_id']==self.selected)  # reversed body hit gives explicit selection priority
        # Only the explicitly selected child has draggable handles. Their
        # outside half and paint/hit priority are retained; hover owns no handle.
        if self.mode == "edit" and self.selected:
            obj = next((o for o in objects if o["object_id"] == self.selected), None)
            if obj:
                for i, p in enumerate(self._current_quad(obj) or ()):
                    if math.dist((x, y), self.to_canvas(p)) <= 14:
                        return obj["object_id"], "corner", i
        if badge:
            for oid, (l,t,r,b) in reversed(list(self.badges.items())):
                if l <= x <= r and t <= y <= b:
                    return oid, "badge", None
            for oid, (l, t) in reversed(list(self.card_positions.items())):
                if l <= x <= l + getattr(self, "card_widths", {}).get(oid, 194) and t <= y <= t + getattr(self, "card_header_heights", {}).get(oid,23):
                    return oid, "mz_badge", None
        if getattr(self,'reference_source',None):
            location=self.reference_source.location_hit(x,y)
            if location:return location
        def near_edge(quad):
            pts = [self.to_canvas(p) for p in quad]
            for a,b in zip(pts, pts[1:]+pts[:1]):
                dx,dy=b[0]-a[0],b[1]-a[1]
                denom=dx*dx+dy*dy
                u=max(0,min(1,((x-a[0])*dx+(y-a[1])*dy)/denom)) if denom else 0
                if math.dist((x,y),(a[0]+u*dx,a[1]+u*dy)) <= 11:
                    return True
            return False
        source = self.to_source(x,y)
        for obj in reversed(objects):
            quad = self._current_quad(obj)
            if quad and (quad_contains_point(quad,*source) or near_edge(quad)):
                return obj["object_id"], "body", None
        # S over an unchanged MT original targets its one editable child.
        for parent in reversed(s["parents"]):
            if not any(o["parent_id"] == parent["parent_id"] for o in objects):
                continue
            quad=parent["quad"]
            if quad and (quad_contains_point(quad,*source) or near_edge(quad)):
                obj=next((o for o in objects if o["parent_id"]==parent["parent_id"]),None)
                if obj: return obj["object_id"], "body", None
        return None

    def _commit(self, action, description, remember=True):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:
            self.message('A3 korzysta z gotowego GT fotografii. Oceń wynik przez menu PPM.');return False
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.commit(action)
        if not self.available():
            return False
        gid=self.h.current_invocation
        before=copy.deepcopy(self._read())
        revision_before=self.h.session.review.review_revision
        self.busy=True
        try:
            value=action()
        except (ValueError, OSError, KeyError, TypeError) as e:
            from ..ranking.mobile_manual_frame_guard import GTRCollisionError
            if isinstance(e,GTRCollisionError):
                if len(self.points)==4:self.points.pop()
                from .mobile_canvas_reservation import handle
                self.busy=False;handle(self,e.reference)
                return False
            self.message("Nie zapisano: "+str(e))
            return False
        finally:
            self.busy=False
        if self.h.session.review.review_revision==revision_before:
            self.message('Nie zmieniono zapisanej oceny.')
            return value if value is not None else True
        self.collision_warning.clear()
        if remember:
            self.shift_preview.clear()
            self.undo_stack.setdefault(gid,[]).append(before)
            self.undo_stack[gid]=self.undo_stack[gid][-30:]
            self.redo_stack[gid]=[]
        self.cache_key=None
        self.refresh_tree()
        self.sync()
        self.update_report()
        self.request_overlay()
        self.message(description+f" · zapisano rewizję {self.h.session.review.review_revision}")
        self.schedule_refresh()
        return value if value is not None else True

    def schedule_refresh(self):
        if getattr(self, 'reference_source', None) is not None:
            return
        if self.refresh_after:
            self.c.after_cancel(self.refresh_after)
        self.refresh_after=self.c.after(650,self.refresh_results)

    def refresh_results(self):
        if self.refresh_after:
            self.c.after_cancel(self.refresh_after)
        self.refresh_after=None
        h=self.h
        if h._closed:
            return
        if self.entry is not None or self.mode=='draw' or self.points or self.drag or h._saving or self.in_refresh:
            self.refresh_after=self.c.after(650,self.refresh_results)
            return
        self.in_refresh=True
        def after():
            self.in_refresh=False
            if not self.drag and self.entry is None:self._drain_frame_verification()
        # Persistence already succeeded. Read-only work must not take the
        # write lock or reload the image while the next corner is being held.
        from .mobile_canvas_statistics_refresh import request
        if not request(h,after=after):
            self.in_refresh=False
            self.schedule_refresh()

    def draw_toggle(self,event=None):
        if getattr(getattr(self,'reference_source',None),'read_only',False):
            self.message('Dowód tylko do odczytu. Wróć do GT sceny, aby rysować.');return 'break'
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:
            self.message('GT fotografii jest gotowe. W A3 przypisz wynik MT do tablicy przez PPM.');return 'break'
        if event is not None and self._is_typing_shortcut(getattr(event,'widget',None)):
            return None
        if event is not None and getattr(event,"state",0)&4:
            return "break"
        if getattr(self,'reference_source',None) and not self.reference_source.prepare_location():
            return 'break'
        if not self.finish_entry() or not self.available():
            return "break"
        self.shift_preview.clear()
        from .mobile_canvas_draw_target import resolve
        try:target=(self.reference_source.draw_target() if getattr(self,'reference_source',None) else resolve(self)) if self.mode!='draw' else None
        except (ValueError,KeyError,TypeError,OSError) as exc:
            self.message(str(exc));return 'break'
        if target and target['kind']=='focus':
            from .mobile_canvas_reservation import handle
            handle(self,target['reference']);return 'break'
        if target and target['kind']=='seed':
            from .mobile_canvas_reference_seed import focus
            focus(self,target['oid']);return 'break'
        self.collision_warning.clear()
        self.draw_target=target
        self.draw_reference_target=((target['gid'],target['oid'],target['plate_id'])
                                    if target and target['kind']=='complete' else None)
        self.mode="select" if self.mode=="draw" else "draw"
        self._geometry_layer_state=None
        self.gt_focus_id=''
        self.selected=self.hover=""
        self.points=[]; self.pointer=None; self.drag=None
        self._click_candidate = None
        self.sync()
        self.request_overlay()
        return "break"

    def _pointer_for_selection(self, event=None):
        """Tk key-event x/y are NOT reliable canvas pointer coordinates.

        On Windows/X11 they may be (0,0) or even negative despite a mouse
        visibly hovering over the plate. Query the real screen pointer, then
        convert to canvas-local pixels. Only fall back to the last Motion event
        on lightweight non-Tk fixtures; never infer plate from text or GT.
        """
        if event is not None and getattr(event, "widget", self.c) is not self.c:
            return None

        def inside(xy):
            return (xy is not None and all(math.isfinite(v) for v in xy)
                    and 0 <= xy[0] < self.c.winfo_width()
                    and 0 <= xy[1] < self.c.winfo_height())

        try:
            xy = (float(self.c.winfo_pointerx() - self.c.winfo_rootx()),
                  float(self.c.winfo_pointery() - self.c.winfo_rooty()))
            if inside(xy):
                return xy
            # Pointer is physically outside: do not reuse stale hover data.
            return None
        except (AttributeError, tk.TclError, TypeError, ValueError, OverflowError):
            pass
        return self.pointer_canvas if inside(self.pointer_canvas) else None

    def _canvas_hotkey_pointer(self, event=None):
        """Read actual pointer *inside* this canvas, not keyboard event x/y.

        When Windows reports a 4–10px pointer offset at DPI scaling, a recent
        canvas Motion point is more accurate for hitting a narrow plate rim.
        The physical pointer MUST still be inside the canvas so a stale hover
        cannot activate shortcuts from the sidebar.
        """
        if self.entry is not None:
            return None
        widget = getattr(event, "widget", self.c) if event is not None else self.c
        if self._is_typing_shortcut(widget):
            return None
        if widget is not self.c:
            try:
                if widget.winfo_toplevel() is not self.h.window:
                    return None
            except (tk.TclError, AttributeError):
                return None
        # A Motion event already contains true canvas-local coordinates. This
        # is more accurate than OS screen coordinates under Windows DPI scaling.
        actual = self._pointer_for_selection(None)
        if actual is None:
            return None
        hover = self.pointer_canvas
        if (hover is not None and len(hover) == 2
                and 0 <= hover[0] < self.c.winfo_width()
                and 0 <= hover[1] < self.c.winfo_height()
                and math.dist(actual, hover) <= 90):
            return tuple(hover)
        return actual

    def _choose_plate(self, oid, *, toggle=False, source=False):
        if oid!=self.selected or toggle:self._geometry_layer_state=None
        if hasattr(self,'clean_bar'):
            from .mobile_canvas_clean_view import eligible,records
            if not eligible(self,oid):
                alias=next((r for r in records(self) if r.get('reference_id')==oid and r['candidate'] and r['gt']),None)
                if not alias:return False
                oid=alias['oid']
            if oid!=self.selected or toggle:self.gt_focus_id=''
            self._s_hover_hit=None
        """Select one operator quad without editing its saved GT or MT."""
        scene = self._read()
        if oid and not any(o["object_id"] == oid
                           for o in (scene or {}).get("objects", ())):
            return False
        self.selected = "" if toggle and oid == self.selected else oid
        self._bad_mt_selection=((id(self.h.session),self.h.current_invocation,self.selected)
                                if source and self.selected else None)
        self.model_location_target = ''
        self.mode = "edit" if self.selected else "select"
        self.hover = ""
        self.h._view_drag_at = None
        if not self.selected:
            self.clear_tree_selection()
        elif not source and hasattr(self,'navigator'):
            # Existing S/N/explicit object selection targets the operator's
            # reference; source inspection uses select_layer(..., BBOX/POSE).
            from .mobile_canvas_geometry_layers import model as layer_model
            value=layer_model(self)
            if value and any(l['kind']=='GTR' for l in value['layers']):
                self._geometry_layer_state=(value['identity'],'GTR')
        self.sync()
        self.request_overlay()
        self.message("Wybrano ramkę — przeciągnij biały uchwyt, aby poprawić narożnik GT."
                     if self.selected else
                     "Wybierz obrys kliknięciem lub klawiszem S.")
        return True

    def select_mode(self, event=None):
        """S at the real pointer location toggles the editable quad."""
        # The real Windows session recorded S with Tk state=8 (Mod1).
        # Reject Ctrl (bit 4) and leave typing in the GT field untouched.
        if event is not None and getattr(event, "state", 0) & 4:
            return "break"
        if self.drag:
            self.message("Dokończ przeciąganie narożnika przed zmianą zaznaczenia.")
            return "break"
        if self.points:
            self.message("Dokończ cztery punkty albo anuluj D/Esc.")
            return "break"
        if not self.finish_entry():
            return "break"
        scene = self._read()
        h = self.h
        if (not scene or h._image is None or h._image_entry != scene["image_entry"]
                or tuple(h._image_source_size or ()) != tuple(scene["image_size"])
                or h._view_center is None):
            self.message("S: to nie jest edytowalna pełna klatka lub nie ma geometrii obrazu.")
            return "break"
        xy = self._canvas_hotkey_pointer(event)
        if xy is None:
            self.message("S: przesuń kursor na zdjęcie, a następnie nad obrys tablicy.")
            return "break" if getattr(event, "widget", self.c) is self.c else None
        self.pointer_canvas = xy
        # Prefer the object resolved on the actual Canvas <Motion> event. This
        # avoids Windows DPI offsets turning S into a silent miss on thin quads.
        cached = getattr(self, "_s_hover_hit", None)
        target = None
        if hasattr(self,'navigator'):
            from .mobile_canvas_navigator import selection_hit
            target=selection_hit(self,*xy)
        if cached is not None:
            group_id, oid, hover_xy = cached
            if (group_id == self.h.current_invocation
                    and math.dist(xy, hover_xy) <= 36
                    and any(o["object_id"] == oid and not o.get("post_nms_suppressed", False)
                            for o in scene["objects"])):
                if hasattr(self,'navigator') and (target is None or target[1]!='layer:GTR'):
                    hover_target=selection_hit(self,*hover_xy)
                    from .mobile_canvas_bbox_marker import hit as marker_hit
                    if hover_target and hover_target[1]=='layer:GTR' and not marker_hit(self,*xy):target=hover_target
                if target is None:target = (oid, "body", None)
        if target is None:
            target = self.hit(*xy, badge=True)
        if target:
            if hasattr(self,'navigator') and target[1]=='layer:BBOX':
                from .mobile_canvas_bbox_marker import hit as marker_hit
                if marker_hit(self,*xy):
                    from .mobile_canvas_navigator import select_layer
                    select_layer(self,'BBOX',target[0])
                    return 'break'
            toggle=True
            if hasattr(self,'navigator') and target[1]=='layer:GTR':
                from .mobile_canvas_geometry_layers import editable
                toggle=target[0]==self.selected and self.mode=='edit' and editable(self)
            self._choose_plate(target[0], toggle=toggle)
        else:
            self.message("Najedź kursorem na ramkę tablicy i naciśnij S.")
        return "break"


    def selection_key_press(self, event):
        if self._canvas_hotkey_pointer(event) is None:
            # Let other panels use S normally. A pointer inside the image
            # receives a visible explanation through select_mode().
            return "break" if getattr(event, "widget", self.c) is self.c else None
        if self.selection_release_after:
            self.c.after_cancel(self.selection_release_after)
            self.selection_release_after = None
        if self.selection_key_down:
            return "break"
        self.selection_key_down = True
        return self.select_mode(event)

    def selection_key_release(self, event=None):
        if event is not None and getattr(event, "widget", None) is not self.c and not self._event_in_review_window(event):
            return None
        # X11 repeats release/press pairs; Windows normally repeats KeyPress.
        # Idle deferral coalesces both into one toggle per physical key press.
        if self.selection_release_after:
            self.c.after_cancel(self.selection_release_after)
        self.selection_release_after = self.c.after_idle(self.reset_selection_key)
        return "break"

    def reset_selection_key(self, event=None):
        if self.selection_release_after:
            self.c.after_cancel(self.selection_release_after)
            self.selection_release_after = None
        self.selection_key_down = False

    def double_press(self,event):
        from .mobile_canvas_stacks import control_hit
        from .mobile_canvas_geometry_layers import contains
        target=self.hit(event.x,event.y,prefer_stack=True) if self.mode!='draw' else None
        corner=bool(target and target[1]=='corner')
        # A short background click after a hold/pan can arrive as Double-1.
        # Arm the same release-time gesture; movement/hold still prevents clear.
        background=bool(self.selected and target is None)
        return self.press(event) if corner or background or self.mode=='draw' or control_hit(self,event.x,event.y) or contains(self,event.x,event.y) else 'break'

    def press(self,event):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:
            self._click_candidate=None;self._outside_click=None;self.c.focus_set()
            if self.h.current_record and self.h.current_record[0]!='a3frame':
                target=self.hit(event.x,event.y,prefer_stack=True)
                if target:self.selected=target[0]
            self.mode='select';self.request_overlay()
            self.h._view_drag_at=(event.x,event.y,self.h._view_center)
            return 'break'
        if getattr(self,'reference_source',None) and self.reference_source.location_press(event):
            return 'break'
        self._outside_click=None;self._left_press_time=time.monotonic()
        had_entry=self.entry is not None
        self._click_candidate = None
        self._click_layer = None
        from .mobile_canvas_geometry_layers import click as layer_click
        if layer_click(self,event):return 'break'
        from .mobile_canvas_stacks import control_hit,cycle
        control=control_hit(self,event.x,event.y)
        if control and self.mode!='draw' and not self.gt_focus_id:
            # Navigation never commits an open GT text edit or changes history.
            if self.entry is not None:return 'break'
            self.c.focus_set()
            cycle(self,event,back=control[1],oid=control[0])
            return 'break'
        if not self.finish_entry():
            return "break"
        if not self.available():
            self._inspect_history(event)
            return "break"
        from .mobile_canvas_stacks import associate_click
        if associate_click(self,event):return 'break'
        self.c.focus_set()
        if self.mode=="draw":
            self.pointer_canvas=(event.x,event.y)
            p=self.to_source(event.x,event.y)
            l,t,r,b=self._read()["roi"]
            if not (l<=p[0]<=r and t<=p[1]<=b):
                self.message("Brakujące ramki rysujemy tylko w analizowanym ROI.")
                return "break"
            from ..ranking.mobile_manual_frame_guard import find_point_collision,require_free_manual_area,GTRCollisionError
            exclude=self.draw_target['plate_id'] if self.draw_target else None
            other=self.to_source(event.x+1,event.y)
            tolerance=math.hypot(other[0]-p[0],other[1]-p[1])
            conflict=find_point_collision(self._read(),p,exclude_plate_id=exclude,tolerance=tolerance,
                                          session=None if getattr(self,'reference_source',None) else self.h.session,gid=self.h.current_invocation)
            if conflict:
                if getattr(self,'reference_source',None):
                    self.message('W tym miejscu jest już obrys tablicy. Wybierz go, aby poprawić narożniki.')
                    return 'break'
                from .mobile_canvas_reservation import handle
                handle(self,conflict)
                from .mobile_canvas_guides import update_draft
                update_draft(self)
                return "break"
            if len(self.points)<4:
                self.points.append(p)
            if len(self.points)==4:
                try:
                    quad=core.ordered_quad(self.points,self._read()["image_size"],self._read()["roi"])
                    require_free_manual_area(self._read(),quad,exclude_plate_id=exclude,session=None if getattr(self,'reference_source',None) else self.h.session,gid=self.h.current_invocation)
                except (ValueError,TypeError) as e:
                    self.points.pop()
                    if isinstance(e,GTRCollisionError) and not getattr(self,'reference_source',None):
                        from .mobile_canvas_reservation import handle
                        handle(self,e.reference)
                    else:self.message(str(e)+" · popraw czwarty punkt albo Esc.")
                    self.request_overlay()
                    return "break"
                target=self.draw_reference_target
                if getattr(self,'reference_source',None):
                    oid=self._commit(lambda:self.reference_source.outline(quad),'Zapisano obrys tablicy')
                elif self.draw_target and self.draw_target['kind']=='scene_complete':
                    from ..ranking.mobile_scene_frame_review import edit
                    intent=self.draw_target
                    def complete_scene_outline():
                        edit(self.h.session,intent['gid'],intent['oid'],quad=quad)
                        return intent['oid']
                    oid=self._commit(complete_scene_outline,'Uzupełniono obrys tej samej fizycznej tablicy')
                elif self.draw_target and self.draw_target['kind']=='create':
                    from ..ranking.mobile_mt_geometry_layers import create_reference
                    intent=self.draw_target
                    oid=self._commit(lambda:create_reference(self.h.session,intent['gid'],intent['oid'],quad,
                        expected_binding=intent['binding'],expected_revision=intent['revision']),
                        'Utworzono jedną GTR dla wybranego MT')
                elif self.draw_target and self.draw_target['kind']=='replace':
                    from ..ranking.mobile_candidate_stacks import redraw_reference
                    intent=self.draw_target
                    oid=self._commit(lambda:redraw_reference(self.h.session,self.h.current_invocation,intent,quad),
                                     'Zastąpiono obrys wybranej GTR')
                elif target:
                    from ..ranking.mobile_candidate_stacks import complete_bbox_reference
                    gid,oid,pid=target
                    if gid!=self.h.current_invocation:
                        self.message('Zmieniono obraz; anuluj szkic przez Esc. Nie zapisano GT.')
                        return 'break'
                    oid=self._commit(lambda:complete_bbox_reference(self.h.session,gid,oid,quad,expected_plate_id=pid,operator_draw=True),
                                     'Narysowano obrys operatora dla istniejącej GT')
                else:
                    oid=self._commit(lambda:core.add_plate(self.h.session,self.h.current_invocation,quad,
                                      partial_review=True),"Dodano poprawną ramkę tablicy operatora")
                if oid:
                    self.selected=oid
                    self.draw_reference_target=None
                    self.draw_target=None
                    self.points=[];self.pointer=None;self.mode="edit"
                    self.sync()
            self.request_overlay()
            return "break"
        self.pointer_canvas = (event.x, event.y)
        target=self.hit(event.x,event.y,prefer_stack=True)
        if target:
            oid,kind,index=target
            if kind=='bbox_spatial_menu':
                from .mobile_canvas_bbox_marker import popup
                popup(self,event)
                return 'break'
            if kind=="badge":
                # A text editor owns its target; it does not change geometry selection.
                from .mobile_canvas_info_badge import activate_inspection
                if not getattr(self,'reference_source',None) and not activate_inspection(self,oid):return 'break'
                self._bad_mt_selection=None
                self.start_entry(oid)
            elif kind=="corner" and self.mode=="edit" and oid==self.selected:
                self._bad_mt_selection=None
                self.gt_focus_id=oid
                self._s_hover_hit=None
                self.message('Korekta GTR — przeciągnij narożnik; GTZ edytujesz na badge. Esc kończy korektę.')
                obj=next(o for o in self._read()["objects"] if o["object_id"]==oid)
                from .mobile_canvas_reservation import focused_row
                from .mobile_canvas_clean_view import records,selected
                row=selected(self)
                reserved=focused_row(self,records(self));q=reserved['gt'] if reserved else row['gt'] if row and row['gt'] else self._current_quad(obj) if getattr(self,'reference_source',None) else obj['quad']
                self.drag=dict(oid=oid,index=index,quad=copy.deepcopy(q),before=copy.deepcopy(q))
                if row and row.get('seed'):
                    self.drag.update(seed=True,revision=self.h.session.review.review_revision,binding=copy.deepcopy(self._read()['source_binding']))
                if hasattr(self,'navigator'):
                    self.collision_warning.clear();self.corner_pulse.clear()
                    from .mobile_canvas_corner_drag import paint as paint_drag
                    if paint_drag(self,publish=False):
                        self.request_overlay();return 'break'
            elif kind in ("body", "mz_badge") or kind.startswith('layer:'):
                # A short click selects; a genuine drag pans the image.
                self._click_candidate = (oid, event.x, event.y)
                self._click_layer=kind.split(':',1)[1] if kind.startswith('layer:') else None
                self.h._view_drag_at=(event.x,event.y,self.h._view_center)
            else:
                self.h._view_drag_at=(event.x,event.y,self.h._view_center)
            self.request_overlay()
            return "break"
        # Panning is permitted also when a GT contour is selected.
        from .mobile_canvas_selection import arm
        arm(self,event,had_entry=had_entry)
        self.h._view_drag_at=(event.x,event.y,self.h._view_center)
        self.request_overlay()
        return "break"

    def motion(self,event):
        if getattr(self,'reference_source',None) and self.reference_source.location_motion(event):
            return 'break'
        from .mobile_canvas_selection import motion as selection_motion
        selection_motion(self,event)
        if self.drag:
            self.pointer_canvas=(event.x,event.y)
            p=self.to_source(event.x,event.y)
            s=self._read(); w,h=s["image_size"]
            obj=next(o for o in s["objects"] if o["object_id"]==self.drag["oid"])
            l,t,r,b=s["roi"] if not obj["parent_id"] else (0,0,w,h)
            p=(max(l,min(r,p[0])),max(t,min(b,p[1])))
            self.drag["quad"][self.drag["index"]]=list(p)
            from .mobile_canvas_corner_drag import paint as paint_drag
            paint_drag(self,publish=False)
            self.request_overlay()
            return "break"
        if self._click_candidate is not None:
            _, x0, y0 = self._click_candidate
            if math.hypot(event.x-x0, event.y-y0) > 5:
                self._click_candidate = None
        return self.pan_motion(event)

    def release(self,event=None):
        if getattr(self,'reference_source',None) and self.reference_source.location_release(event):
            return 'break'
        click_candidate = self._click_candidate
        self._click_candidate = None
        was_dragging_corner = self.drag is not None
        if self.drag:
            state=self.drag
            if state["quad"] != state["before"]:
                from .mobile_canvas_stacks import write_reference
                if state.get('seed'):
                    from ..ranking.mobile_mt_geometry_layers import create_reference
                    ok=self._commit(lambda:create_reference(self.h.session,self.h.current_invocation,state['oid'],state['quad'],
                        expected_binding=state['binding'],expected_revision=state['revision']),"Zapisano GTR operatora dla wybranego MT")
                else:ok=self._commit(lambda:write_reference(self,state['oid'],quad=state['quad']),"Zapisano korektę referencji GT")
                if not ok:
                    if getattr(self,'reference_source',None):
                        self.request_overlay()
                        return 'break'  # keep the failed draft; navigation retries
                    self.message("Nie zapisano nieprawidłowego obrysu. Przywrócono ostatnią poprawną ramkę.")
            self.drag=None
            self.request_overlay()
        if not was_dragging_corner:
            self.pan_release(event)
        self.h._view_drag_at=None
        # Select only after a stationary click. Moving the contour itself
        # continues to pan without changing the selected tablicę.
        if (not was_dragging_corner and click_candidate is not None
                and event is not None and time.monotonic()-getattr(self,'_left_press_time',time.monotonic())<=.25):
            oid, x0, y0 = click_candidate
            if math.hypot(event.x-x0, event.y-y0) <= 5:
                kind=getattr(self,'_click_layer',None)
                if kind:
                    from .mobile_canvas_navigator import select_layer
                    from .mobile_canvas_clean_view import records
                    row=next((r for r in records(self) if r['oid']==oid),None)
                    if row and row['candidate'] and select_layer(self,kind,oid):return 'break'
                boxes=[getattr(self,'mt_source_badge_hits',{}).get(oid),getattr(self,'mt_stack_badges',{}).get(oid)]
                source=any(box and box[0]<=x0<=box[2] and box[1]<=y0<=box[3] for box in boxes)
                self._choose_plate(oid,source=source)
        if not was_dragging_corner:
            from .mobile_canvas_selection import release as selection_release
            selection_release(self,event)
        return "break"

    def hover_motion(self, event):
        self.pointer_canvas = (event.x, event.y)
        s = self._read()
        if not s or self.h._image_entry != s["image_entry"] or self.h._view_center is None:
            self._s_hover_hit = None
            return
        if self.mode == "draw":
            self._s_hover_hit = None
            self.pointer = self.to_source(event.x, event.y)
            if hasattr(self,'clean_bar'):
                from .mobile_canvas_guides import update_draft
                update_draft(self)
            else:self.request_overlay()
        elif not self.drag:
            target = self.hit(event.x, event.y, badge=True)
            selection_target=target
            if hasattr(self,'navigator'):
                from .mobile_canvas_navigator import selection_hit
                selection_target=selection_hit(self,event.x,event.y)
            self._s_hover_hit = ((self.h.current_invocation, selection_target[0],
                                  (event.x, event.y)) if selection_target else None)
            cursor = "fleur" if target and target[1] in ("corner", "photo_location") else "hand2" if target and (target[1] in ("badge", "body", "mz_badge") or target[1].startswith('layer:')) else "arrow"
            if self.c.cget("cursor") != cursor:
                self.c.configure(cursor=cursor)

    def right_press(self, event):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:return a3.menu(event)
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.context_menu(event)
        """One right-click on a plate toggles its error flag immediately.

        PPM directly targets the visible FINAL contour. Empty canvas still
        pans; correcting GT does not expose hidden inference frames.
        """
        # Refuse a GTR hit before finishing text, clearing D, selecting an MT,
        # or touching undo/redo. Shared relation IDs do not identify a layer.
        self._outside_click=None
        clean_oid=None
        from .mobile_canvas_geometry_layers import contains as layer_badge_hit
        if layer_badge_hit(self,event.x,event.y):return 'break'
        if hasattr(self,'clean_bar') and not getattr(event,'state',0)&4:
            from .mobile_canvas_clean_view import right_hit
            try:clean_oid=right_hit(self,event.x,event.y)
            except ValueError as exc:
                self.message(str(exc));return 'break'
        if not self.finish_entry():
            return "break"
        if getattr(event,'state',0)&4:
            from .mobile_canvas_stacks import context_menu
            context_menu(self,event)
            return 'break'
        if self.mode == "draw":
            return self.escape()
        if not self.available():
            return "break"
        self.pointer_canvas = (event.x, event.y)
        if hasattr(self,'clean_bar'):
            oid=clean_oid
            if oid:
                from .mobile_canvas_clean_view import selected
                row=selected(self)
                if row and row.get('seed') and row['oid']==oid:
                    self.gt_focus_id='';self._reserved_edit=None
                from ..ranking.mobile_candidate_stacks import toggle_unassociated
                self._click_candidate=None;self.h._view_drag_at=None
                self.selected=oid;self.mode='edit';self.show_fp_var.set(True)
                self._bad_mt_selection=(id(self.h.session),self.h.current_invocation,oid)
                self._commit(lambda:toggle_unassociated(self.h.session,self.h.current_invocation,oid,
                             partial_review=True),'Zmieniono decyzję wskazania MT FINAL')
            else:self.h._view_drag_at=(event.x,event.y,self.h._view_center)
            return 'break'
        target = self.hit(event.x, event.y)
        from .mobile_canvas_stacks import ambiguous
        if target and ambiguous(self,event.x,event.y,target):
            self.message('Wybierz podramkę (Z/X lub badge) przed decyzją PPM.')
            return 'break'
        from .mobile_canvas_stacks import enhanced,view,members
        if target and enhanced(self):
            v=view(self)
            candidate=next((c for c in v['candidates'] if c['object_id']==target[0]),None)
            if candidate:
                try:core.require_source_final(self.h.session,self.h.current_invocation,candidate['object_id'])
                except ValueError as exc:
                    self.message(str(exc));return 'break'
                if not self.selected and candidate['plate_id'] and len(members(v,candidate['plate_id']))>1:
                    self.message('Wybierz podramkę (Z/X) przed decyzją PPM.');return 'break'
                oid= self.selected if self.selected==target[0] else target[0]
                from ..ranking.mobile_candidate_stacks import toggle_unassociated
                self.selected=oid;self.mode='edit'
                self._bad_mt_selection=(id(self.h.session),self.h.current_invocation,oid)
                self._commit(lambda:toggle_unassociated(self.h.session,self.h.current_invocation,oid),'Zmieniono powiązanie wskazania z tablicą')
                return 'break'
        if target:
            oid = target[0]
            obj = next((o for o in self._read()["objects"]
                        if o["object_id"] == oid and not o.get("post_nms_suppressed", False)), None)
            if obj is not None:
                try:core.require_source_final(self.h.session,self.h.current_invocation,oid)
                except ValueError as exc:
                    self.message(str(exc));return 'break'
                self._click_candidate = None
                self.h._view_drag_at = None  # PPM on a frame must not initiate pan.
                self.selected = oid
                self._bad_mt_selection=(id(self.h.session),self.h.current_invocation,oid)
                self.mode = "edit"
                description = ("Cofnięto oznaczenie błędnej ramki" if obj["error"] else
                               "Oznaczono wskazaną ramkę jako błędną")
                self._commit(
                    lambda: core.toggle_error(self.h.session, self.h.current_invocation, oid),
                    description,
                )
                self.request_overlay()
                return "break"
        self.h._view_drag_at = (event.x, event.y, self.h._view_center)
        return "break"

    def pan_motion(self,event):
        state=self.h._view_drag_at
        if state and self.h._image is not None:
            x,y,center=state
            if center:
                z=max(self.h._view_pixel_scale,1e-12)
                from .mobile_canvas_pan import request_pan
                request_pan(self.h,(center[0]-(event.x-x)/z,center[1]-(event.y-y)/z))
        return "break"

    def pan_release(self,event=None):
        from .mobile_canvas_pan import finish_pan
        if event is not None:
            self.pan_motion(event)  # Include the last pointer position before release.
        finish_pan(self.h)
        self.h._view_drag_at=None
        return "break"

    def back_point(self,event=None):
        if self.mode=="draw" and self.points:
            self.points.pop();self.request_overlay()
        else:
            self.no_delete()
        return "break"

    def escape(self,event=None):
        if getattr(self,'reference_source',None):self.reference_source.cancel_location()
        self._outside_click=None
        if hasattr(self,'clean_bar'):
            from .mobile_canvas_info_badge import remember_on_escape
            remember_on_escape(self)
        self.collision_warning.clear()
        self.corner_pulse.clear()
        self.shift_preview.clear()
        self.gt_focus_id='';self._reserved_edit=None;self._geometry_layer_state=None
        self.history_selected = ''
        if self.entry is not None:
            self.entry_var.set(self.entry_original)
            self.destroy_entry()
            self.c.focus_set()
        self.points=[];self.pointer=None;self.drag=None;self.mode="select"
        self.draw_reference_target=None
        self.draw_target=None
        self._click_candidate = None
        self.selected=self.hover=""
        self.pending_tree_target = None
        self.h._view_drag_at=None
        # Explicit cancellation clears an image/group row too. Ordinary sync
        # must still preserve parent-row navigation when no plate is selected.
        self.clear_tree_selection()
        self.sync()
        self.request_overlay()
        return "break"

    def start_entry(self,oid):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:
            self.message('Tekst pochodzi z kanonicznego GT. PPM pozwala przypisać tablicę lub oznaczyć odczyt N/A.');return
        requested_oid=oid
        from .mobile_canvas_stacks import enhanced,view
        from .mobile_canvas_spatial_reference import entry_reference
        reference=entry_reference(self,oid)
        if reference:
            oid=reference['reference_id']
        elif enhanced(self):
            v=view(self);candidate=next((c for c in v['candidates'] if c['object_id']==oid),None)
            if candidate and candidate['plate_id']:
                p=v['plates'][candidate['plate_id']]
                registry=self._read().get('physical_gt_v1',{})
                ref=next((plate['reference_object_id'] for plate in registry.get('plates',[]) if plate['gt_id']==candidate['plate_id']),None)
                oid=ref or p.get('primary_object_id') or p['origin_object_id']
        if self.gt_focus_id not in ('',requested_oid,oid):self.gt_focus_id=''
        if not self.available():
            return
        policy = self._gt_policy(oid)
        if not policy["gt_editable"]:
            self.message(policy["gt_reason"])
            return
        if self.entry is not None:
            if self.entry_target == (self.h.current_invocation, oid):
                self.entry.focus_set()
                return
            if not self.finish_entry():
                return
        obj=next(o for o in self._read()["objects"] if o["object_id"]==oid)
        if obj["error"]:
            self.message("Ramka ma flagę błędu. Jeden klik PPM na ramce cofa tę flagę.")
            return
        self.entry_target=(self.h.current_invocation,oid)
        value=self.reference_source.display_text(obj) if getattr(self,'reference_source',None) else obj['gt_text']
        self.entry_original=value
        self.entry_var=tk.StringVar(value=value)
        self.entry=tk.Entry(self.c.master if hasattr(self,'clean_bar') else self.c,textvariable=self.entry_var,font=("Segoe UI",11,"bold"),
            bg="#302618",fg="#FFE2AE",insertbackground="#FFFFFF",relief="flat")
        # Widget+class only: ordinary text editing remains, global Z2/app
        # shortcuts cannot consume D/S/Q/E typed as a plate number.
        self.entry.bindtags((str(self.entry),self.entry.winfo_class()))
        self.entry.bind("<Return>",lambda e:self.entry_enter())
        self.entry.bind("<KP_Enter>",lambda e:self.entry_enter())
        self.entry.bind("<Escape>",self.escape)
        self.entry.bind("<FocusOut>",self.entry_focus_out)
        self.entry.bind('<Tab>',lambda e:self.entry_tab(bool(e.state & 1)))
        self.entry.bind('<Shift-Tab>',lambda e:self.entry_tab(True))
        self.entry.bind('<ISO_Left_Tab>',lambda e:self.entry_tab(True))
        self.entry.bind("<KeyRelease>",self.entry_typed)
        if getattr(self,'reference_source',None):
            self.entry.bind('<Control-s>',lambda e:self.entry_enter())
            self.entry.bind('<Control-S>',lambda e:self.entry_enter())
        self.entry.bind("<Button-3>",self.reference_source.context_menu if getattr(self,'reference_source',None) else lambda e:self.unreadable())
        self.request_overlay()
        x,y=self.card_positions.get(oid,(4,4))
        if hasattr(self,'clean_bar'):
            from .mobile_canvas_clean_view import place_entry
            self.overlay();place_entry(self)
        else:self.entry.place(x=x+1,y=y+25,width=192,height=23)
        self.entry.focus_set();self.entry.selection_range(0,"end")
        self.sync()
        self.message("Edytujesz GTZ · Enter lub wyjście zapisuje; Esc cofa do ostatniego zapisu. PPM w polu: nieczytelna tablica.")

    def entry_tab(self, backwards=False):
        if self.entry is None:return 'break'
        target=self.entry.tk_focusPrev() if backwards else self.entry.tk_focusNext()
        if self.finish_entry() and target is not None:
            try:target.focus_set()
            except tk.TclError:pass
        return 'break'

    def _gt_policy(self, oid):
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.policy(oid)
        from ..ranking.mobile_canvas_presentation import execution_state, plate_review_state
        h = self.h
        try:
            # Re-read the source binding on entry and save, not only the cached
            # UI revision. Cancellation/NMS can invalidate an already open field.
            scene = core.scene(h.session, h.current_invocation)
            if scene.get('scope') == 'SCENE_FRAME':
                return dict(gt_editable=any(o['object_id'] == oid for o in scene['objects']), gt_reason='', gt_note='ASSISTED · pełna scena')
            execution = execution_state(h.session, h.current_invocation,
                image_entry=h._image_entry, image_size=h._image_source_size)
            from .mobile_canvas_spatial_reference import entry_reference
            reference=entry_reference(self,oid,scene)
            if reference:oid=reference['reference_id']
            obj = next((o for o in scene["objects"] if o["object_id"] == oid), None)
            policy=plate_review_state(scene, obj, execution)
            # F5A manual GT is entered before revealing/linking MT. A saved
            # hand-drawn reference must not require a model parent to edit GTZ.
            manual_reference=(getattr(self,'manual_gt_first',False) and obj
                and obj.get('kind')=='manual' and not obj.get('parent_id')
                and not obj.get('attempt_id') and not obj.get('error'))
            if ((reference or manual_reference) and execution.get('state')=='MT_EXECUTED'
                    and execution.get('mt_invocation_id')==scene['mt_invocation_id']
                    and execution.get('image_entry')==scene['image_entry']):
                policy=dict(policy,gt_editable=True,gt_reason='',gt_note='')
            if policy['gt_editable'] and obj and not obj.get('quad'):
                policy=dict(policy,gt_editable=False,gt_reason=
                    'MT nie ma poprawnych narożników, a wskazanie nie ma obrysu GT. '
                    'Wybierz poprawny BBOX i naciśnij D, aby uzupełnić obrys GT; istniejącą ramkę powiąż przez Ctrl+PPM.')
            return policy
        except (ValueError, KeyError, TypeError, AttributeError):
            return dict(gt_editable=False, gt_reason="Nie potwierdzono końcowej detekcji MT. Historyczne GT zachowano; edycja niedostępna.")

    def _guard_gt_entry(self):
        if self.entry is None:
            return True
        gid, oid = self.entry_target
        policy = self._gt_policy(oid) if gid == self.h.current_invocation else dict(
            gt_editable=False, gt_reason="Zmieniono obraz. Niezapisany tekst zamknięto; wcześniejsze GT zachowano.")
        if policy["gt_editable"]:
            return True
        self.destroy_entry()  # cancels autosave; never clears persisted GT
        self.message(policy["gt_reason"])
        return False

    def entry_typed(self,event=None):
        if self.entry is None:
            return
        if not self._guard_gt_entry():
            self.sync()
            return
        val=self.entry_var.get()
        if val != val.upper():
            pos=self.entry.index("insert")
            self.entry_var.set(val.upper());self.entry.icursor(pos)
        if self.entry_after:
            self.c.after_cancel(self.entry_after)
        self.entry_after=self.c.after(650,self.autosave_entry)
        self.request_overlay()

    def autosave_entry(self):
        self.entry_after=None
        if self.entry is not None:
            self.save_entry(close=False)

    def save_entry(self,close=True):
        if self.entry is None:
            return True
        if not self._guard_gt_entry():
            self.sync()
            return True
        gid,oid=self.entry_target
        if gid!=self.h.current_invocation:
            self.message("Nie zmieniono celu GT. Wróć do obrazu z aktywną edycją.")
            return False
        value=self.entry_var.get()
        if value!=self.entry_original:
            from .mobile_canvas_stacks import write_reference
            if not self._commit(lambda:write_reference(self,oid,text=value),"Zapisano GT wybranej tablicy"):
                return False
            self.entry_original=value
        if close:
            self.destroy_entry()
            self.sync()
        return True

    def finish_entry(self):
        return self.save_entry(close=True)

    def entry_focus_out(self, event=None):
        if getattr(self,'reference_source',None) and self.reference_source.menu_open:return 'break'
        if getattr(self.h,'_export_has_focus',lambda:False)():
            self.prepare_export()
            return 'break'
        return self.finish_entry()

    def entry_enter(self):
        if self.finish_entry():
            self.c.focus_set()
        return "break"

    def destroy_entry(self):
        if self.entry_after:
            self.c.after_cancel(self.entry_after);self.entry_after=None
        entry,self.entry=self.entry,None
        self.entry_target=None
        if entry is not None:
            entry.destroy()
        self.request_overlay()

    def unreadable(self):
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.unreadable()
        if self.entry is None:
            return "break"
        if not self._guard_gt_entry():
            self.sync()
            return "break"
        gid,oid=self.entry_target
        reason=simpledialog.askstring("Nieczytelny numer", "Przyczyna nieczytelności tej tablicy:",parent=self.h.window)
        if self.entry_target != (gid, oid) or not self._guard_gt_entry():
            self.sync()
            return "break"
        from .mobile_canvas_stacks import write_reference
        if reason and self._commit(lambda:write_reference(self,oid,text='',reason=reason),"Zapisano przyczynę nieczytelności"):
            self.destroy_entry()
            self.sync()
        return "break"

    def prepare_export(self):
        """Read-only preflight; keep drafts and all ordinary gestures intact."""
        if self.entry_after:
            self.c.after_cancel(self.entry_after);self.entry_after=None
        clean_reason=getattr(self,'clean_exclusion',None)
        reason_open=bool(clean_reason is not None and clean_reason.winfo_manager())
        scene=self._read()
        reason_dirty=bool(scene and self.reason_gid==self.h.current_invocation
            and self.excluded_var.get() and scene.get('exclusion_reason','')!=self.reason_var.get().strip())
        if (self.entry is not None or self.points or self.drag or self.mode=='draw'
                or reason_open or reason_dirty):
            self.message('Zakończ lub anuluj edycję przed eksportem. Eksport odczytuje zapisany stan.')
            if self.entry is not None:self.entry.focus_set()
            return False
        if self.in_refresh or self.busy:
            self.message('Poczekaj na zakończenie zapisu przed eksportem.')
            return False
        return True

    def prepare_navigation(self):
        if getattr(self,'reference_source',None) and not self.reference_source.prepare_location():return False
        from .mobile_canvas_pan import cancel_pan
        cancel_pan(self.h)
        self.collision_warning.clear()
        self.shift_preview.clear()
        if self.points:
            self.message("Dokończ ramkę (4 punkty) albo anuluj D/Esc przed zmianą obrazu. Szkic pozostaje na canvasie.")
            return False
        if self.drag:
            self.release()
            if self.drag and getattr(self,'reference_source',None):return False
        if not self.finish_entry():
            return False
        return self._flush_exclusion_reason()

    def confirm(self, event=None):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:return a3.primary(event)
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.confirm(event)
        if event is not None:
            if (not self._event_in_review_window(event)
                    or self._is_typing_shortcut(getattr(event, 'widget', self.c))):
                return None
            if getattr(event, 'keysym', '') == 'space' and getattr(event, 'state', 0) & 4:
                return None
        if getattr(event, 'keysym', '') == 'space':
            if getattr(self, '_frame_space_down', False):
                return 'break'
            self._frame_space_down = True
        if self.mode=='draw':
            self.message('Dokończ ramkę lub anuluj szkic przed zatwierdzeniem.')
            return 'break'
        if not self.prepare_navigation():
            return "break"
        deferred = self.in_refresh and self.h._saving
        if not deferred and not self.available():
            return "break"
        from ..ranking import mobile_canvas_partial_review as partial
        current = partial.progress(self._read())
        from .mobile_canvas_clean_view import selected,records
        row=selected(self)
        oid = self.selected
        if not oid:
            xy = self._canvas_hotkey_pointer(event)
            target = self.hit(*xy,prefer_stack=True) if xy is not None else None
            oid = target[0] if target else ''
            if target:
                badge=getattr(self,'_context_badge',None)
                reference=target[1] in ('badge','corner','layer:GTR')
                pid=badge['plate_id'] if reference and badge and badge['oid']==oid and badge['kind']=='GTR' else None
                rows=records(self)
                row=next((r for r in rows if r['pid']==pid and r['gt']),None) if pid else next(
                    (r for r in rows if r['oid']==oid and (r['gt'] if reference else r['candidate'])),None)
        if not oid:
            if current['empty_eligible']:
                return self._queue_frame_verification(None) if deferred else self.confirm_empty_image()
            self.message('Wybierz ramkę tablicy lub najedź na nią i naciśnij Spację: przełącz weryfikację skojarzonych ramek.')
            return 'break'
        pid=row['pid'] if row and row['gt'] else None
        if not pid and oid not in current['object_decisions']:
            self.message('Wybierz GTR albo końcowe wskazanie modelu tej tablicy.')
            return 'break'
        target=dict(oid=oid if not pid else None,plate_id=pid)
        from .mobile_canvas_geometry_layers import model as current_layers
        layer=current_layers(self)
        bad_reference=bool(row and row['fp'] and (self.gt_focus_id or (layer and layer['active']=='GTR')))
        if row and row['fp'] and row['candidate'] and not (bad_reference and row['plate']):
            target=dict(oid=row['candidate']['object_id'],plate_id=None)
        if row and not row['fp'] and not bad_reference and row['candidate'] and row['pid'] and row['candidate'].get('spatial_relation',{}).get('plate_id')==row['pid']:
            target['spatial_oid']=row['candidate']['object_id']
        if deferred:
            return self._queue_frame_verification(target)
        return self._toggle_frame_verification(target)

    def _toggle_frame_verification(self, target):
        if target is None:
            return self.confirm_empty_image()
        from ..ranking import mobile_canvas_partial_review as partial
        args={k:v for k,v in target.items() if k!='spatial_oid'}
        from .mobile_canvas_spatial_reference import canonical_action
        verified=partial.stack_is_verified(self._read(),**(
            dict(oid=target['spatial_oid']) if target.get('spatial_oid') else args))
        action=(lambda:canonical_action(self.h.session,self.h.current_invocation,target['spatial_oid'],target['plate_id'],operation='toggle')) if target.get('spatial_oid') else (lambda:partial.toggle_stack(self.h.session,self.h.current_invocation,**args))
        self._commit(action,
                     'Cofnięto weryfikację skojarzonych ramek tablicy' if verified else 'Potwierdzono skojarzone ramki tablicy')
        return "break"

    def _queue_frame_verification(self, target):
        # Statistics temporarily owns the session. Keep each released press,
        # bound to its original source/target; never write from its worker.
        context=(id(self.h.session),self.h.current_invocation,self.h._image_entry)
        self._pending_frame_toggles=getattr(self,'_pending_frame_toggles',[])+[(context,target)]
        if not getattr(self,'_frame_toggle_after',None):
            self._frame_toggle_after=self.c.after(40,self._drain_frame_verification)
        self.message('Przełączenie weryfikacji oczekuje na zakończenie odświeżania wyników.')
        return 'break'

    def _drain_frame_verification(self):
        if getattr(self,'_frame_toggle_after',None):
            self.c.after_cancel(self._frame_toggle_after)
        self._frame_toggle_after=None
        if self.h._closed or not getattr(self,'_pending_frame_toggles',[]):
            self._pending_frame_toggles=[]
            return
        if self.h._saving or self.in_refresh:
            self._frame_toggle_after=self.c.after(40,self._drain_frame_verification)
            return
        pending=getattr(self,'_pending_frame_toggles',[])
        self._pending_frame_toggles=[]
        context=(id(self.h.session),self.h.current_invocation,self.h._image_entry)
        for source,target in pending:
            if source==context:
                self._toggle_frame_verification(target)

    def release_frame_space(self, event=None):
        self._frame_space_down = False
        if getattr(event, 'keysym', '') == 'space' and not self._is_typing_shortcut(event.widget):
            return 'break'

    def confirm_empty_image(self):
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.empty()
        if not self.prepare_navigation() or not self.available():
            return 'break'
        if self._read().get('scope') == 'SCENE_FRAME':
            from ..ranking.mobile_scene_frame_review import confirm_empty
            self._commit(lambda: confirm_empty(self.h.session, self.h.current_invocation), 'Potwierdzono brak tablic na pełnej scenie')
            return 'break'
        self.message('Brak ramek i zapisanych referencji: miary N/A. Pominiętą tablicę można dorysować przez D.')
        return 'break'

    def exclude_image(self):
        self.excluded_var.set(True)
        self.toggle_excluded()

    def cycle(self,event=None):
        if not self.prepare_navigation() or not self.available():
            return "break"
        ids=[o["object_id"] for o in self._read()["objects"] if not o.get("post_nms_suppressed", False)]
        from .mobile_canvas_stacks import enhanced,view,active
        if enhanced(self):
            v=view(self)
            ids=[(active(self,v,pid) or {}).get('object_id') or p['origin_object_id'] for pid,p in v['plates'].items()]
        if hasattr(self,'clean_bar'):
            from .mobile_canvas_clean_view import records
            ids=[r['oid'] for r in records(self)]
            self.gt_focus_id=''
        if ids:
            self._bad_mt_selection=None
            self.selected=ids[(ids.index(self.selected)+1)%len(ids)] if self.selected in ids else ids[0]
            self.mode="edit"
            self.plate_fit_previous=None
            self.plate_fit_active=False
            self.plate_fit_gid=""
            self.sync();self.request_overlay()
        return "break"

    def navigate(self,direction):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:return a3.navigate(direction)
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.navigate(direction)
        if not self.prepare_navigation() or self.h._saving:
            return "break"
        from .mobile_canvas_observation_blocks import navigate
        if navigate(self, direction):
            return "break"
        # Q/E move between actually displayed images, never via a hidden
        # duplicate source record (e.g. source entry 20 linked with image 19).
        tree = self.h.subject_tree
        ids = list(tree.get_children(""))
        if not ids:
            return "break"
        current = self.tree_branch if self.tree_branch in ids else None
        if current is None:
            current = next((i for i,k in self.h._subject_ids.items()
                            if i in ids and k == self.h.subject_key), None)
        if current is None:
            current = next((owner for alias,(owner,gid) in self.linked_tree_rows.items()
                            if self.h._subject_ids.get(alias)==self.h.subject_key and owner in ids),None)
        idx = ids.index(current) if current in ids else 0
        target = ids[max(0,min(len(ids)-1,idx+direction))]
        if target != current:
            tree.selection_set(target)
            tree.focus(target)
            tree.see(target)
            self.h._select_subject()
        return "break"

    def fit(self):
        self.plate_fit_active=False
        self.plate_fit_gid=""
        self.plate_fit_previous=None
        self.plate_fit_saved=None
        self.h.fit_image()
        return "break"

    def toggle_roi_overlay(self, event=None):
        if getattr(self, 'reference_source', None) is not None:
            return 'break'
        self.show_roi_overlay = not self.show_roi_overlay
        self.request_overlay()
        self.message("Obszar analizy: widoczny" if self.show_roi_overlay
                     else "Obszar analizy: ukryty tylko na ekranie; ocena pozostaje bez zmian")
        return "break"

    def fit_plate_toggle(self, event=None):
        """A / toolbar: cycle physical plates and fit the next one, as in Z2.

        Same approach as Z2: bbox in original image pixels, 20% horizontal /
        25% vertical margin (at least 16 px), centering via actual renderer's
        _view_zoom / _view_center, no GT or prediction modification.
        """
        # The real Windows session recorded S with Tk state=8 (Mod1).
        # A accepts Windows Mod1, but leaves Ctrl+A and GT typing untouched.
        if event is not None and getattr(event, "state", 0) & 4:
            return "break"
        if event is not None and self._is_typing_shortcut(getattr(event, "widget", self.c)):
            return None
        scene = self._read()
        if (not scene or self.h._image is None or self.h._view_center is None
                or self.h._image_entry != scene["image_entry"]
                or tuple(getattr(self.h,"_image_source_size", ()) or ()) != tuple(scene["image_size"])):
            self.message("A: zbliżenie wymaga pełnej klatki z prawidłową geometrią.")
            return "break"
        if self.drag or self.points or self.mode=='draw' or self.entry is not None:
            self.message("A: najpierw zakończ lub anuluj edycję numeru, narożnika albo szkic.")
            return "break"
        from . import mobile_canvas_plate_cycle as cycle
        choices=cycle.targets(self);target=cycle.next_target(self,choices)
        quad=target['quad'] if target else None
        if not quad:
            self.message("A: na tym obrazie nie ma ramek tablic do przełączania.")
            return "break"
        try:
            try:
                self.c.update_idletasks()
            except (AttributeError, tk.TclError):
                pass
            iw, ih = self.h._image.size
            source_w, source_h = map(float, scene["image_size"])
            if not all((iw > 0, ih > 0, source_w > 0, source_h > 0)):
                raise ValueError("Brak rozmiaru źródła")
            xs = [float(p[0]) * iw / source_w for p in quad]
            ys = [float(p[1]) * ih / source_h for p in quad]
            if not all(math.isfinite(z) for z in xs + ys):
                raise ValueError("Nieskończone współrzędne")
            bx1, bx2, by1, by2 = min(xs), max(xs), min(ys), max(ys)
            bw, bh = bx2 - bx1, by2 - by1
            if bw < 1e-4 or bh < 1e-4:
                raise ValueError("Brak wymiarów ramki")
            pad_x = max(16.0, bw * .20)
            pad_y = max(16.0, bh * .25)
            left, right = max(0.0, bx1 - pad_x), min(iw, bx2 + pad_x)
            top, bottom = max(0.0, by1 - pad_y), min(ih, by2 + pad_y)
            cw, ch = self.c.winfo_width(), self.c.winfo_height()
            if cw < 80 or ch < 80:
                raise ValueError("Za mały canvas")
            scale = min(cw / max(1., right-left), ch / max(1.,bottom-top))
            fit = max(1e-8, min((cw - 16) / iw, (ch - 16) / ih))
            zoom = max(1.0, min(256.0, scale / fit))
            center = ((left + right) * .5, (top + bottom) * .5)
        except (TypeError, ValueError, IndexError, OverflowError, ZeroDivisionError):
            self.message("A: nie można dopasować tablicy. Sprawdź współrzędne jej narożników.")
            return "break"
        if not cycle.select(self,target):return 'break'
        if not self.plate_fit_active or self.plate_fit_gid!=self.h.current_invocation:
            self.plate_fit_previous = (self.h._view_zoom, tuple(self.h._view_center))
        self.plate_fit_active = True
        self.plate_fit_gid = self.h.current_invocation
        self.h._view_zoom = zoom
        self.h._view_center = center
        self.h._paint_image()
        self.message(f"Tablica {choices.index(target)+1}/{len(choices)} · A: następna tablica · F: cały obraz")
        return "break"

    def show_crop(self):
        # Resolve the selected source directly. Navigating the hidden record
        # tree can load asynchronously and leave a different crop in its cache.
        if not self._read() or self.mode=='draw' or self.drag:return
        try:core.require_source_final(self.h.session,self.h.current_invocation,self.selected)
        except (ValueError,KeyError,TypeError):return
        obj=core._object(self._read(),self.selected)
        attempt=self.h.session.attempts[obj['attempt_id']]
        self.h._sync_crop_preview_source(attempt)
        self.h._show_crop_preview()

    def undo(self,redo=False):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:return a3.undo(redo)
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.restore(redo=redo)
        self.collision_warning.clear()
        if not self.prepare_navigation() or not self.available():
            return "break"
        if self._read().get('scope') == 'SCENE_FRAME':
            from ..ranking.mobile_scene_frame_review import restore
            self._commit(lambda: restore(self.h.session, self.h.current_invocation, redo=redo), 'Ponowiono ocenę sceny' if redo else 'Cofnięto ocenę sceny', remember=False)
            return 'break'
        if self.mode=='draw':
            self.mode='select';self.draw_target=None;self.draw_reference_target=None
        gid=self.h.current_invocation
        from . import mobile_bbox_spatial_review as bbox_review
        try:choice=bbox_review.history_choice(self.h.session,gid,redo=redo)
        except (ValueError,OSError,KeyError) as exc:
            self.message('Historia oceny BBOX niedostępna: '+str(exc));return 'break'
        if choice=='diagnostic':
            from .mobile_canvas_bbox_marker import apply
            apply(self,lambda:bbox_review.restore(self.h.session,gid,redo=redo))
            return 'break'
        if redo and choice is None and gid in bbox_review.load(self.h.session)['histories']:
            return 'break'
        from ..ranking.mobile_candidate_stacks import HISTORY,restore_history
        if HISTORY in self.h.session.invocation_annotation(gid).get(core.KEY,{}):
            self._commit(lambda:restore_history(self.h.session,gid,redo=redo),'Ponowiono wspólną GT' if redo else 'Cofnięto wspólną GT',remember=False)
            return 'break'
        src=self.redo_stack if redo else self.undo_stack
        dst=self.undo_stack if redo else self.redo_stack
        stack=src.setdefault(gid,[])
        if stack:
            prior=copy.deepcopy(stack[-1])
            now=copy.deepcopy(self._read())
            def change():
                live=core.scene(self.h.session,gid)
                if prior["parents"]!=live["parents"] or prior["source_binding"]!=live["source_binding"]:
                    raise ValueError("Historia dotyczy innego źródła")
                prior["events"]=copy.deepcopy(live["events"])
                prior.update(checked=False,exclusion_reason="")
                core._store(self.h.session,gid,prior,"redo_canvas_edit" if redo else "undo_canvas_edit")
            if self._commit(change,"Ponowiono zmianę" if redo else "Cofnięto zmianę operatora",remember=False):
                stack.pop();dst.setdefault(gid,[]).append(now)
        return "break"

    def dispose(self):
        if hasattr(self,'task_locator'):self.task_locator.clear()
        from .mobile_canvas_pan import cancel_pan
        cancel_pan(self.h)
        self.collision_warning.clear()
        self.corner_pulse.clear()
        self.shift_preview.dispose()
        """Stop controller callbacks before the window and its Tcl commands die."""
        self._pending_frame_toggles=[]
        for name in ("entry_after", "refresh_after", "paint_after", "selection_release_after", "fit_release_after", "_frame_toggle_after"):
            callback = getattr(self, name, None)
            if callback:
                try:
                    self.c.after_cancel(callback)
                except tk.TclError:
                    pass
                setattr(self, name, None)
        self.paint_scheduled = False
        token = getattr(self, "_win_s_poll_id", None)
        if token is not None:
            try:
                self.c.after_cancel(token)
            except (AttributeError, tk.TclError):
                pass
            self._win_s_poll_id = None
        self._remove_shortcut_fallback()
        tag = getattr(self, "shortcut_tag", None)
        if tag:
            for widget, original in tuple(self.shortcut_original_tags.items()):
                try:
                    if tag in widget.bindtags():
                        widget.bindtags(original)
                except (tk.TclError, AttributeError):
                    pass
            for key in ("s", "S", "a", "A"):
                for action in ("KeyPress", "KeyRelease"):
                    try:
                        self.c.unbind_class(tag, "<" + action + "-" + key + ">")
                    except (tk.TclError, AttributeError):
                        pass
            for key in ('z','Z','y','Y'):
                try:self.c.unbind_class(tag,'<Control-KeyPress-'+key+'>')
                except (tk.TclError,AttributeError):pass
            self.shortcut_original_tags.clear()
            self.shortcut_tag = None

    def build_report_tab(self, tabs):
        self.report_host=CanvasResultsPanel(tabs,p=self.p,on_open_image=self.open_result_image)
        self.review_tabs=tabs
        tabs.tab(1,text="Analiza historyczna")
        tabs.add(self.report_host,text="Canvas · wyniki")
        tabs.insert(1,self.report_host)
        # Keep the action's raised CTA styling; only the caption changes.
        self.h.science_toggle.configure(text="Tabela techniczna")

    def update_report(self):
        from ..ranking.mobile_canvas_presentation import current_summary
        from ..ranking.mobile_session_metrics import result_key
        if hasattr(self.contexts,'live_metrics'):
            self.contexts.live_metrics.render(self.h.stats.get('live_session_metrics_v2'),result_key(self.h.session))
        if not hasattr(self,"report_host") or self.h.session is None:
            return
        inv=self.inventory()
        self.update_header(inv)
        data = self.h.stats.get("canvas_review") or {}
        mt = self.h.stats.get('mt_object_evaluation_v2') or {}
        if not data and (mt.get('session_id')==self.h.session.session_id
                and mt.get('source_archive_sha256')==self.h.session.review.source_archive_sha256
                and mt.get('review_revision')==self.h.session.review.review_revision):
            data = core.summary(self.h.session)
        current = current_summary(data, self.h.session)
        if current:
            data = dict(data, mt_object_evaluation_v2=mt)
        refreshing = not current and bool(data or self.in_refresh or self.refresh_after or self.h._saving)
        self.report_host.render(data if current else {}, inv, refreshing=refreshing,review_status=self.h.session.review.review_status)
        self.h.science_dashboard.render_canvas(data if current else None, inv,
            refreshing=refreshing, review_status=self.h.session.review.review_status)

    def _render_contexts(self, model):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:
            self.contexts.render(a3.context());a3.decorate();return
        if hasattr(self,'clean_bar'):
            from .mobile_canvas_clean_view import context,sync_image_actions
            from ..ranking.mobile_session_metrics import result_key
            value=context(self,model)
            from .mobile_canvas_visibility import gt_only
            value.update(session_metrics=None if gt_only(self) else self.h.stats.get('live_session_metrics_v2'),
                         session_metrics_key=result_key(self.h.session))
            self.contexts.render(value)
            sync_image_actions(self)
            return
        from ..ranking.mobile_canvas_presentation import review_contexts
        data = review_contexts(self.h.session, self._read(), model, self.inventory())
        if model.get('stack_caption'):data['object']=model['stack_caption']
        from .mobile_canvas_stacks import view as stack_view
        v=stack_view(self)
        if v and v['conflicts']:data['object']+='\nKonflikt historycznych GT — zachowano dawny widok; wspólna edycja zablokowana.'
        report = getattr(self.h, 'report', None)
        session = self.h.session
        if session and report and report.source_archive_sha256.lower() == session.review.source_archive_sha256.lower():
            key = (str(report.report_id), report.source_archive_sha256.lower())
            state = getattr(self.h.browser, '_comparison_verification', {}).get(key)
            if state and state.state == 'STALE':data['session'] += ' · eksport nieaktualny'
        self.contexts.render(data)

    def _render_drag_geometry(self,scene):
        """Only the selected four-corner preview changes before the save.

        Do not rebuild diagnostics/history, MT totals or session widgets for
        mouse pixels. The existing sync/worker publishes those after commit.
        """
        from ..ranking.mobile_canvas_presentation import geometry_assessment,percent
        obj=next((o for o in scene['objects'] if o['object_id']==self.drag['oid']),None)
        if obj is None:return
        preview=getattr(self,'_clean_geometry_preview',None)
        row=preview[2] if preview and preview[:2]==(self.cache_key,self.drag['oid']) else None
        if row and not row['not_measurable_reason']:
            # Keep the exact canonical diagnostic pair, including a secondary
            # FINAL sharing this GT. Its detector duplicate status does not
            # remove the existing, separate diagnostic geometry measurement.
            try:
                value=core.compare_mt_pose_to_gt(row['mt_quad'],self.drag['quad'],*scene['image_size'])
                caption=f"Polygon IoU {percent(value['polygon_iou_pct'])} · NMCE {percent(value['nmce_pct'])} · niezapisana korekta"
            except (ValueError,TypeError,KeyError,OverflowError):
                caption='IoU / NMCE — nieocenialny obrys operatora'
        else:
            model=geometry_assessment(scene,obj,self._canvas_execution,live_quad=self.drag['quad'])
            caption=model['geometry']
            if model['geometry_status']=='LIVE':caption+=' · niezapisana korekta'
        text='\n'.join('Położenie: '+caption if line.startswith('Położenie: ') else line
                       for line in self.contexts.data.get('object','').split('\n'))
        label=self.contexts.fields['object']
        if label.cget('text')!=text:label.configure(text=text)

    def _build_panel(self, right):
        from .mobile_canvas_ui import PlateAssessment, CanvasDecisionPanel, ReviewContexts
        p = self.p
        scroll = right.master
        header_host = scroll.master
        header_host.rowconfigure(0, weight=0)
        header_host.rowconfigure(1, weight=1)
        for child in header_host.winfo_children():
            if child.winfo_manager() == 'grid':child.grid_configure(row=1)
        self.contexts = ReviewContexts(header_host, p=p)
        self.contexts.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.panel = tk.Frame(right, bg=p["bg"], padx=1, pady=3)
        self.panel.grid(row=0, column=0, sticky="new")
        ui_label(self.panel, "DZIAŁANIA I SZCZEGÓŁY", p=p, bg=p["bg"], bold=True, size=9).pack(fill="x", pady=(0, 4))
        self.status = ui_label(self.panel, "Wczytywanie…", p=p, bg=p["blue_bg"], fg=p["blue"],
                               bold=True, wrap=self.sidebar_width - 48)
        self.status.pack(fill="x", ipady=6)
        self.decision_panel = CanvasDecisionPanel(self.panel, self, p=p)
        self.assessment = PlateAssessment(self.panel, p=p)
        self.assessment.pack(fill="x", pady=(7, 0))
        self.assessment.plain.pack_forget()
        self.assessment.on_render = self._render_contexts
        # Explain the causal chain *before* presenting a confirmation action.
        # Shown only if this scene contains a trace-backed 2->1 NMS result.
        self.reconciliation_box = tk.Frame(self.panel, bg=p["surface"], padx=11, pady=11,
                                           highlightbackground=p["border"], highlightthickness=1)
        self.reconciliation_title = ui_label(self.reconciliation_box, "", p=p, bg=p["surface"],
                                             fg=p["blue"], bold=True, size=10,
                                             wrap=self.sidebar_width - 70)
        self.reconciliation_title.pack(fill="x", pady=(0, 7))
        self.reconciliation_body = ui_label(self.reconciliation_box, "", p=p, bg=p["surface"],
                                            fg=p["fg"], size=9,
                                            wrap=self.sidebar_width - 70)
        self.reconciliation_body.pack(fill="x")
        controls = tk.Frame(self.panel, bg=p["bg"], pady=7)
        controls.pack(fill="x")
        self.checked_var = tk.BooleanVar(value=False)
        self.excluded_var = tk.BooleanVar(value=False)
        options = dict(bg=p["bg"], fg=p["fg"], activebackground=p["bg"], activeforeground=p["fg"],
                       selectcolor=p["surface"], font=("Segoe UI", 10), anchor="w", bd=0,
                       highlightthickness=1, highlightbackground=p["bg"], highlightcolor=p["blue"])
        # Kept hidden for legacy panel adapters; it has no user action.
        self.approve = tk.Checkbutton(controls, text="", variable=self.checked_var,
                                      state='disabled', **options)
        self.approve.bind('<Configure>',lambda e:self.approve.configure(wraplength=max(120,e.width-36)))
        self.exclude = tk.Checkbutton(controls, text="Nie można ocenić całego obrazu", variable=self.excluded_var,
                                      command=self.toggle_excluded, **options)
        self.exclude.pack(fill="x", pady=(2, 0))
        self.exclude_hint = ui_label(controls,
                 "Ta opcja wyłącza z oceny cały obraz, nie jedno wskazanie telefonu.",
                 p=p, bg=p["bg"], fg=p["muted"], size=9,
                 wrap=self.sidebar_width - 48)
        self.exclude_hint.pack(fill="x", pady=(1, 3))
        self.reason_host = tk.Frame(controls, bg=p["bg"])
        ui_label(self.reason_host, "Przyczyna (opcjonalnie)", p=p, bg=p["bg"], size=9, fg=p["muted"]).pack(fill="x")
        self.reason_var = tk.StringVar(value="")
        self.reason_entry = tk.Entry(self.reason_host, textvariable=self.reason_var, font=("Segoe UI", 9),
                                    bg=p["surface"], fg=p["fg"], insertbackground=p["fg"], width=22)
        self.reason_entry.pack(fill="x", pady=(2, 4))
        self.reason_entry.bind("<FocusOut>", self.save_exclusion_reason)
        self.reason_entry.bind("<Return>", self.save_exclusion_reason)
        self.reason_entry.bindtags((str(self.reason_entry), self.reason_entry.winfo_class(), str(self.h.window)))
        self.reason_gid = ""
        counts = tk.Frame(self.assessment.technical.body, bg=p["bg"])
        counts.pack(fill="x", pady=(0, 6))
        counts.columnconfigure(0, weight=1, uniform="count")
        counts.columnconfigure(1, weight=1, uniform="count")
        self.tally_cards = {}
        for i, (key, title, tone) in enumerate((("expected", "Tablice", "blue"), ("mt", "Detekcje MT", "blue"),
                    ("missed", "Pominięte tablice", "amber"), ("false_positive", "Błędne wykrycia", "red"))):
            card = MetricCard(counts, title, p=p, tone=tone, compact=True)
            card.grid(row=i//2, column=i%2, sticky="nsew", padx=2, pady=3)
            self.tally_cards[key] = card
        self.tally = ui_label(self.panel, "", p=p, bg=p["bg"], fg=p["muted"],
                              size=9, wrap=self.sidebar_width - 48)
        self.tally.pack(fill="x", pady=(0, 7))
        # Important operator action: separate from informational headings.
        # A two-line label + wraplength survives narrow laptop layouts/DPI scaling.
        self.nms_action = self.make_action_button(self.reconciliation_box,
            "Popraw wcześniejszą ocenę: 2 → 1 wynik",
            self.accept_nms, primary=True, wraplength=self.sidebar_width - 90)
        self.nms_action.configure(pady=11, padx=12, font=("Segoe UI", 10, "bold"))
        self.nms_action.pack_forget()
        self.reconciliation_box.pack_forget()
        selected = tk.Frame(self.panel, bg=p["surface"], padx=9, pady=8,
                            highlightbackground=p["border"], highlightthickness=1)
        selected.pack(fill="x")
        ui_label(selected, "ZAZNACZENIE", p=p, size=9, fg=p["muted"], bold=True).pack(fill="x")
        self.selection = ui_label(selected, "", p=p, bold=True, wrap=self.sidebar_width - 72)
        self.selection.pack(fill="x", pady=(5, 3))
        self.selection_hint = ui_label(selected, "", p=p, size=9, fg=p["muted"],
                                      wrap=self.sidebar_width - 72)
        self.selection_hint.pack(fill="x")
        self.crop = self.make_action_button(selected, "Podgląd cropu MZ  →",
                                            self.show_crop, compact=True)
        self.crop.pack(fill="x", pady=(7, 0))
        self.help_section = Disclosure(self.panel, "Skróty i obsługa", p=p)
        self.help_section.pack(fill="x", pady=(10, 6))
        for key, text in (("D", "Nowa tablica: 4 punkty"), ("Klik lub S", "Zaznacz ramkę; S również odznacza"),
                          ('Z / X', 'W dół / w górę stosu ramek jednej tablicy; bez zapisu oceny'),
                          ('Ctrl+klik badge MT', 'Przyjmij wybrany końcowy obrys jako referencję GT (ASSISTED)'),
                          ('Ctrl+klik badge GT', 'Powiąż aktywny kandydat z tą tablicą; GT edytujesz tylko raz'),
                          ("LPM", "Przeciągnij zaznaczony narożnik"),
                          ("R", "Usuń ręczną ramkę / cofnij korektę MT"),
                          ("T", "Pokaż lub ukryj granicę ROI (tylko widok)"),
                          ("N", "Przejdź do następnej tablicy"),
                          ("PPM", "Jedno kliknięcie na ramce: oznacz błąd lub cofnij"),
                          ("GT", "Kliknij dolny badge"), ("A · Q/E", "Następna tablica + dopasowanie · poprzedni/następny obraz"),
                          ("Rolka · F", "Zoom · dopasuj obraz"), ("Ctrl+Z/Y", "Cofnij / ponów zmianę"),
                          ("V", "Przełącz: cała klatka / wejście MT"),
                          ("Esc", "Anuluj bieżącą edycję")):
            ui_label(self.help_section.body, key, p=p, bold=True, size=9, fg=p["blue"]).pack(fill="x", pady=(4, 0))
            ui_label(self.help_section.body, text, p=p, size=9,
                     wrap=self.sidebar_width - 82).pack(fill="x")
        self.metrics = ui_label(self.panel,
            "Wyniki telefonu pozostają w historii. „Poza oceną” nie znaczy "
            "„błędna tablica”. Zmiany operatora są zapisywane osobno.",
                                 p=p, bg=p["bg"], size=9, fg=p["muted"],
                                 wrap=self.sidebar_width - 48)
        self.metrics.pack(fill="x", pady=(3, 9))

    def leave_canvas(self, event=None):
        self.pointer_canvas = None
        self._s_hover_hit = None
        self.hover = ""
        from .mobile_canvas_guides import clear
        clear(self)
        # Selection survives leaving the canvas; pointer motion never owns it.

    def remove_selected(self, event=None):
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.remove()
        from .mobile_canvas_geometry_layers import editable
        if hasattr(self,'navigator') and not editable(self):
            self.message('BBOX i Pose są źródłowymi wynikami MT. Wybierz GTR (Z/X), aby cofnąć lub usunąć obrys operatora.')
            return 'break'
        from .mobile_canvas_stacks import view,members
        v=view(self)
        candidate=next((c for c in (v or {}).get('candidates',[]) if c['object_id']==self.selected),None)
        if candidate and candidate['plate_id'] and len(members(v,candidate['plate_id']))>1:
            self.message('R nie usuwa wspólnej GT. Skoryguj jej narożniki albo cofnij konkretną zmianę Ctrl+Z.')
            return 'break'
        # Windows sends state=8 for normal hotkeys; only Ctrl inhibits R.
        if event is not None and getattr(event, "state", 0) & 4:
            return "break"
        if self.mode != "edit" or not self.selected:
            self.message("Najedź na ramkę, naciśnij S, a następnie R. Oryginał MT nie zostanie usunięty.")
            return "break"
        if not self.prepare_navigation() or not self.available():
            return "break"
        s = self._read()
        obj = next((o for o in s["objects"] if o["object_id"] == self.selected), None)
        if obj is None:
            return "break"
        oid, parent = obj["object_id"], obj["parent_id"]
        from .mobile_canvas_spatial_reference import selected_relation
        relation=selected_relation(self)
        if relation:
            stored=(v or {}).get('links',{}).get(relation[0],{}).get('plate_id')
            if stored!=relation[1]:
                reference=(v or {})['plates'][relation[1]]['origin_object_id']
                obj=next((o for o in s['objects'] if o['object_id']==reference),None)
                if obj is None:return 'break'
                oid,parent=obj['object_id'],obj['parent_id']
        description = ("Usunięto obrys GTR; MT pozostaje źródłem. Numer GT zachowano. D: narysuj, Ctrl+Z: przywróć" if parent else
                       "Usunięto ręczną ramkę z widoku i oceny; Ctrl+Z przywraca")
        if self._commit(lambda: core.remove_operator_geometry(self.h.session, self.h.current_invocation, oid), description):
            self.gt_focus_id='';self._reserved_edit=None;self._geometry_layer_state=None;self.mode='select'
            if not parent:
                self.selected = self.hover = ""
            self.sync(); self.refresh_tree(); self.request_overlay()
        return "break"

    def _sync_reconciliation_story(self, scene, tally, enabled):
        story = reconciliation_story(scene, tally)
        if not story:
            self.nms_action.pack_forget()
            self.reconciliation_box.pack_forget()
            return
        from .mobile_canvas_stacks import enhanced
        if enhanced(self) and not scene.get('_post_nms_pending'):
            self.reconciliation_box.pack_forget();self.nms_action.pack_forget()
            self.assessment.fields['correction_history'].configure(text=story['title']+'\n'+story['body'])
            self.assessment.fields['correction_history'].pack(fill='x')
            return
        self.reconciliation_title.configure(text=story["title"])
        self.reconciliation_body.configure(text=story["body"])
        # Top of the sidebar: context, reason, consequences, action — in order.
        self.reconciliation_box.pack(fill="x", pady=(8, 6), after=self.status)
        if story["action"]:
            self.nms_action.configure(text=story["action"], state="normal" if enabled else "disabled")
            self.nms_action.pack(fill="x", pady=(9, 0))
        else:
            self.nms_action.pack_forget()

    def _confirm_reconciliation(self):
        """Explain the decision with explicit cancel/confirm actions, default cancel.

        Nothing is saved until the positive button is pressed; this dialog
        does not decide how many physical plates are present on the image.
        """
        p = self.p
        parent = self.h.window
        dialog = tk.Toplevel(parent)
        dialog.title("Dlaczego trzeba poprawić wcześniejszą ocenę?")
        dialog.transient(parent)
        dialog.configure(bg=p["bg"])
        dialog.resizable(True, True)
        content = tk.Frame(dialog, bg=p["bg"], padx=19, pady=16)
        content.pack(fill="both", expand=True)
        dialog_width = min(620, max(430, dialog.winfo_screenwidth()-100))
        wrap = dialog_width - 70
        ui_label(content, "Poprawa wcześniejszej oceny — decyzja operatora",
                 p=p, bg=p["bg"], fg=p["fg"], bold=True, size=12,
                 wrap=wrap).pack(fill="x", pady=(0, 11))
        steps = (
            ("1. Co wykonał telefon?",
             "Na jednej klatce zapisał dwa wstępne wskazania, a po filtrowaniu "
             "nakładających się ramek użył tylko jednego do rozpoznawania."),
            ("2. Dlaczego program prosi o Twoją decyzję?",
             "Stara ocena na komputerze potraktowała obydwa wskazania jak "
             "oddzielne tablice. Ponieważ zawiera wcześniejsze poprawki "
             "człowieka, nie wolno nam zmienić tej oceny bez Twojej zgody."),
            ("3. Co zmieni Twoje potwierdzenie?",
             "Do oceny wyników telefonu pozostanie jedno wskazanie. Drugie oraz "
             "jego wcześniejsze GT, poprawki i zdarzenia będą nadal w historii. "
             "Nie zostanie uznane za fałszywy alarm. Status sprawdzenia obrazu "
             "zostanie cofnięty do ponownej kontroli."),
            ("4. Czego potwierdzenie NIE rozstrzyga?",
             "Nie potwierdza, że na zdjęciu jest tylko jedna prawdziwa tablica. "
             "Po zmianie obejrzyj całe ROI; jeśli widzisz dodatkową tablicę, "
             "oznacz ją ręcznie (D). Sprawdź ramki i GT, a dopiero potem "
             "zaznacz sprawdzenie obrazu."),
        )
        for heading, body in steps:
            ui_label(content, heading, p=p, bg=p["bg"], fg=p["blue"],
                     bold=True, size=10, wrap=wrap).pack(fill="x", pady=(4, 2))
            ui_label(content, body, p=p, bg=p["bg"], fg=p["fg"],
                     size=10, wrap=wrap).pack(fill="x", pady=(0, 7))
        ui_label(content, "Czy poprawić zapis wcześniejszej oceny w opisany sposób?",
                 p=p, bg=p["bg"], fg=p["fg"], bold=True,
                 wrap=wrap).pack(fill="x", pady=(8, 11))
        buttons = tk.Frame(content, bg=p["bg"])
        buttons.pack(fill="x")
        choice = [False]
        def finish(value):
            choice[0] = bool(value)
            dialog.destroy()
        cancel = self.make_action_button(buttons, "Nie — wróć bez zmian", lambda: finish(False))
        cancel.pack(side="left", fill="x", expand=True, padx=(0, 6))
        accept = self.make_action_button(buttons, "Tak — popraw wcześniejszą ocenę",
                                         lambda: finish(True), primary=True)
        accept.pack(side="left", fill="x", expand=True, padx=(6, 0))
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
        dialog.bind("<Escape>", lambda event: finish(False))
        dialog.update_idletasks()
        width = max(dialog_width, dialog.winfo_reqwidth())
        height = min(dialog.winfo_reqheight(), max(520, dialog.winfo_screenheight()-70))
        x = parent.winfo_rootx() + max(0, (parent.winfo_width()-width)//2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height()-height)//2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        dialog.wait_visibility()
        dialog.grab_set()
        cancel.focus_set()  # Default action is always to leave the review unchanged.
        parent.wait_window(dialog)
        return choice[0]

    def accept_nms(self):
        if not self.prepare_navigation() or not self.available():
            return
        s = self._read()
        if not s or not s.get("_post_nms_pending"):
            return
        if len(s['parents']) == 2 and not self._confirm_reconciliation():
            return
        self._commit(lambda: core.accept_post_nms_raw_only(self.h.session, self.h.current_invocation),
            "Poprawiono wcześniejszą ocenę. Drugie wstępne wskazanie zachowano "
            "w historii. Teraz sprawdź wszystkie prawdziwe tablice, ramki i GT "
            "przed zaznaczeniem obrazu jako sprawdzonego.")

    def toggle_checked(self):
        checked = bool(self.checked_var.get())
        if not self.prepare_navigation() or not self.available():
            self.sync(); return
        reason = self._read().get("exclusion_reason", "") if checked and self._read() else ""
        self._commit(lambda: core.set_scene_review_state(self.h.session, self.h.current_invocation,
                      checked=checked, exclusion_reason=reason),
                     "Zapisano stan sprawdzenia obrazu" if checked else "Cofnięto sprawdzenie obrazu")
        self.sync()

    def toggle_excluded(self):
        excluded = bool(self.excluded_var.get())
        if not self.prepare_navigation() or not self.available():
            self.sync(); return
        reason = (self.reason_var.get().strip() or "Operator oznaczył obraz jako niemożliwy do oceny") if excluded else ""
        self._commit(lambda: core.set_scene_review_state(self.h.session, self.h.current_invocation,
                      checked=True, exclusion_reason=reason),
                     "Obraz sprawdzony, ale nie do oceny" if excluded else "Przywrócono ocenialność; obraz nadal sprawdzony")
        self.sync()

    def _flush_exclusion_reason(self):
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.pending_action is None
        if self.reason_gid != self.h.current_invocation or not self.excluded_var.get():
            return True
        scene = self._read()
        if not scene or not scene.get("exclusion_reason"):
            return True
        reason = self.reason_var.get().strip() or "Operator oznaczył obraz jako niemożliwy do oceny"
        if scene["exclusion_reason"] == reason:
            return True
        return bool(self._commit(lambda: core.set_scene_review_state(
            self.h.session, self.h.current_invocation, checked=True, exclusion_reason=reason),
            "Zapisano przyczynę wyłączenia obrazu"))

    def save_exclusion_reason(self, event=None):
        if event is not None and getattr(self.h,'_export_has_focus',lambda:False)():
            self.prepare_export()
            return 'break'
        saved = self._flush_exclusion_reason()
        if saved and event is not None and getattr(event, "keysym", "") == "Return":
            self.c.focus_set()
        return "break"

    def _configure_tree(self):
        tree = self.h.subject_tree
        style = ttk.Style(self.h.window)
        style.configure("MobileReview.Observations.Treeview", background=self.p["surface"],
                        fieldbackground=self.p["surface"], foreground=self.p["fg"],
                        rowheight=35, font=("Segoe UI", 10, "bold"))
        # A strong selection remains visible after keyboard focus moves to Canvas.
        style.map("MobileReview.Observations.Treeview",
                  background=[("selected", "!focus", "#135481"),
                              ("selected", "#135481")],
                  foreground=[("selected", "!focus", "#FFFFFF"),
                              ("selected", "#FFFFFF")])
        tree.configure(style="MobileReview.Observations.Treeview", selectmode="browse")
        # One image = one task. Do not expose model candidates, raw-NMS entries,
        # duplicated source rows or the raw matrix of status abbreviations.
        # The underlying Treeview columns/values are retained for compatibility
        # with the report presenter, but none are displayed in this operator list.
        tree.configure(show="tree headings", displaycolumns=())
        tree.heading("#0", text="Obraz  ·  Co zrobić", anchor="w")
        tree.column("#0", width=self.left_width - 38, minwidth=200, stretch=True)
        tree.tag_configure("canvas_error", background=self.p["red_bg"], foreground=self.p["red"])
        tree.tag_configure("canvas_pending", background=self.p["surface"], foreground=self.p["fg"])
        tree.tag_configure("canvas_ready", background=self.p["green_bg"], foreground=self.p["green"])
        tree.tag_configure("canvas_excluded", background=self.p["purple_bg"], foreground=self.p["purple"])
        # Fallback for Windows ttk themes dimming selections on inactive focus.
        tree.tag_configure("canvas_active", background="#135481", foreground="#FFFFFF")

    @staticmethod
    def _image_action(rows):
        """Human task, derived only from existing persisted review states."""
        if rows and all('review_tasks' in row for row in rows):
            from .mobile_canvas_review_tasks import image_action
            return image_action(rows)
        if not rows:
            return "Poza oceną", "canvas_excluded"
        if all(row["status"] == "EXCLUDED" for row in rows):
            return "Poza oceną", "canvas_excluded"
        if all(row["status"] == "READY" for row in rows):
            return "Zweryfikowany", "canvas_ready"
        if all(row.get('progress', {}).get('complete') or row['status'] == 'EXCLUDED' for row in rows):
            if any(row.get('approval_issues',row.get('issues',[])) for row in rows):
                return 'Uzupełnij dane', 'canvas_pending'
            return "Zweryfikowany", "canvas_ready"
        if any(row.get("scene") and row["scene"].get("_post_nms_pending") for row in rows):
            return "Wyjaśnij rozbieżność", "canvas_error"
        from ..ranking.mobile_canvas_presentation import review_progress
        progress = [review_progress(row['scene']) for row in rows if row.get('scene')]
        decisions = sum(p['decisions'] for p in progress)
        if decisions:
            return f'Decyzje: {decisions}', 'canvas_pending'
        if any(p['unresolved'] for p in progress):
            return 'Nie można ustalić', 'canvas_pending'
        if any(row.get("scene") and row["scene"].get("checked")
               and row["cells"][-1] != "✓" for row in rows):
            return "Uzupełnij dane", "canvas_pending"
        if any(row.get("scene") is None for row in rows):
            return "Brak pełnego obrazu", "canvas_excluded"
        return "Do sprawdzenia", "canvas_pending"

    def inventory(self):
        key = (id(self.h.session), getattr(getattr(self.h.session, "review", None), "review_revision", -1))
        if key != self.inventory_key:
            from .mobile_canvas_review_tasks import inventory
            self.inventory_data = inventory(self.h.session,build_inventory(self.h.session),cache=self.task_inventory_cache)
            self.inventory_key = key
        return self.inventory_data

    def refresh_tree(self):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:return a3.render_tree()
        if getattr(self, 'reference_source', None) is not None:
            return self.reference_source.refresh_list()
        h = self.h
        if not h.session or not hasattr(h, "_subject_ids"):
            return
        inv = self.inventory()
        from .mobile_canvas_observation_blocks import render
        if render(self, inv):
            return
        tree = h.subject_tree
        # Only a proven shared executed invocation permits hiding an alias.
        # The source session, review objects, GT and evaluation counts stay intact.
        linked = linked_measurement_rows(h.session, h._subject_ids, inv["groups"])
        previously_selected = tree.selection()
        desired = set()
        display_index = 0
        self.tree_refreshing = True
        try:
            for iid, key in h._subject_ids.items():
                if iid in linked:
                    continue
                display_index += 1
                gids = [gid for gid in subject_groups(h.session, key)
                        if gid in inv["groups"]]
                rows = [inv["groups"][gid] for gid in gids]
                cells = core.merge_review_cells(row["cells"] for row in rows)
                action, tag = self._image_action(rows)
                if not rows or all(row.get('scene') is None for row in rows):
                    from ..ranking.mobile_mt_history import history_evidence
                    histories=[history_evidence(h.session,gid) for gid in subject_groups(h.session,key)]
                    histories=[item for item in histories if item]
                    if len(histories)==1:
                        action=histories[0]['list_label']
                label = f"Obraz {display_index:02d}  ·  {action}"
                values = (f"{display_index:02d}",) + cells
                desired.add(iid)
                if tree.exists(iid):
                    if tree.parent(iid):
                        tree.move(iid, "", "end")
                    tree.item(iid, text=label, values=values, tags=(tag,), open=False)
                else:
                    tree.insert("", "end", iid=iid, text=label,
                                values=values, tags=(tag,), open=False)
                # Remove stale descendants from AKT66 or older trees. A
                # descendant represents a raw candidate, not another task.
                for cid in tuple(tree.get_children(iid)):
                    tree.delete(cid)
            from .mobile_canvas_scene_frames import add_rows
            add_rows(self, desired)
            for iid in tuple(tree.get_children("")):
                if iid not in desired:
                    tree.delete(iid)
            # There are deliberately no object/group targets in the image list.
            # Object editing continues via Canvas S / A and the right inspector.
            self.tree_targets = {}
            self.linked_tree_rows = linked
        finally:
            self.tree_refreshing = False
        h._schedule_status_pills()
        self.update_header(inv)
        if previously_selected:
            old = previously_selected[0]
            new = linked.get(old, (old, None))[0]
            if tree.exists(new) and tree.selection() != (new,):
                tree.selection_set(new)
        self.sync_tree_selection()

    def update_header(self, inv=None):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:return a3.update_header()
        from ..ranking.mobile_canvas_presentation import progress_caption
        inv = inv or self.inventory()
        extra = f"  •  {inv['excluded']} poza oceną" if inv['excluded'] else ""
        if inv['unavailable']:
            extra += f"  •  tylko podgląd: {inv['unavailable']}"
        self.h.progress_label.configure(text=progress_caption(inv)+extra)
        if inv.get('operator_review_progress') or inv.get('experiment_family') == 'F5A':
            from ..ranking.mobile_review_status import approval_caption
            review = self.h.session.review
            self.h.progress_label.configure(text=approval_caption(inv, review.review_status, review.review_revision))
        elif 'task_decisions' in inv:
            self.h.progress_label.configure(text=f"Sprawdzone: {inv.get('approval_ready',inv['inspected'])}/{inv['total']} · Decyzje tablic: {inv['task_decisions']} · Osobne kroki: {inv['task_steps']}"+extra)
        if hasattr(self.h, "form_status_label"):
            from ..ranking.mobile_verification_policy import verification_issues
            session = self.h.session
            review=getattr(session,'review',None)
            key=(id(session),id(review),getattr(review,'review_revision',None))
            if getattr(self,'_header_validation_key',None)!=key:
                self._header_complete=bool(session and not verification_issues(session))
                self._header_validation_key=key
            # Display-only cache; the report approval command revalidates.
            complete=self._header_complete
            if getattr(session,'_history_inspection_only',False):
                status, tone = "TYLKO HISTORIA", "amber"
                self.h.progress_label.configure(text='Historia · brak obrazów dopuszczonych do oceny')
            elif complete and session.review.review_status == "COMPLETED":
                status, tone = "WERYFIKACJA ZATWIERDZONA", "green"
            elif complete:
                status, tone = "GOTOWA DO ZATWIERDZENIA", "amber"
            else:
                status, tone = "W TOKU", "blue"
            if inv.get('operator_review_progress', {}).get('operator_state') == 'NOT_STARTED':
                status, tone = "Weryfikacja operatora: Nie rozpoczęto", "blue"
            elif inv.get('operator_review_progress', {}).get('operator_state') == 'UNVERIFIED':
                status, tone = "Nie można potwierdzić zapisu operatora", "amber"
            self.h.form_status_label.configure(text=status, fg=self.p[tone],
                                               font=("Segoe UI", 9, "bold"))

    def clear_tree_selection(self):
        """Explicit S-off/Escape only; never deletes rows or changes review data."""
        tree = self.h.subject_tree
        current = tree.selection()
        if current:
            tree.selection_remove(*current)
        tree.focus("")

    def sync_tree_selection(self):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:return a3.sync_selection()
        if getattr(self, 'reference_source', None) is not None:
            return
        """Highlight the current image, never an internal MT/GT record.

        A programmatic highlight of the current image must not undo an active
        S-selected plate. select_tree_row handles that same-image event as noop.
        """
        if self.tree_refreshing or not hasattr(self.h, "_subject_ids"):
            return
        from .mobile_canvas_observation_blocks import sync_selection
        if sync_selection(self):
            return
        if self.h.current_invocation in getattr(self.h.session, 'scene_frames', {}):
            iid=next((i for i,(key,gid) in getattr(self,'scene_mt_tree_ids',{}).items() if gid==getattr(self,'scene_mt_focus',None)),None)
            iid=iid or next((i for i,key in getattr(self,'scene_tree_ids',{}).items() if key==self.h.current_invocation),None)
            if iid and self.h.subject_tree.exists(iid):
                if self.h.subject_tree.selection()!=(iid,):self.h.subject_tree.selection_set(iid)
                self.h.subject_tree.focus(iid)
            return
        tree = self.h.subject_tree
        key = self.h.subject_key
        roots = set(tree.get_children(""))
        iid = next((i for i, subject in self.h._subject_ids.items()
                    if subject == key and i in roots), None)
        if iid is None:
            alias = next((i for i, subject in self.h._subject_ids.items()
                          if subject == key), None)
            if alias in self.linked_tree_rows:
                iid = self.linked_tree_rows[alias][0]
        if iid is None and self.h.current_invocation:
            iid = next((i for i, subject in self.h._subject_ids.items()
                        if i in roots and self.h.current_invocation in
                        subject_groups(self.h.session, subject)), None)
        previous_active = getattr(self, "_active_tree_iid", "")
        if previous_active != iid and previous_active and tree.exists(previous_active):
            oldtags=tuple(t for t in tree.item(previous_active, "tags") if t != "canvas_active")
            tree.item(previous_active, tags=oldtags)
        if iid and tree.exists(iid):
            tags=tuple(t for t in tree.item(iid,"tags") if t != "canvas_active")
            tree.item(iid, tags=tags+("canvas_active",))
            self._active_tree_iid = iid
            self.tree_branch = iid
            if tree.selection() != (iid,):
                tree.selection_set(iid)
            tree.focus(iid)
            tree.see(iid)

    def select_tree_row(self, iid):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:return a3.select(iid)
        if self.tree_refreshing:
            return True
        from .mobile_canvas_observation_blocks import select
        if select(self, iid):
            return True
        if iid in getattr(self, 'scene_mt_tree_ids', {}):
            from .mobile_canvas_scene_frames import open_scene
            key, self.scene_mt_focus = self.scene_mt_tree_ids[iid]
            return open_scene(self, key)
        if iid in getattr(self, 'scene_tree_ids', {}):
            from .mobile_canvas_scene_frames import open_scene
            self.scene_mt_focus = None
            return open_scene(self, self.scene_tree_ids[iid])
        if iid in self.linked_tree_rows:
            # A legacy asynchronous selection event may reference an old row.
            # Never navigate to a hidden source alias or create a new GT task.
            return True
        if iid not in self.h._subject_ids:
            return True
        self.tree_branch = iid
        key = self.h._subject_ids[iid]
        if key == self.h.subject_key:
            # Tk fires selection events for programmatic highlights too. Do
            # not erase the currently edited plate/GT on such an event.
            return True
        if not self.prepare_navigation():
            self.sync_tree_selection()
            return True
        # The host now opens the selected image. The raw source IDs are still
        # available in h._subject_ids; no session content is mutated here.
        self.selected = self.hover = ""
        self.mode = "select"
        self.plate_fit_saved = None
        return False

    def open_result_image(self,gid):
        from .mobile_canvas_observation_blocks import open_gid
        grouped = open_gid(self, gid)
        if grouped is not None:
            return grouped
        if not self.prepare_navigation():return False
        tree=self.h.subject_tree
        iid=next((i for i in tree.get_children('') if gid in subject_groups(self.h.session,self.h._subject_ids[i])),None)
        if iid is None:return False
        self.review_tabs.select(0);tree.selection_set(iid);tree.focus(iid);tree.see(iid);self.h._select_subject()
        if gid!=self.h.current_invocation:
            group=self.h.session.mt_invocations.get(gid)
            aid=str(group.records[0].get('id') or '') if group and group.records else ''
            if aid:self.pending_tree_target=(gid,'');self.h._multi_focus_attempt(aid)
        return True

    def next_pending(self):
        a3=getattr(self.h,'_f5a_a3',None)
        if a3 and a3.active:return a3.next_pending()
        from .mobile_canvas_observation_blocks import next_pending
        if next_pending(self):
            return
        from ..ranking.mobile_scene_frame_review import scene as full_scene
        for iid, key in getattr(self, 'scene_tree_ids', {}).items():
            row = self.h.session.scene_frames[key]
            if row['write_status'] == 'WRITTEN' and not full_scene(self.h.session, key)['checked']:
                self.h.subject_tree.selection_set(iid); self.h.subject_tree.focus(iid); self.h.subject_tree.see(iid)
                self.select_tree_row(iid)
                return
        from ..ranking.mobile_mt_object_evaluation import evaluate_unit
        for iid,(_,gid) in getattr(self,'scene_mt_tree_ids',{}).items():
            group=self.h.session.mt_invocations[gid]
            if group.cancelled or group.execution_failed:continue
            from ..ranking.mobile_scene_frame_review import mt_review_issues
            if mt_review_issues(self.h.session,gid)==[]:continue
            if not evaluate_unit(self.h.session,gid)['n_qualified']:
                self.h.subject_tree.selection_set(iid);self.h.subject_tree.focus(iid);self.h.subject_tree.see(iid)
                self.select_tree_row(iid);return
        inv = self.inventory()
        tree = self.h.subject_tree
        roots = tuple(tree.get_children(""))
        for gid, item in inv["groups"].items():
            if 'review_tasks' in item and not item['review_tasks']['needs_review']:
                continue
            if item["status"] in ("READY", "EXCLUDED") or ('approval_issues' in item and not item['approval_issues']):
                continue
            iid = next((iid for iid in roots
                        if gid in subject_groups(self.h.session, self.h._subject_ids[iid])), None)
            if iid is None:
                continue
            tree.selection_set(iid)
            tree.focus(iid)
            tree.see(iid)
            self.h._select_subject()
            if gid != self.h.current_invocation:
                group = self.h.session.mt_invocations.get(gid)
                records = group.records if group else []
                aid = str(records[0].get("id") or records[0].get("attempt_id") or "") if records else ""
                if aid:
                    self.pending_tree_target = (gid, "")
                    self.h._multi_focus_attempt(aid)
            return
        self.message("Wszystkie dostępne obrazy zostały sprawdzone lub wyłączone z oceny.")

    def route_scroll(self, event):
        """Handle a Windows wheel delivered to the focused image over a side panel."""
        try:
            x,y=self.h.window.winfo_pointerxy()
            if hasattr(self,'clean_bar'):
                panel=self.h.review_sidebar_scroll
                if (panel.winfo_ismapped() and panel.winfo_rootx()<=x<panel.winfo_rootx()+panel.winfo_width()
                        and panel.winfo_rooty()<=y<panel.winfo_rooty()+panel.winfo_height()):
                    return self.h._on_sidebar_wheel(event)
            areas=[(self.h.review_sidebar_scroll,self.h._on_sidebar_wheel),
                   (self.report_host.canvas,self.report_host.wheel)]
            if hasattr(self.h,"science_dashboard"):
                areas.append((self.h.science_dashboard.canvas,self.h.science_dashboard._wheel))
            for canvas,handler in areas:
                if (canvas.winfo_ismapped() and canvas.winfo_rootx()<=x<canvas.winfo_rootx()+canvas.winfo_width()
                        and canvas.winfo_rooty()<=y<canvas.winfo_rooty()+canvas.winfo_height()):
                    return handler(event)
        except (tk.TclError,AttributeError):
            pass
        return None
