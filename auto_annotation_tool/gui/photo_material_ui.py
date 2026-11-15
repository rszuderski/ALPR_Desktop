"""Shared material/GT actions for the existing browser, viewer and Canvas host."""
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from ..ranking import photo_material as model


def choose_material(parent, *, recursive=None):
    try:
        recent = model.recent_material()
    except (OSError, ValueError, KeyError):
        recent = None
    source = filedialog.askdirectory(parent=parent, title='Wybierz katalog fotografii',
        initialdir=str(recent.root if recent else Path.home()), mustexist=True)
    if not source:
        return None
    if recursive is None:
        recursive = messagebox.askyesnocancel('Podkatalogi',
            'Uwzględnić także fotografie w podkatalogach?\nTak — cały wybrany katalog. Nie — tylko jego pierwszy poziom.', parent=parent)
        if recursive is None:
            return None
    return model.DirectoryMaterial.select(source, recursive=recursive)


def open_material_editor(parent, material=None, *, on_close=None):
    from .z4_photo_reference import PhotoReferenceWindow
    try:
        material = material or choose_material(parent)
        if material is None:
            return None
        root = parent._root()
        windows = getattr(root, '_material_gt_windows', {})
        root._material_gt_windows = windows
        key = str(material.default_output.resolve())
        previous = windows.get(key)
        if previous is not None and not previous._closed and previous.window.winfo_exists():
            if on_close is not None and on_close not in previous._material_close_callbacks:
                previous._material_close_callbacks.append(on_close)
            previous.window.deiconify(); previous.window.lift()
            return previous
        reference = material.reference(create=True)
        editor = PhotoReferenceWindow(parent, reference.material, reference=reference)
        editor._material_close_callbacks = [on_close] if on_close is not None else []
        editor.window.title('Oznaczanie fotografii — ' + material.root.name)
        editor.header_title.configure(text='Oznaczanie fotografii · ' + material.root.name)
        assisted = material.annotation_context().get('predictions_exposed') or reference.predictions_exposed
        editor.status_var.set('Zaznacz tablice i wpisz numery. ' +
            ('Korekty po obejrzeniu wyników pozostają asystowane.' if assisted else 'Zmiany są zapisywane w sesji GT.'))
        windows[key] = editor
        def closed(event):
            if event.widget is not editor.window:
                return
            windows.pop(key, None)
            for callback in editor._material_close_callbacks:
                callback()
        editor.window.bind('<Destroy>', closed, add='+')
        return editor
    except (OSError, ValueError, KeyError, TypeError) as exc:
        messagebox.showerror('Nie można otworzyć oznaczania', str(exc), parent=parent)
        return None


def open_recent_editor(parent, *, on_close=None):
    try:
        material = model.recent_material()
        return open_material_editor(parent, material, on_close=on_close)
    except (OSError, ValueError, KeyError) as exc:
        messagebox.showerror('Nie można wznowić sesji GT', str(exc), parent=parent)
        return None


def import_material_reference(parent, material):
    from tkinter.scrolledtext import ScrolledText
    path = filedialog.askopenfilename(parent=parent, title='Wczytaj dane referencyjne dla wybranego materiału',
        filetypes=[('Dane referencyjne', '*.json')])
    if not path:
        return None
    try:
        report = model.reference_compatibility(material, path)
        accepted = {'value': False}
        dialog = tk.Toplevel(parent); dialog.title('Powiązanie danych referencyjnych')
        dialog.geometry('760x440'); dialog.minsize(540, 320)
        text = ScrolledText(dialog, wrap='word', padx=12, pady=10)
        text.pack(fill='both', expand=True)
        lines = [f'Materiał: {material.root.name}', f'Zgodne fotografie: {len(report["mapping"])}/{len(material.order)}',
            'Zgodność obejmuje SHA-256, orientację i geometrię obrazu.',
            'Oryginalny plik GT pozostanie niezmieniony. Poprzednia lokalna wersja zostanie zachowana.', '']
        labels={'NEW_OR_AMBIGUOUS_IMAGE':'Nowa fotografia lub niejednoznaczne powiązanie',
            'IMAGE_IDENTITY_OR_GEOMETRY_CONFLICT':'Inna zawartość lub geometria',
            'REFERENCE_IMAGE_NOT_MATCHED':'Fotografia z GT bez zgodnego odpowiednika'}
        fields={'source_file_sha256':'SHA-256', 'width':'szerokość', 'height':'wysokość',
            'exif_orientation':'orientacja EXIF', 'coordinate_space':'układ współrzędnych'}
        lines += [f'{labels[row["kind"]]}: {row["path"]}' +
            (' · ' + ', '.join(fields.get(f,f) for f in row['fields']) if row.get('fields') else '') for row in report['differences']]
        if not report['differences']:
            lines.append('Wszystkie fotografie są zgodne. Można podłączyć także nieukończone oznaczenia.')
        elif report['mapping']:
            lines.append('\nPo potwierdzeniu zostaną wczytane wyłącznie zgodne fotografie. Pozostałe będą bez oznaczeń.')
        text.insert('1.0', '\n'.join(lines)); text.configure(state='disabled')
        actions = tk.Frame(dialog); actions.pack(fill='x', padx=12, pady=10)
        def confirm():
            accepted['value'] = True; dialog.destroy()
        tk.Button(actions, text='Podłącz zgodne fotografie' if report['differences'] else 'Podłącz GT',
            command=confirm, state='normal' if report['mapping'] else 'disabled').pack(side='left')
        tk.Button(actions, text='Anuluj', command=dialog.destroy).pack(side='right')
        dialog.transient(parent); dialog.grab_set(); parent.wait_window(dialog)
        if not accepted['value']:
            return None
        return model.import_reference(material, path, accept_partial=bool(report['differences']), expected_sha=report['source_sha256'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        messagebox.showerror('Nie podłączono danych referencyjnych', str(exc), parent=parent)
        return None


def export_material_reference(parent, material):
    path = filedialog.asksaveasfilename(parent=parent, title='Eksport danych referencyjnych',
        initialfile=material.root.name+'_GT.json', defaultextension='.json', filetypes=[('Dane referencyjne', '*.json')])
    if not path:
        return
    try:
        model.export_reference(material, path)
    except (OSError, ValueError, KeyError) as exc:
        messagebox.showerror('Nie wyeksportowano GT', str(exc), parent=parent)


class MaterialPresentationActions:
    """Adds material actions to FinalApp; renderer/run/scoring stay in their owners."""
    def material_actions_available(self):
        return (not self.closed and self.mode not in ('PREPARING','COUNTDOWN','RUNNING','TRANSITION')
            and not self._post_roll_hold and not self._selection_loading)

    def sync_presentation_ui(self, *, geometry_ok=True, selection_ok=True):
        if not hasattr(self,'material_preview') or self.closed:return
        idle=self.material_actions_available()
        material=self.source_material or self.material
        self.material_preview.set_material(material)
        self.material_preview.controls()
        self.preview_title.set(self.source_material.root.name if self.source_material is not None else
            'Zapisany pokaz' if self.material is not None else 'Twoje fotografie')
        self.advanced_btn.configure(state='normal' if idle else 'disabled')
        self.recursive_checkbox.configure(state='normal' if idle else 'disabled')
        self.monitor_combo.configure(state='readonly' if idle else 'disabled')
        enabled=idle and self.source_material is not None
        self.reference_menu_btn.configure(state='normal' if enabled else 'disabled')
        state=getattr(self,'_material_reference_state',{}).get('state','none')
        self.edit_reference_btn.configure(state='normal' if enabled else 'disabled',
            text='Utwórz oznaczenia' if state=='none' else 'Otwórz oznaczenia')
        if self.mode in ('PREPARING','COUNTDOWN','RUNNING','TRANSITION') or self._post_roll_hold:
            self.stop_btn.grid()
            if not self._post_roll_hold:self.stop_btn.configure(text='Zatrzymaj')
        else:self.stop_btn.grid_remove()
        if self.source_material is not None and self.prepared is None:
            self.calibration_btn.configure(state='normal' if idle and geometry_ok and selection_ok else 'disabled')

    def preview_calibration(self):
        if self.calibration_preview is not None:return self.toggle_calibration()
        if not self.material_actions_available():return
        if self.source_material is not None and self.prepared is None:
            return self.prepare_material(after_prepare=self.toggle_calibration)
        return self.toggle_calibration()

    def choose_photographs(self):
        if not self.material_actions_available():
            return
        source = filedialog.askdirectory(parent=self.window, title='Wybierz katalog fotografii', mustexist=True)
        if not source:
            return
        recursive = self.recursive_photographs.get()
        self._selection_generation += 1
        generation = self._selection_generation
        self._selection_loading = True
        self.refresh_plan()
        def failed(error):
            if self.closed or generation != self._selection_generation:
                return
            self._selection_loading = False
            self.status.set('Nie wczytano materiału: ' + str(error))
            self.refresh_plan()
        def selected(material):
            if not self.closed and generation == self._selection_generation:
                self.use_material(material)
        self.background(lambda:model.DirectoryMaterial.select(source, recursive=recursive), selected, failed)

    def use_material(self, material):
        self.close_calibration(); self._dispose_viewer()
        if self.cache:
            self.cache.close(); self.cache = None
        self._loading = True
        try:
            self.paths['positives'].set(str(material.root)); self.paths['xml'].set(''); self.paths['negatives'].set('')
            self.paths['output'].set(str(material.directory/'presentations'))
        finally:
            self._loading = False
        self.source_material = material
        self.mode='IDLE'
        self.recursive_photographs.set(material.data['recursive'])
        self.prepared = None; self.manifest = None; self.material = None; self.run = None
        self.analysis_reference = None; self.current_index = 0
        self._selection_loading = False; self._selection_error = ''
        self.audit_result = dict(no_reference=True, images=list(material.order))
        self.set_info.set(material.root.name)
        self.selected_folder.set(str(material.root))
        self.status.set('Wybierz Dane referencyjne lub rozpocznij pokaz. GT jest opcjonalne.')
        self.refresh_material_reference(); self.refresh_plan()

    def refresh_material_reference(self):
        if self.closed or not self.source_material:
            return
        status = self.source_material.reference_status()
        self._material_reference_state=status
        self.reference_info.set(f'Materiał: {status["total"]} fotografii | GT: {status["label"]} ({status["complete"]}/{status["total"]})')
        if self.prepared and status['state'] != 'invalid':
            from ..ranking import final180_reference as presentation
            self.analysis_reference = model.reference_for_presentation(self.source_material, self.prepared)
            self.gt_preflight = presentation.reference_preflight(self.prepared, reference_path=self.analysis_reference)
        if status['state'] == 'invalid':
            self.analysis_reference = None
            self.status.set('GT niezgodne: '+status.get('error','')+'. Pokaz fotografii nie zmienia oryginalnego GT.')
        self.sync_presentation_ui()

    def edit_material_reference(self):
        if not self.material_actions_available():
            return
        if self.source_material is None:
            messagebox.showinfo('Dane referencyjne', 'Wybierz katalog fotografii. Edycja referencji historycznego pokazu jest dostępna z jego wyników.', parent=self.window)
            return
        return open_material_editor(self.window, self.source_material, on_close=self.refresh_material_reference)

    def import_material_reference(self):
        if self.source_material is not None and self.material_actions_available():
            result = import_material_reference(self.window, self.source_material)
            if result:
                self.refresh_material_reference()

    def export_material_reference(self):
        if self.source_material is not None and self.material_actions_available():
            export_material_reference(self.window, self.source_material)

    def show_material_binding(self):
        if self.source_material is None:
            return
        status = self.source_material.reference_status()
        origin = ('GT nieutworzone' if status['state']=='none' else
            'oznaczenia asystowane' if status.get('predictions_exposed') else 'oznaczanie bez podglądu predykcji')
        messagebox.showinfo('Powiązanie i zgodność',
            f'Materiał: {self.source_material.root.name}\nFotografie: {len(self.source_material.order)}\n'
            f'GT: {status["label"]}\nTożsamość: SHA-256, ścieżka względna i geometria EXIF.\n'
            f'Źródło: {origin}\n'
            f'{status.get("error", "")}\nSesja GT: {self.source_material.default_output}', parent=self.window)

    def prepare_material(self, *, start_after=False, after_prepare=None):
        from ..ranking import final180_reference as presentation
        try:
            screen = self._select_monitor()
            seconds = presentation.parse_dwell(self.dwell.get())
        except (ValueError, TypeError) as exc:
            self.status.set(str(exc)); return
        material = self.source_material
        self._generation += 1
        generation = self._generation
        self.mode = 'PREPARING'; self.dwell_entry.configure(state='disabled');self.stop_btn.configure(state='normal');self.refresh_plan()
        def done(result):
            if self.closed or generation != self._generation:
                return
            self.mode = 'IDLE'
            self.dwell_entry.configure(state='normal'); self.stop_btn.configure(state='disabled')
            self.loaded(result[0])
            self.source_material = material; self.analysis_reference = result[1]
            from .final180_tv import remember_selection
            remember_selection(seconds, result[0])
            self.refresh_material_reference()
            if start_after:
                self.begin_countdown()
            elif after_prepare is not None:after_prepare()
        def failed(error):
            if generation != self._generation or self.closed:
                return
            self.mode = 'IDLE'; self.dwell_entry.configure(state='normal'); self.stop_btn.configure(state='disabled')
            self.status.set('Nie przygotowano pokazu: '+str(error)); self.refresh_plan()
        self.status.set('Przygotowuję pokaz z zapisanego materiału. GT pozostaje opcjonalne.')
        def work():
            try:
                result=model.prepare_presentation(material, screen=(screen['w'],screen['h']), dwell_seconds=seconds)
                self.queue.put((done,result,None))
            except Exception as exc:
                self.queue.put((failed,str(exc),None))
        # The general background helper intentionally rejects active modes.
        # PREPARING owns this worker, just like the historical prepare() path.
        self.workers.submit(work)
