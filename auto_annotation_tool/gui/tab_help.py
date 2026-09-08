#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zakładka Z5: instrukcja pracy, architektura i mapa systemu.

Z5 opisuje pojęcia, odpowiedzialności modułów i zasady pracy w kampanii,
eksporcie oraz eksperymentach badawczych.
"""

import tkinter as tk
from tkinter import ttk

from ..config import CONFIG
from .inertial_scroll import InertialScrollController
from .web_slim_scrollbar import WebSlimScrollbar


class HelpTab:
    """Zakładka instrukcji i architektury aplikacji."""

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.frame = ttk.Frame(parent)
        self._startup_ui_ready = False
        self._doc_widgets = []
        self._inertial_scroll = InertialScrollController(self.frame, decay=0.80, interval_ms=14)
        self._create_widgets()
        self.frame.after_idle(self._mark_startup_ui_ready)

    def _mark_startup_ui_ready(self):
        self._startup_ui_ready = True

    def is_startup_ui_ready(self) -> bool:
        return bool(getattr(self, "_startup_ui_ready", False))

    def _create_widgets(self):
        palette = getattr(self.app, "palette", {}) or {}

        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, padx=12, pady=(10, 6))
        ttk.Label(
            header,
            text="Z5. Instrukcje, architektura i metodologia",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor=tk.W)
        ttk.Label(
            header,
            text=(
                f"{CONFIG.APP_NAME} ver. {CONFIG.VERSION}: kampania, zasoby, PZ2/PZ3/PZ4, "
                "ranking, eksport mobilny i zasady testów."
            ),
        ).pack(anchor=tk.W, pady=(4, 0))

        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.t1 = self._create_text_page(
            "[PZ1] Jak pracować",
            palette=palette,
            filler=self._fill_workflow,
        )
        self.t2 = self._create_text_page(
            "[PZ2] Kampania",
            palette=palette,
            filler=self._fill_campaign,
        )
        self.t3 = self._create_text_page(
            "[PZ3] Dane i kod",
            palette=palette,
            filler=self._fill_architecture,
        )
        self.t4 = self._create_text_page(
            "[PZ4] Eksport i badania",
            palette=palette,
            filler=self._fill_export_research,
        )

    def _create_text_page(self, title, palette, filler):
        page = ttk.Frame(self.notebook)
        self.notebook.add(page, text=title)

        host = ttk.Frame(page)
        host.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(
            host,
            wrap=tk.WORD,
            bg=palette.get("doc_bg", palette.get("panel", "#1f1f1f")),
            fg=palette.get("doc_fg", palette.get("fg", "#f3f3f3")),
            insertbackground=palette.get("doc_fg", palette.get("fg", "#f3f3f3")),
            selectbackground=palette.get("accent", palette.get("primary", "#4fc1ff")),
            selectforeground="#ffffff",
            font=("Segoe UI", 10),
            padx=20,
            pady=20,
        )
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = WebSlimScrollbar(
            host,
            orient=tk.VERTICAL,
            command=text.yview,
            auto_hide=False,
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.configure(yscrollcommand=scrollbar.set)
        text.web_vbar = scrollbar
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            text.bind(sequence, self._on_doc_text_mousewheel, add="+")

        self._configure_tags(text, palette)
        try:
            self.app.style_text_widget(text, role="doc")
        except Exception:
            pass
        text.insert(tk.END, f"{title}\n", "H1")
        filler(text)
        text.configure(state=tk.DISABLED)
        self._doc_widgets.append(text)
        return text

    def _configure_tags(self, widget, palette):
        widget.tag_configure(
            "H1",
            font=("Segoe UI", 18, "bold"),
            foreground=palette.get("accent", palette.get("primary", "#4fc1ff")),
            spacing3=12,
        )
        widget.tag_configure(
            "H2",
            font=("Segoe UI", 13, "bold"),
            foreground=palette.get("fg", palette.get("text", "#f3f3f3")),
            spacing1=14,
            spacing3=6,
        )
        widget.tag_configure(
            "H3",
            font=("Segoe UI", 11, "bold"),
            foreground=palette.get("accent", palette.get("primary", "#4fc1ff")),
            spacing1=10,
            spacing3=4,
        )
        widget.tag_configure("P", font=("Segoe UI", 10), spacing1=2, spacing3=8)
        widget.tag_configure("B", font=("Segoe UI", 10, "bold"))
        widget.tag_configure(
            "CODE",
            font=("Consolas", 9),
            background=palette.get("code_bg", palette.get("panel_alt", "#252526")),
            foreground=palette.get("code_fg", palette.get("fg", "#dcdcaa")),
        )
        widget.tag_configure(
            "CALLOUT",
            font=("Segoe UI", 10),
            background=palette.get("panel_alt", palette.get("doc_bg", "#252526")),
            spacing1=8,
            spacing3=8,
            lmargin1=14,
            lmargin2=14,
            rmargin=14,
        )
        widget.tag_configure(
            "OK",
            foreground=palette.get("success", "#16833a"),
            font=("Segoe UI", 10, "bold"),
        )
        widget.tag_configure(
            "WARN",
            foreground=palette.get("warning", "#a16207"),
            font=("Segoe UI", 10, "bold"),
        )
        widget.tag_configure(
            "BAD",
            foreground=palette.get("danger", "#b42318"),
            font=("Segoe UI", 10, "bold"),
        )

    def apply_theme(self):
        try:
            palette = getattr(self.app, "palette", {}) or {}
            self.app.style_panel_surface(self.frame, background=palette.get("panel", "#252526"))
        except Exception:
            palette = {}

        for widget in list(getattr(self, "_doc_widgets", []) or []):
            try:
                self.app.style_text_widget(widget, role="doc")
            except Exception:
                pass
            try:
                self._configure_tags(widget, palette)
            except Exception:
                pass

    def _paragraph(self, widget, text):
        widget.insert(tk.END, f"{text}\n\n", "P")

    def _section(self, widget, title):
        widget.insert(tk.END, f"{title}\n", "H2")

    def _subsection(self, widget, title):
        widget.insert(tk.END, f"{title}\n", "H3")

    def _bullet_list(self, widget, items):
        for item in items:
            widget.insert(tk.END, f"• {item}\n", "P")
        widget.insert(tk.END, "\n", "P")

    def _numbered_list(self, widget, items):
        for idx, item in enumerate(items, start=1):
            widget.insert(tk.END, f"{idx}. {item}\n", "P")
        widget.insert(tk.END, "\n", "P")

    def _code_block(self, widget, text):
        widget.insert(tk.END, f"{text.rstrip()}\n\n", "CODE")

    def _callout(self, widget, text):
        widget.insert(tk.END, f"{text}\n\n", "CALLOUT")

    @staticmethod
    def _text_widget_can_scroll(widget, units: int) -> bool:
        if widget is None or units == 0:
            return False
        try:
            first, last = widget.yview()
            if units < 0 and float(first) <= 0.0:
                return False
            if units > 0 and float(last) >= 1.0:
                return False
            return True
        except Exception:
            return False

    def _on_doc_text_mousewheel(self, event=None):
        widget = getattr(event, "widget", None)
        units = self._inertial_scroll.mousewheel_units(event)
        if units == 0 or not self._text_widget_can_scroll(widget, units):
            return None
        self._inertial_scroll.queue_canvas_by_units(
            widget,
            units,
            magnitude=self._inertial_scroll.mousewheel_magnitude(event),
        )
        return "break"

    def _fill_workflow(self, widget):
        self._paragraph(
            widget,
            "Ten ekran opisuje model pracy aplikacji: zakładki, bramki, zasoby, artefakty i zasady "
            "podejmowania decyzji w kampanii.",
        )

        self._section(widget, "Najważniejsza zasada")
        self._paragraph(
            widget,
            "Graf kampanii prowadzi użytkownika przez decyzje, a karty robocze wykonują pracę. "
            "Stan projektu zapisuje CampaignManager. Bramka nie powinna być uznawana za gotową "
            "na podstawie samego koloru w UI, tylko na podstawie kontraktu zasobów i zatwierdzonej pracy.",
        )
        self._callout(
            widget,
            "W praktyce: Z1 pokazuje mapę i bramki, Z2 kontroluje tablice na zdjęciach, "
            "Z3/PZ2 kontroluje znaki na cropach tablic, Z3/PZ3 buduje dataset znaków, "
            "Z4 trenuje, rankinguje i eksportuje modele.",
        )

        self._section(widget, "Mapa zakładek")
        self._bullet_list(
            widget,
            [
                "Z1 Kampania: graf, bramki, zasoby, praca bramki, ślad projektu i przejścia między iteracjami.",
                "Z2 Tablice: kontrola ramek tablic na zdjęciach, import AT do kontroli, zatwierdzanie [OK] do puli projektu.",
                "Z3 Znaki: PZ2 detekcja/ocr/manualne poprawki znaków, PZ3 eksport datasetu znaków.",
                "Z4 Dataset i trening: PZ1 warianty datasetów, PZ2 trening, historia runów, ranking, porównania i eksport mobilny.",
                "Z5 Instrukcje: ten przewodnik, mapa kodu i skrócona metodologia eksperymentów.",
            ],
        )

        self._section(widget, "Dwa tryby pracy")
        self._bullet_list(
            widget,
            [
                "Tryb kampanii prowadzi pracę przez graf i iteracje. To tryb docelowy do stabilnego budowania projektu badawczego.",
                "Tryb swobodny pozwala pracować bez pełnego grafu. Jest przydatny do testów, inspekcji i pracy technicznej, ale nie zastępuje śladu kampanii.",
            ],
        )

        self._section(widget, "Przepływ dla modelu tablic MT")
        self._numbered_list(
            widget,
            [
                "Wybieramy zbiór obrazów O.",
                "Dodajemy albo wytwarzamy anotacje tablic AT zgodne z O.",
                "AT importowane z zewnątrz trafiają do kontroli w Z2, a nie od razu do puli [OK].",
                "W Z2 użytkownik poprawia ramki i zatwierdza obrazy/tablice jako [OK].",
                "Zatwierdzone tablice zasilają projektową pulę YOLO i mogą posłużyć do wariantu datasetu tablic.",
                "W Z4 trenujemy model tablic MT, wybieramy wynik bramki i domykamy iterację przez T06.",
            ],
        )

        self._section(widget, "Przepływ dla modelu znaków MZ")
        self._numbered_list(
            widget,
            [
                "Źródłem są zatwierdzone tablice z bieżącej iteracji albo z zasobów dziedziczonych przez projekt.",
                "Z3/PZ2 wykonuje detekcję i korektę znaków na wyodrębnionych tablicach.",
                "Pipeline znaków składa się z prawdziwych bloków: YB wykrywa boxy, YS pracuje na istniejących boxach, OCR odczytuje tekst całej tablicy i może wspomóc podział.",
                "Manualne poprawki mają pierwszeństwo przed automatycznymi wynikami. Detekcja nie powinna niszczyć manuali.",
                "Z3/PZ3 eksportuje dataset znaków AZ/MZ do treningu.",
                "Z4 trenuje model znaków MZ, pozwala porównać modele i wskazać wynik bramki.",
            ],
        )

        self._section(widget, "Statusy, które warto rozumieć")
        self._bullet_list(
            widget,
            [
                "Bieżąca iteracja: praca wykonana i zatwierdzona w bieżącym cyklu grafu.",
                "Dziedziczone z poprzednich iteracji: zasób istnieje i może spełniać minimum, ale nie jest przyrostem bieżącej iteracji.",
                "Do kontroli: zasób pasuje technicznie, ale nie jest jeszcze zatwierdzony przez użytkownika w odpowiedniej karcie.",
                "Przerwana praca: karta została opuszczona lub aplikacja została zamknięta bez formalnego powrotu do grafu.",
                "Gotowa do zatwierdzenia: bramka ma spełniony kontrakt, ale decyzja zamknięcia nadal należy do użytkownika.",
            ],
        )

        self._section(widget, "Słownik skrótów")
        self._bullet_list(
            widget,
            [
                "O: zbiór obrazów wejściowych.",
                "AT: anotacje tablic na obrazach.",
                "AZ: anotacje znaków na wyodrębnionych tablicach.",
                "MT: model tablic, czyli detektor tablic.",
                "MZ: model znaków, czyli detektor znaków na cropie tablicy.",
                "YB: blok detekcji ramek znaków.",
                "YS: blok pracy na znakach w istniejących ramkach.",
                "OCR: odczyt tekstu tablicy; sam OCR nie jest źródłem precyzyjnych boxów znaków.",
                "data.yaml: opis datasetu YOLO. Przy eksporcie mobilnym jest używany głównie do kalibracji INT8.",
                ".alprmodel: paczka eksportowa dla klienta mobilnego, zawierająca manifest, modele, etykiety i ustawienia inferencji.",
            ],
        )

    def _fill_campaign(self, widget):
        self._paragraph(
            widget,
            "Kampania jest maszyną stanów. Jej zadaniem jest pilnowanie, co użytkownik wybrał, "
            "co zostało faktycznie wykonane, które zasoby są kandydatami, a które są "
            "zatwierdzonym materiałem projektowym.",
        )

        self._section(widget, "Graf kampanii")
        self._code_block(
            widget,
            """
E1 --T01--> E2 --T03--> E3 --T05--> E4Z --T06--> E1
             `--T04--> E4T --T06--> E1

E1 --T02--> E3
            """,
        )
        self._callout(
            widget,
            "Graf kampanii obejmuje bramki T01-T06. Zamknięcie iteracji po treningu albo świadomym "
            "pominięciu treningu odbywa się przez T06.",
        )

        self._section(widget, "Bramki")
        self._bullet_list(
            widget,
            [
                "T01: wybór ścieżki przez E2, czyli praca od obrazów i anotacji tablic.",
                "T02: skrót E1 -> E3, czyli przejście do pracy nad znakami na podstawie dostępnych lub importowanych zasobów.",
                "T03: przekazanie zatwierdzonych tablic z E2 do pracy nad znakami w E3.",
                "T04: przygotowanie datasetu i trening modelu tablic MT.",
                "T05: przygotowanie znaków, datasetu znaków i przejście do treningu modelu znaków MZ.",
                "T06: domknięcie toru treningowego i powrót do E1 kolejnej iteracji.",
            ],
        )

        self._section(widget, "Tory pracy")
        self._bullet_list(
            widget,
            [
                "Tor tablic: E1 -> T01 -> E2 -> T04 -> E4T -> T06 -> E1.",
                "Tor znaków od obrazów: E1 -> T01 -> E2 -> T03 -> E3 -> T05 -> E4Z -> T06 -> E1.",
                "Tor znaków od gotowych tablic: E1 -> T02 -> E3 -> T05 -> E4Z -> T06 -> E1.",
            ],
        )

        self._section(widget, "Kontrakty zasobów")
        self._paragraph(
            widget,
            "Najdelikatniejsza część programu to powiązanie zasobów. Dlatego obowiązuje zasada: "
            "nie wystarczy, że zasób istnieje. Musi być zgodny z kontekstem iteracji.",
        )
        self._bullet_list(
            widget,
            [
                "O i AT tworzą komplet: anotacje AT muszą pasować do obrazów O po nazwach plików i zakresie danych.",
                "O/AT prowadzą do wyodrębnionych tablic, które dalej mogą być kontrolowane jako materiał dla znaków.",
                "AZ działa na wyodrębnionych tablicach, a nie bezpośrednio na pierwotnym zbiorze zdjęć.",
                "MT i MZ są niezależne od aktywnego katalogu obrazów. Model może być użyty ponownie, ale jego wybór musi być jawny.",
                "Zmiana O nie powinna po cichu kasować AT/AZ, ale powinna oznaczyć je jako wymagające ponownego dopasowania lub kontroli.",
            ],
        )

        self._section(widget, "Import AT w T01/T02")
        self._paragraph(
            widget,
            "Import AT nigdy nie powinien automatycznie oznaczać materiału jako [OK]. Pasujące anotacje trafiają "
            "do kontroli w Z2. Dopiero po kontroli i zatwierdzeniu obraz/tablica staje się materiałem projektowym.",
        )
        self._bullet_list(
            widget,
            [
                "Pasujące AT: anotacje, których nazwy plików występują w wybranym zbiorze O.",
                "Niepasujące AT: anotacje odrzucane w tym imporcie, bo nie mają odpowiedniego obrazu w O.",
                "Zatwierdzone [OK]: tylko ta część przechodzi dalej i powinna być liczona jako przyrost iteracji.",
                "Pominięte lub nieskontrolowane: nie powinny otwierać bramki ani udawać gotowego materiału.",
            ],
        )

        self._section(widget, "Zmiana decyzji T01/T02")
        self._paragraph(
            widget,
            "T01 i T02 są alternatywami startowymi tej samej iteracji. Zmiana decyzji jest bezpieczna tylko zanim "
            "użytkownik zatwierdzi materiał, który realnie zmienia pulę projektową.",
        )
        self._bullet_list(
            widget,
            [
                "Jeśli w T02 zatwierdzono kontrolę i obrazy/tablice zostały dodane do puli projektowej, T01 w tej iteracji powinno zostać zablokowane.",
                "Analogicznie, jeśli T01 wykonało zatwierdzony przyrost, T02 nie powinno po cichu przejmować innego stanu.",
                "Niedostępna alternatywa powinna być zwinięta albo wizualnie wygaszona bez zielonych badge sugerujących aktywność.",
            ],
        )

        self._section(widget, "Praca w bieżącej iteracji")
        self._paragraph(
            widget,
            "Pole Praca w bramce opisuje działania bieżącej iteracji. Jeśli bramka może być zatwierdzona tylko dlatego, "
            "że odziedziczyła zasoby z poprzednich iteracji, UI powinien powiedzieć to wprost.",
        )
        self._bullet_list(
            widget,
            [
                "Wykonane oznacza: w bieżącej iteracji powstał zatwierdzony rezultat.",
                "Brak przyrostu oznacza: można korzystać z dziedziczonych zasobów, ale w tej iteracji nic nowego jeszcze nie dodano.",
                "Przerwana oznacza: użytkownik rozpoczął pracę w karcie i opuścił ją bez domknięcia właściwym CTA.",
                "Zalecane CTA powinno wskazywać najbliższy sensowny krok w bieżącej iteracji.",
            ],
        )

    def _fill_architecture(self, widget):
        self._paragraph(
            widget,
            "Ta część pomaga odnaleźć się w kodzie. Nazwy plików są tu podane po to, żeby kolejny agent "
            "albo człowiek nie musiał zgadywać, gdzie mieszka dany fragment programu.",
        )

        self._section(widget, "Warstwy aplikacji")
        self._bullet_list(
            widget,
            [
                "Bootstrap i konfiguracja: main.py, dependency_bootstrap.py, config.py, session.py, utils.py.",
                "Shell aplikacji: gui/app.py, app_startup.py, app_shutdown.py, app_theme.py, app_menu.py.",
                "Kampania i graf: campaign_manager.py, gui/tab_campaign.py, gui/campaign_dashboard_ui.py, gui/campaign_graph_actions.py.",
                "Kontrakty zasobów: campaign_resource_state.py, campaign_resource_catalog.py, campaign_resource_contracts.py, campaign_transition_specs.py.",
                "Z2 tablice: gui/tab_annotation.py oraz moduły z2_* odpowiedzialne za canvas, import, workflow, staging i prawy panel.",
                "Z3 znaki: gui/tab_character_annotation.py oraz moduły z3_* odpowiedzialne za pipeline, detekcję, preview, metadata i PZ3.",
                "Z4 dataset/trening: gui/tab_training.py oraz moduły z4_* odpowiedzialne za dataset, trening, ranking, porównania i eksport.",
                "Eksport mobilny: gui/z4_model_export.py i dokumentacja kontraktu w docs/eksport_mobilny_kwantyzacja.md.",
                "Pomoc kontekstowa AS: gui/free_mode_assistant.py i integracja w gui/app.py oraz shellach kart.",
            ],
        )

        self._section(widget, "Drzewo robocze")
        self._paragraph(
            widget,
            "Program ma tworzyć potrzebne katalogi po przeniesieniu źródeł na inną maszynę. "
            "Nie zakładamy ręcznego zakładania struktury przez użytkownika.",
        )
        self._code_block(
            widget,
            """
Workspace/
  1_raw/
  2_annotations/
  3_auto_annotations/
  4_datasets/
  5_training_runs/
  6_exports/
  7_rankings/
  8_presets/
  9_projects/
    <projekt>/
      _staging/
      5_training_runs/
      ...
            """,
        )

        self._section(widget, "Najważniejsze artefakty")
        self._bullet_list(
            widget,
            [
                "annotations.xml: anotacje CVAT dla tablic lub znaków.",
                "manifest/run metadata: opis runu, źródła danych, liczników i stanu przerwania.",
                "data.yaml: opis datasetu YOLO oraz punkt wejścia dla treningu lub kalibracji eksportu INT8.",
                "best.pt: najlepszy checkpoint treningu według metryki walidacyjnej.",
                "last.pt: ostatni checkpoint, zwykle punkt wznowienia po przerwanym treningu.",
                "training_history.json i metadane runów: źródło historii, rankingu, wyboru modelu i eksportu.",
                ".alprmodel: paczka mobilna z manifestem, wariantami modelu i parametrami inferencji.",
            ],
        )

        self._section(widget, "Zasada jednego źródła prawdy")
        self._paragraph(
            widget,
            "UI może prezentować statusy, ale nie powinien ich wymyślać. Prawda o projekcie powinna pochodzić "
            "z CampaignManagera, katalogu zasobów, kontraktów i metadanych artefaktów. To chroni przed "
            "niespójnościami typu: bramka wygląda na otwartą, ale zasoby po wejściu do modala mówią coś innego.",
        )

        self._section(widget, "Reguły bezpieczeństwa przepływu")
        self._bullet_list(
            widget,
            [
                "Logika bramek opiera się na jawnej numeracji T01-T06 i tym samym kontrakcie w UI, stanie projektu oraz kodzie.",
                "Nie promujemy automatycznie modelu, datasetu ani importowanych anotacji bez jawnego wyboru lub kontroli.",
                "Karta robocza musi wiedzieć, z której bramki została otwarta i do której bramki ma wrócić.",
                "Długie operacje muszą mieć realny splash/progress, a nie puste okno albo 100% wiszące przez kilkanaście sekund.",
                "Manuale użytkownika mają pierwszeństwo przed automatem. Detekcja może pomagać, ale nie niszczyć ręcznych poprawek.",
                "Nazwy długich artefaktów powinny mieć czytelne ID: MZ/MT dla modeli, DS dla datasetów, TRN dla runów.",
            ],
        )

        self._section(widget, "Dokumenty wspierające")
        self._bullet_list(
            widget,
            [
                "docs/mapa_funkcji_i_kodu.md: rozbudowana mapa funkcji i plików.",
                "docs/eksport_mobilny_kwantyzacja.md: kontrakt eksportu, formaty, kwantyzacja i pakiet .alprmodel.",
                "docs/siatka_eksperymentow_mobilnych_alpr.md: plan eksperymentów i metryki porównawcze.",
                "docs/podbudowa_literaturowa_metodyki_testow_alpr.md: podbudowa literaturowa testów.",
                "docs/specyfikacja_agenta_aplikacji_mobilnej_alpr.md: handoff dla agenta Android.",
            ],
        )

    def _fill_export_research(self, widget):
        self._paragraph(
            widget,
            "Eksport mobilny nie jest osobnym światem. To ostatni odcinek tej samej historii: "
            "wybieramy modele, dokumentujemy parametry, tworzymy paczkę i sprawdzamy ją na urządzeniu.",
        )

        self._section(widget, "Model mobilny i kompletny pakiet ALPR")
        self._bullet_list(
            widget,
            [
                "Model mobilny alpr.model.v1 zawiera jeden MP, MT albo MZ. Służy do testu izolowanego lub podmiany jednej roli na telefonie.",
                "Kompletny pakiet ALPR alpr.package.v1 wymaga MT+MZ; opcjonalny MP rozszerza go do MP+MT+MZ.",
                "MP jest opcjonalnym detektorem pojazdów. Jeśli ma trafić do paczki, ALPR Desktop pobiera lub wskazuje model i wykonuje konwersję przed przekazaniem go do Androida.",
                "Paczka .alprmodel jest archiwum z manifestem, wariantami modeli, etykietami, progami, metadanymi i sumami SHA-256.",
            ],
        )

        self._section(widget, "Formaty eksportu")
        self._bullet_list(
            widget,
            [
                "LiteRT/TFLite FP32: najbezpieczniejszy punkt startowy dla Androida i wariant referencyjny.",
                "LiteRT/TFLite INT8: wariant badawczy i wdrożeniowy, mniejszy i potencjalnie szybszy, ale wymagający kalibracji.",
                "ONNX FP32: wariant kontrolny do testów narzędziowych i porównania poza Androidem.",
                "NCNN: wariant eksperymentalny, sensowny dopiero jeśli klient mobilny ma dla niego runtime.",
            ],
        )
        self._callout(
            widget,
            "Zaznaczenie kilku formatów oznacza eksport kilku wariantów tego samego checkpointu, "
            "a nie wybór kilku różnych modeli logicznych.",
        )

        self._section(widget, "Kalibracja data.yaml")
        self._paragraph(
            widget,
            "data.yaml przy eksporcie nie służy do ponownego treningu. Dla INT8 wskazuje reprezentatywny "
            "zbiór obrazów, na którym narzędzie eksportu mierzy typowe zakresy aktywacji modelu.",
        )
        self._bullet_list(
            widget,
            [
                "Dla MT kalibracja powinna używać pełnych obrazów lub scen podobnych do wejścia detektora tablic.",
                "Dla MZ kalibracja powinna używać cropów tablic podobnych do wejścia detektora znaków.",
                "Losowy albo niezgodny YAML może dać szybki plik, ale słaby lub niestabilny wynik inferencji.",
                "FP32 zwykle nie wymaga kalibracji i jest najlepszym wariantem kontrolnym.",
            ],
        )

        self._section(widget, "Parametry w prawym panelu eksportu")
        self._bullet_list(
            widget,
            [
                "imgsz: rozmiar wejścia modelu w inferencji; wpływa na jakość, czas i pamięć.",
                "conf: próg pewności detekcji; niższy próg zwiększa czułość, ale może dodać fałszywe wykrycia.",
                "IoU: próg NMS; decyduje, kiedy nachodzące detekcje są traktowane jako duplikaty.",
                "Kwantyzacja: sposób zmniejszenia precyzji wag/aktywacji, zwykle po to, by przyspieszyć model i zmniejszyć rozmiar.",
                "Nazwa pliku eksportu: powinna być generowana z ID projektu, iteracji, modeli, formatu i daty, ale użytkownik może ją doprecyzować.",
            ],
        )

        self._section(widget, "Metodyka eksperymentów")
        self._paragraph(
            widget,
            "W pracy badawczej nie wybieramy modelu na oko. Porównujemy kandydatów na tej samej puli rankingowej, "
            "a finalny test zostawiamy jako nietykany sprawdzian końcowy.",
        )
        self._numbered_list(
            widget,
            [
                "Trening używa train/val i zapisuje metryki modelu.",
                "Ranking porównuje kandydatów na wspólnym zbiorze rankingowym.",
                "Eksport tworzy warianty mobilne tych samych checkpointów.",
                "Aplikacja Android mierzy pakiet end-to-end na urządzeniu.",
                "Finalny test potwierdza wynik dopiero po wyborze kandydatów.",
            ],
        )

        self._section(widget, "Metryki i wykresy")
        self._bullet_list(
            widget,
            [
                "Jakość MT: mAP50, mAP50-95, precision, recall, F1 i skuteczność wycinania tablic.",
                "Jakość MZ: mAP50, mAP50-95, precision, recall, F1, accuracy znaków, CER i accuracy całej tablicy.",
                "Jakość pakietu: poprawny odczyt pełnej rejestracji, błędy brakujących znaków, nadmiarowych znaków i duplikatów.",
                "Wydajność mobilna: latency p50/p90/p95, FPS, RAM, rozmiar paczki, zużycie energii jeśli klient to mierzy.",
                "Wykresy: jakość kontra czas, FP32 kontra INT8, ranking pakietów, krzywe precision/recall i rozkład błędów.",
            ],
        )

        self._section(widget, "Odpowiedzialność ALPR i aplikacji mobilnej")
        self._bullet_list(
            widget,
            [
                "ALPR odpowiada za trening, ranking, metadane, eksport, manifest i powtarzalność pakietu.",
                "Aplikacja mobilna odpowiada za uruchomienie paczki na urządzeniu, pomiar czasu, pamięci i prezentację demonstracyjną.",
                "Wynik badawczy powstaje dopiero po połączeniu obu stron: jakości modeli z ALPR i realnej pracy na smartfonie.",
            ],
        )

        self._section(widget, "Literatura i standardy, na których opieramy opis")
        self._bullet_list(
            widget,
            [
                "YOLO/Ultralytics: metryki detekcji, eksport modeli, format data.yaml i praktyka treningu detektorów.",
                "TensorFlow Lite / LiteRT: inferencja mobilna i kwantyzacja INT8 na zbiorze reprezentatywnym.",
                "ONNX: przenośny format grafu modelu i wariant kontrolny eksportu.",
                "MLPerf Mobile: rozdzielenie jakości modelu, czasu inferencji i warunków testu urządzenia.",
                "Klasyczne metryki IR/CV: precision, recall, F1, IoU, mAP i analiza błędów end-to-end.",
            ],
        )
