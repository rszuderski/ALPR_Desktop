from __future__ import annotations

from dataclasses import dataclass
import re
import tkinter as tk

from .web_slim_scrollbar import WebSlimScrollbar, blend_hex_colors


FREE_MODE_ASSISTANT_GLOSSARY: dict[str, str] = {
    "AS": "globalny asystent kontekstu; wyjaśnia bieżącą kartę, najbliższy krok, pojęcia i ryzyka bez wykonywania akcji.",
    "AT": "anotacje tablic powiązane ze zbiorem obrazów; po imporcie wymagają kontroli w Z2 przed zasileniem puli projektu.",
    "AZ": "anotacje znaków na wyodrębnionych tablicach; są źródłem datasetu znaków i treningu modelu MZ.",
    "YB": "blok pipeline znaków odpowiedzialny za wykrywanie ramek znaków na tablicy.",
    "YS": "blok pipeline znaków pracujący na istniejących ramkach i przypisujący klasy znaków.",
    "MB": "manual box, czyli ręcznie ustawiona ramka znaku mająca pierwszeństwo przed automatem.",
    "MS": "manual sign, czyli ręcznie wpisany znak mający pierwszeństwo przed automatycznym odczytem.",
    "bieżąca iteracja": "cykl pracy projektu, w którym użytkownik podejmuje decyzje i wytwarza przyrost danych albo modelu.",
    "do kontroli": "stan zasobu zgodnego technicznie, ale wymagającego ręcznego sprawdzenia przed użyciem jako materiał projektowy.",
    "komplet O-AT": "para zbioru obrazów i pasujących anotacji tablic; tylko zgodny komplet może prowadzić do kontroli i dalszej pracy.",
    "kontrakt zasobu": "jednoznaczna ocena zasobu: wymagany, opcjonalny, spełniony, do kontroli albo niespełniony w danym kontekście.",
    "model projektowy": "model wskazany jako wynik bramki lub domyślny model projektu dla kolejnych iteracji.",
    "model startowy": "checkpoint wybrany jako punkt startu treningu; nie jest tym samym co wynik bramki.",
    "przyrost iteracji": "materiał dodany i zatwierdzony w bieżącej iteracji, odróżniany od zasobów dziedziczonych.",
    "wynik bramki": "artefakt jawnie wybrany jako rezultat przejścia, np. model po treningu albo zatwierdzony dataset.",
    "zasób dziedziczony": "zasób pochodzący z poprzednich iteracji projektu; może spełniać kontrakt, ale nie jest przyrostem bieżącej iteracji.",
    "T01": "bramka E1 -> E2; prowadzi do pracy od obrazów i anotacji tablic.",
    "T02": "bramka E1 -> E3; skrót do pracy nad znakami na podstawie dostępnych tablic.",
    "T03": "bramka E2 -> E3; przekazuje zatwierdzone tablice do pracy nad znakami.",
    "T04": "bramka E2 -> E4T; prowadzi do datasetu i treningu modelu tablic.",
    "T05": "bramka E3 -> E4Z; prowadzi do przygotowania datasetu znaków i modelu znaków.",
    "T06": "bramka E4T/E4Z -> E1; domyka iterację po treningu albo świadomym pominięciu treningu.",
    "akcje": "operacje dostępne w polu Praca bramki; prowadzą do kart roboczych, np. Z2, Z3 albo Z4.",
    "bramka": "interaktywny panel przy kraw\u0119dzi grafu kampanii; pokazuje status przej\u015bcia, zasoby, akcje i zatwierdzenie.",
    "elektroda": "prze\u0142\u0105cznik wyboru bramki na grafie; po jej w\u0142\u0105czeniu dana \u015bcie\u017cka staje si\u0119 aktywna.",
    "graf": "mapa przej\u015b\u0107 kampanii pokazuj\u0105ca etapy jako w\u0119z\u0142y oraz decyzje jako kraw\u0119dzie z bramkami.",
    "kraw\u0119d\u017a": "po\u0142\u0105czenie mi\u0119dzy etapami grafu kampanii; reprezentuje jedno mo\u017cliwe przej\u015bcie.",
    "w\u0119ze\u0142": "g\u0142\u00f3wny punkt grafu, np. etap E1, E2, E3, E4T albo E4Z.",
    "zasoby": "dane wymagane przez wybran\u0105 bramk\u0119, np. katalog zdj\u0119\u0107, model albo anotacje.",
    "zatwierd\u017a": "pole bramki zamykaj\u0105ce wybrane przej\u015bcie po spe\u0142nieniu jego warunk\u00f3w.",
    "anotacja": "opis obiektów na obrazie, np. położenie tablicy, znaku, boxu albo poligonu.",
    "artefakt": "plik lub katalog wytworzony przez krok aplikacji, np. XML, cropy, dataset albo model.",
    "autoanotacja": "automatyczne tworzenie wstępnych ramek przez model; użytkownik później sprawdza i poprawia wynik.",
    "box": "prostokątna ramka opisująca położenie obiektu na obrazie, np. tablicy albo znaku.",
    "batch": "liczba obrazów przetwarzanych jednocześnie podczas treningu; większy batch szybciej zużywa pamięć GPU.",
    "checkpoint": "zapisany plik modelu, zwykle .pt, od którego można zacząć inferencję, walidację albo dalszy trening.",
    "confidence": "próg pewności detekcji; niższy próg daje więcej propozycji, wyższy odrzuca słabsze trafienia.",
    "crop": "wycięty fragment obrazu, np. sama tablica wycięta z pełnego zdjęcia pojazdu.",
    "CVAT": "zewnętrzne narzędzie do ręcznego poprawiania anotacji obrazów.",
    "data.yaml": "plik opisu datasetu YOLO: wskazuje klasy oraz ścieżki train, val i test. Przy eksporcie INT8 służy jako reprezentatywne źródło obrazów do kalibracji, a nie jako nowy trening.",
    "dataset": "uporządkowany zestaw danych treningowych: obrazy oraz odpowiadające im etykiety/anotacje.",
    "epoka": "jedno pełne przejście treningu po danych treningowych.",
    "eksport": "zapisanie gotowych danych do formatu używanego dalej, np. XML, YOLO albo zestawu CVAT.",
    "fit": "dopasowanie ramek do obrazu lub obszaru pracy, aby wynik był spójny z podglądem.",
    "gold pack": "wybrany, zaufany zestaw przykładów, z którego buduje się lepszy dataset znaków.",
    "GPU": "karta graficzna używana do szybszej inferencji lub treningu modeli.",
    "inferencja": "uruchomienie gotowego modelu na danych, aby uzyskać predykcje, np. boxy albo odczyt znaków.",
    "INT8": "wariant kwantyzowany do 8-bitowych liczb całkowitych; może być mniejszy i szybszy na telefonie, ale wymaga kalibracji i kontroli jakości względem FP32.",
    "IoU": "próg nakładania ramek używany m.in. w NMS; wpływa na to, czy nakładające się detekcje zostaną połączone, odrzucone albo zostawione.",
    "iteracja": "jeden pełny cykl pracy projektu: przygotowanie danych, anotacja, ewentualnie znaki, dataset i trening.",
    "kalibracja": "pomiarowy etap konwersji INT8: konwerter przepuszcza reprezentatywne obrazy przez model i dobiera zakresy liczbowe aktywacji.",
    "kampania": "projekt prowadzony etapami przez wizard, z pamięcią iteracji, modeli i zatwierdzonych artefaktów.",
    "katalog": "folder na dysku zawierający dane danego kroku, np. obrazy, run albo gotowy dataset.",
    "korekta": "ręczne sprawdzenie i poprawienie anotacji po automatycznym albo wcześniejszym etapie pracy.",
    "klasa": "nazwa typu obiektu, którego uczy się model, np. konkretnego znaku albo tablicy.",
    "modal": "okno dialogowe wymagające decyzji użytkownika przed kontynuacją danego działania.",
    "model": "plik wag lub konfiguracja sieci neuronowej używana do detekcji, OCR albo treningu.",
    "manifest": "plik metadanych pakietu .alprmodel; opisuje rolę modelu, warianty wykonawcze, progi, etykiety, wejścia/wyjścia i sumy kontrolne.",
    "MT": "model tablic; wykrywa tablice na pełnym obrazie albo w scenie wejściowej.",
    "MZ": "model znaków; pracuje na wyciętej tablicy i wykrywa znaki potrzebne do odczytu numeru.",
    "obraz": "pojedynczy plik graficzny używany jako wejście do anotacji, datasetu albo treningu.",
    "NCNN": "opcjonalny mobilny format wykonawczy z plikami .param i .bin; traktujemy go jako wariant eksperymentalny, dopóki klient mobilny nie ma pełnej ścieżki NCNN.",
    "NMS": "post-processing usuwający nadmiarowe, nakładające się detekcje; korzysta m.in. z progu IoU.",
    "OCR": "rozpoznawanie znaków z obrazu, np. odczyt liter i cyfr z wyciętej tablicy.",
    "ONNX": "format kontrolny i diagnostyczny modelu; pomaga porównywać wynik eksportu z checkpointem, ale na Androidzie zwykle jest fallbackiem, nie główną ścieżką.",
    "overlay": "nakładka na obszar roboczy pokazująca stan procesu, postęp albo krótkie sterowanie bez przechodzenia do innej karty.",
    "plik .alprmodel": "archiwum ZIP: jeden model mobilny alpr.model.v1 (MP, MT lub MZ) albo kompletny pakiet ALPR alpr.package.v1 (MT+MZ lub MP+MT+MZ).",
    "model mobilny": "jeden MP, MT albo MZ w formacie alpr.model.v1; pozwala podmienić jedną rolę na telefonie lub wykonać test izolowany.",
    "pakiet ALPR": "kompletny zestaw MT+MZ albo MP+MT+MZ w formacie alpr.package.v1, przeznaczony do pełnego rozpoznawania tablic.",
    "pakiet MT+MZ": "kompletny pakiet ALPR zawierający model tablic MT, model znaków MZ i opis pipeline; to właściwy kandydat do testu end-to-end na telefonie.",
    "perfect": "status oznaczający, że przykład jest sprawdzony i nadaje się do datasetu.",
    "poligon": "wielopunktowy obrys obiektu; dokładniejszy niż zwykły prostokątny box.",
    "preview run": "roboczy zestaw podglądowy, zwykle używany do sprawdzenia cropów przed dalszym etapem.",
    "PT": "plik wag modelu PyTorch/YOLO, zwykle z rozszerzeniem .pt.",
    "LiteRT/TFLite": "główny format wykonawczy dla Androida; FP32 jest bezpiecznym wariantem referencyjnym, a INT8 wymaga kalibracji.",
    "FP32": "wariant zmiennoprzecinkowy 32-bitowy; zwykle najbardziej zgodny z checkpointem i najlepszy jako punkt odniesienia jakości.",
    "kwantyzacja": "zmiana precyzji liczbowej modelu, np. z FP32 na INT8, aby zmniejszyć rozmiar i koszt inferencji.",
    "ranking": "porównanie wyników modeli lub treningów, pomagające wybrać najlepszy wariant.",
    "review pack": "zestaw przykładów przygotowany do ręcznego sprawdzenia, często poza aplikacją.",
    "runtime": "silnik uruchamiający model na urządzeniu, np. LiteRT/TFLite, ONNX Runtime albo NCNN.",
    "run": "katalog konkretnego przebiegu pracy, zawierający pliki i artefakty danego kroku.",
    "split": "podział datasetu na części: train do uczenia, val do kontroli jakości i test do końcowej oceny.",
    "tablica": "tablica rejestracyjna widoczna na zdjęciu lub wycięta jako crop do dalszej pracy.",
    "test": "część datasetu odkładana do końcowej oceny modelu po treningu.",
    "tor": "wybrana ścieżka pracy, np. tablice, znaki, anotacja ręczna albo autoanotacja.",
    "train": "część datasetu używana bezpośrednio do uczenia modelu.",
    "trening": "proces uczenia modelu na przygotowanym datasecie.",
    "val": "część walidacyjna datasetu, używana do sprawdzania jakości podczas treningu.",
    "walidacja": "sprawdzenie jakości modelu na danych, których nie używa bezpośrednio do uczenia.",
    "wariant": "konkretna wersja datasetu lub konfiguracji, którą można porównać z innymi.",
    "wariant wykonawczy": "konkretny plik modelu w pakiecie, np. TFLite FP32, TFLite INT8 albo ONNX FP32, zbudowany z tego samego checkpointu.",
    "VRAM": "pamięć karty graficznej; jej brak zwykle wymaga mniejszego batcha albo niższej rozdzielczości.",
    "wizard": "prowadzenie projektowe w Z1, które pilnuje kolejności etapów kampanii.",
    "znak": "pojedynczy znak z tablicy, np. litera albo cyfra rozpoznawana w Z3.",
    "XML": "plik anotacji zawierający informacje o obiektach i ich położeniu na obrazach.",
    "YOLO": "rodzina modeli do detekcji obiektów; w aplikacji służy m.in. do tablic, znaków i datasetów YOLO.",
    "YOLO Detect": "tryb YOLO wykrywający prostokątne boxy obiektów.",
    "YOLO Pose": "tryb YOLO uczący punkty/kształt obiektu, np. narożniki albo poligony.",
    "Z1": "zakładka wizarda kampanii i kontroli etapów projektu.",
    "Z2": "zakładka pracy nad tablicami rejestracyjnymi.",
    "Z3": "zakładka pracy nad znakami wyciętymi z tablic.",
    "Z4": "zakładka przygotowania datasetu i treningu modeli.",
    "Z5": "zakładka instrukcji, dziennika architektury i opisów programu.",
    "PZ1": "pierwsza podzakładka bieżącego etapu; jej sens zależy od tego, czy jesteś w Z2, Z3 czy Z4.",
    "PZ2": "druga podzakładka bieżącego etapu; zwykle kolejny krok pracy po PZ1.",
    "PZ3": "trzecia podzakładka bieżącego etapu, najczęściej związana z eksportem albo pracą dodatkową.",
    "źródło": "dane wejściowe potrzebne do danego kroku, np. XML z pasującym katalogiem obrazów albo gotowy dataset.",
    "źródło bez splitu": "dataset YOLO zapisany jako katalog images/labels, jeszcze bez podziału train/val/test. PZ1 tworzy z niego wariant treningowy.",
    "źródło Z2": "zgodna para danych z Z2: XML anotacji tablic oraz katalog obrazów, z których ten XML powstał.",
}

FREE_MODE_ASSISTANT_GLOSSARY_ALIASES: dict[str, str] = {
    "as": "AS",
    "asystent": "AS",
    "globalny as": "AS",
    "adnotacje tablic": "AT",
    "anotacje tablic": "AT",
    "at": "AT",
    "adnotacje znaków": "AZ",
    "anotacje znaków": "AZ",
    "az": "AZ",
    "yb": "YB",
    "ys": "YS",
    "manual box": "MB",
    "manual sign": "MS",
    "mb": "MB",
    "ms": "MS",
    "bramka t01": "T01",
    "bramka t02": "T02",
    "bramka t03": "T03",
    "bramka t04": "T04",
    "bramka t05": "T05",
    "bramka t06": "T06",
    "t01": "T01",
    "t02": "T02",
    "t03": "T03",
    "t04": "T04",
    "t05": "T05",
    "t06": "T06",
    "kontrakt": "kontrakt zasobu",
    "kontrakty": "kontrakt zasobu",
    "kontrakty zasobów": "kontrakt zasobu",
    "komplet o at": "komplet O-AT",
    "komplet o-at": "komplet O-AT",
    "o-at": "komplet O-AT",
    "do sprawdzenia": "do kontroli",
    "kontrola": "do kontroli",
    "przyrost": "przyrost iteracji",
    "przyrost bieżącej iteracji": "przyrost iteracji",
    "dziedziczone": "zasób dziedziczony",
    "zasoby dziedziczone": "zasób dziedziczony",
    "wynik": "wynik bramki",
    "wynik bramki": "wynik bramki",
    "model bazowy": "model startowy",
    "model do treningu": "model startowy",
    "model wynikowy": "wynik bramki",
    "model podpięty": "model projektowy",
    "model przypięty": "model projektowy",
    "akcja": "akcje",
    "akcji": "akcje",
    "praca": "akcje",
    "pole praca": "akcje",
    "praca bramki": "akcje",
    "bramki": "bramka",
    "bramek": "bramka",
    "elektrody": "elektroda",
    "grafu": "graf",
    "krawedz": "kraw\u0119d\u017a",
    "krawedzi": "kraw\u0119d\u017a",
    "kraw\u0119dzi": "kraw\u0119d\u017a",
    "mapa przejsc": "graf",
    "mapa przej\u015b\u0107": "graf",
    "wezel": "w\u0119ze\u0142",
    "wezly": "w\u0119ze\u0142",
    "w\u0119z\u0142y": "w\u0119ze\u0142",
    "zasob": "zasoby",
    "zasobow": "zasoby",
    "zasob\u00f3w": "zasoby",
    "zatwierdz": "zatwierd\u017a",
    "zatwierdzenie": "zatwierd\u017a",
    "adnotacje": "anotacja",
    "anotacje": "anotacja",
    "anotacji": "anotacja",
    "artefakty": "artefakt",
    "boxy": "box",
    "boxów": "box",
    "boxowanie": "box",
    "crop tablicy": "crop",
    "cropy": "crop",
    "cropów": "crop",
    "batch size": "batch",
    "checkpointy": "checkpoint",
    "checkpointów": "checkpoint",
    "datasetu": "dataset",
    "datasety": "dataset",
    "datasetów": "dataset",
    "detekcja": "YOLO Detect",
    "detekcji": "YOLO Detect",
    "epoki": "epoka",
    "etykieta": "anotacja",
    "etykiety": "anotacja",
    "folder": "katalog",
    "folderu": "katalog",
    "foldery": "katalog",
    "klasy": "klasa",
    "klas": "klasa",
    "karta graficzna": "GPU",
    "karty graficznej": "GPU",
    "model PT": "model",
    "modele": "model",
    "modeli": "model",
    "model tablic": "MT",
    "model tablicy": "MT",
    "model znaków": "MZ",
    "model znakow": "MZ",
    "mt": "MT",
    "mz": "MZ",
    "poligony": "poligon",
    "progi": "confidence",
    "próg": "confidence",
    "alprmodel": "plik .alprmodel",
    ".alprmodel": "plik .alprmodel",
    "fp32": "FP32",
    "int8": "INT8",
    "iou": "IoU",
    "kalibracji": "kalibracja",
    "kwantyzacji": "kwantyzacja",
    "litert": "LiteRT/TFLite",
    "lite rt": "LiteRT/TFLite",
    "tflite": "LiteRT/TFLite",
    "manifestu": "manifest",
    "ncnn": "NCNN",
    "nms": "NMS",
    "onnx": "ONNX",
    "pakiet": "pakiet ALPR",
    "pakietu": "pakiet ALPR",
    "pakiet mobilny alpr": "pakiet ALPR",
    "pakiet mobilny": "plik .alprmodel",
    "pakiet alpr": "pakiet ALPR",
    "pakiet mt mz": "pakiet MT+MZ",
    "pakiet mt+mz": "pakiet MT+MZ",
    "ramka": "box",
    "ramki": "box",
    "ramek": "box",
    "runu": "run",
    "runy": "run",
    "splity": "split",
    "splitu": "split",
    "tablica perfect": "perfect",
    "tablice": "tablica",
    "tablic": "tablica",
    "treningu": "trening",
    "walidacji": "walidacja",
    "walidować": "walidacja",
    "wariantu": "wariant",
    "warianty": "wariant",
    "yaml": "data.yaml",
    "yolo detect": "YOLO Detect",
    "yolo pose": "YOLO Pose",
    "yolo znaków": "YOLO",
    "zdjęcie": "obraz",
    "zdjęcia": "obraz",
    "zdjęć": "obraz",
    "znaki": "znak",
    "znaków": "znak",
    "źródła": "źródło",
    "źródło images labels": "źródło bez splitu",
    "źródło images/labels": "źródło bez splitu",
    "źródło niesplitowane": "źródło bez splitu",
    "źródłowy": "źródło",
    "źródłowych": "źródło",
}


def get_mobile_export_assistant_context() -> dict:
    return {
        "location": "[Integracje] Eksport mobilny",
        "goal": (
            "Ten ekran eksportuje gotowe checkpointy jako model mobilny lub kompletny pakiet ALPR, "
            "który aplikacja Android może bezpiecznie zaimportować, zwalidować i uruchomić."
        ),
        "current": (
            "Pojedynczy MP, MT albo MZ pozwala podmienić jedną rolę i wykonać test izolowany. Kompletny pakiet ALPR zawiera "
            "MT+MZ albo MP+MT+MZ z manifestem, wariantami runtime i progami inferencji."
        ),
        "workflow": (
            "Wybierz model mobilny MP, MT lub MZ do podmiany jednej roli albo pakiet ALPR MT+MZ lub MP+MT+MZ do testu całego potoku.",
            "Formaty w prawym panelu to warianty wykonawcze tego samego checkpointu, np. LiteRT/TFLite FP32, LiteRT/TFLite INT8, ONNX FP32 albo NCNN.",
            "data.yaml wskazujesz tylko wtedy, gdy eksportujesz INT8. To nie są importowane obrazy i nie jest trening, tylko próbka do kalibracji zakresów liczbowych.",
            "Modal eksportu podpowiada zgodne pliki data.yaml: najpierw dataset przypisany do modelu, potem zgodne datasety projektu i katalogi globalne danego toru.",
            "Dla MP wybieraj YAML pojazdów, dla MT YAML tablic/pose, a dla MZ YAML znaków. Nie mieszaj torów, bo kalibracja INT8 może wtedy pogorszyć inferencję mimo poprawnego pliku wyjściowego.",
            "imgsz, conf i IoU trafiają do manifestu jako domyślne parametry inferencji/post-processingu na telefonie; nie zmieniają wytrenowanego checkpointu.",
            "Pakiet .alprmodel zawiera manifest, warianty modelu, etykiety, progi, metadane, wersję kontraktu i sumy SHA-256.",
        ),
        "glossary": (
            "model mobilny = pojedynczy MP, MT albo MZ, schema alpr.model.v1",
            "pakiet ALPR = MT+MZ albo MP+MT+MZ do testu end-to-end, schema alpr.package.v1",
            "data.yaml = opis datasetu YOLO używany przy INT8 jako źródło kalibracji",
            "zgodny data.yaml = YAML z tego samego toru co model: MP->pojazdy, MT->tablice/pose, MZ->znaki",
            "kalibracja = pomiar zakresów aktywacji na reprezentatywnych obrazach",
            "kwantyzacja = zmiana precyzji liczbowej modelu, np. FP32 -> INT8",
            "LiteRT/TFLite = główny format wykonawczy dla Androida",
            "ONNX = wariant kontrolny/fallback i narzędzie diagnostyczne",
            "NCNN = eksperymentalny wariant runtime mobilnego",
            "manifest = kontrakt dla aplikacji mobilnej: co jest w pakiecie i jak to uruchomić",
            "wariant wykonawczy = jeden format modelu zbudowany z tego samego best.pt",
        ),
        "caution": (
            "Nie wybieraj przypadkowego data.yaml do INT8. Zła kalibracja może dać szybki, mały model, "
            "który gorzej rozpoznaje tablice lub znaki. Jeśli nie masz reprezentatywnego YAML-a dla danego toru, "
            "bezpieczniejszy do testu porównawczego jest FP32. Jeden MZ nie wykona pełnego ALPR; do demonstracji end-to-end "
            "potrzebny jest komplet MT+MZ."
        ),
        "references": (
            "docs/eksport_mobilny_kwantyzacja.md",
            "docs/siatka_eksperymentow_mobilnych_alpr.md",
            "docs/specyfikacja_agenta_aplikacji_mobilnej_alpr.md",
        ),
    }


def _normalize_glossary_key(value: str) -> str:
    raw = str(value or "").strip().strip(" .,:;()[]{}")
    if not raw:
        return ""
    raw = raw.split("=", 1)[0].strip().strip(" .,:;()[]{}")
    folded = raw.casefold()
    for key in FREE_MODE_ASSISTANT_GLOSSARY:
        if folded == key.casefold():
            return key
    alias = FREE_MODE_ASSISTANT_GLOSSARY_ALIASES.get(folded)
    if alias:
        return alias
    return raw


def _contains_keyword(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def get_step3_free_mode_assistant_context(host) -> dict:
    try:
        selected_tab = str(host.main_nb.select())
    except Exception:
        selected_tab = ""

    if selected_tab == str(getattr(host, "tab_extract", "")):
        return {
            "location": "[Z3] Autoanotacja znaków tablic / [PZ1] Wyodrębnianie zaanotowanych tablic",
            "goal": "Ta podzakładka bierze źródło z Z2, czyli XML i zgodny katalog obrazów, a następnie wyodrębnia tablice do dalszej pracy nad znakami.",
            "current": "Źródło Z2 musi być zgodnym kompletem obrazy + anotacje tablic. Wyodrębnione tablice stają się wejściem PZ2.",
            "workflow": (
                "Wskaż albo potwierdź źródło Z2: annotations.xml oraz zgodny folder obrazów.",
                "Jeśli program znajdzie pasujący folder obrazów, świadomie potwierdź podpięcie w modalu.",
                "Uruchom wyodrębnianie tablic, aby utworzyć preview run z cropami tablic.",
                "Po sukcesie przejdź do PZ2, gdzie będziesz analizować znaki na wyodrębnionych tablicach.",
            ),
            "glossary": (
                "crop = wycięty obraz samej tablicy",
                "preview run = roboczy zestaw cropów",
                "źródło Z2 = XML + zgodne obrazy",
            ),
            "caution": "XML i katalog obrazów muszą pochodzić z tego samego zestawu; inaczej cropy będą niespójne.",
            "references": ("docs/mapa_funkcji_i_kodu.md",),
        }
    if selected_tab == str(getattr(host, "tab_detect", "")):
        return {
            "location": "[Z3] Autoanotacja znaków tablic / [PZ2] Wykrywanie znaków i analiza",
            "goal": (
                "PZ2 jest pierwszym krokiem pracy T05: tutaj przygotowujesz anotacje znaków na wyodrębnionych tablicach. "
                "Poprawiasz ramki, wpisujesz znaki i doprowadzasz tablice do statusu perfect. "
                "Sam zbiór przygotowany w PZ2 nie otwiera jeszcze T05; po zbudowaniu sensownego materiału trzeba przejść do PZ3 i wyeksportować źródłowy dataset znaków."
            ),
            "current": (
                "Pipeline PZ2 składa się z bloków YB, YS i OCR. Manualne ramki oraz ręcznie wpisane znaki mają pierwszeństwo przed wynikiem automatu."
            ),
            "workflow": (
                "W PZ2 popraw ramki znaków i doprowadź możliwie dużo tablic do statusu perfect.",
                "Szuflada PZ2 pokazuje lokalny stan pracy: ile jest tablic i znaków, poziom jakości zbioru oraz braki do kolejnego poziomu.",
                "Gdy zbiór jest sensowny, użyj przycisku „Krok 2: dataset PZ3”.",
                "W PZ3 utwórz źródłowy dataset znaków AZ. Dopiero ten eksport domyka warunek bramki T05.",
            ),
            "glossary": (
                "perfect = tablica gotowa do datasetu",
                "YOLO znaków = boxy znaków",
                "OCR = odczyt znaków z boxów",
                "1R = tablica jednorzędowa",
                "2R = tablica dwurzędowa",
                "2R? = program podejrzewa układ dwurzędowy, ale nie ma pewności",
                "2R* / 1R* = układ ręcznie wymuszony przez użytkownika",
                "YB = blok wykrywania ramek znaków",
                "YS = blok rozpoznawania znaków w istniejących ramkach",
                "MB/MS = ręczna ramka lub ręcznie wpisany znak",
            ),
            "caution": "Nie myl poziomu jakości w PZ2 z otwarciem T05. PZ2 przygotowuje materiał, PZ3 tworzy artefakt datasetu.",
            "references": ("docs/mapa_funkcji_i_kodu.md", "DZIENNIK_ARCHITEKTURY_I_ZMIAN.md"),
        }
    if selected_tab == str(getattr(host, "tab_dataset", "")):
        return {
            "location": "[Z3] Autoanotacja znaków tablic / [PZ3] Integracje i dataset (YOLO)",
            "goal": "Ta podzakładka domyka pracę nad znakami: zbiera sprawdzone tablice perfect, opcjonalne poprawki CVAT i eksportuje źródłowy dataset znaków do dalszej pracy w Z4.",
            "current": "Eksport PZ3 tworzy artefakt AZ. Ten artefakt jest później wybierany w Z4/PZ1 jako źródło wariantu treningowego MZ.",
            "workflow": (
                "Sprawdź, że pracujesz na perfectach z aktywnego runu PZ2.",
                "Jeśli poprawki zewnętrzne nie są potrzebne, wybierz strategie i źródła gold packa.",
                "Jeśli potrzebujesz CVAT, wyeksportuj review pack, popraw boxy znaków w CVAT i zaimportuj XML z powrotem do PZ3.",
                "Wyeksportuj źródłowy dataset YOLO znaków i przeczytaj modal z wynikiem operacji.",
                "Przejdź do Z4/PZ1, aby utworzyć wariant treningowy i split; trening uruchamiasz dopiero w Z4/PZ2.",
            ),
            "glossary": (
                "gold pack = wybrane tablice perfect używane jako zaufane źródło datasetu znaków",
                "review pack = zestaw cropów tablic wysyłany do ręcznego sprawdzenia w CVAT",
                "Poprawki CVAT = ręczne korekty boxów znaków wracające z CVAT do PZ3 i włączane do datasetu",
            ),
            "caution": "CVAT w PZ3 używa cropów tablic i boxów znaków, nie boxów tablic na pełnych zdjęciach.",
            "references": ("docs/mapa_funkcji_i_kodu.md", "DZIENNIK_ARCHITEKTURY_I_ZMIAN.md"),
        }
    return {
        "location": "[Z3] Autoanotacja znaków tablic",
        "goal": "Ta zakładka prowadzi od tablic przygotowanych w Z2 do cropów, korekty znaków i źródłowego datasetu znaków.",
        "current": "PZ2 przygotowuje i kontroluje znaki. PZ3 tworzy dataset znaków używany w Z4.",
        "workflow": (
            "PZ1 wyodrębnia tablice z obrazów i XML z Z2.",
            "PZ2 rozpoznaje i poprawia znaki na cropach tablic.",
            "PZ3 zbiera perfecty, opcjonalne poprawki CVAT i eksportuje źródłowy dataset znaków.",
            "Z4 przejmuje dopiero wariant treningowy, split i trening modelu.",
        ),
        "glossary": (
            "PZ1 = wyodrębnianie zaanotowanych tablic",
            "PZ2 = wykrywanie znaków i analiza",
            "PZ3 = integracje i dataset YOLO",
        ),
        "caution": "Wariant treningowy i split końcowo przygotujesz w Z4.",
        "references": ("docs/mapa_funkcji_i_kodu.md",),
    }


@dataclass(frozen=True)
class FreeModeAssistantContext:
    location: str = ""
    goal: str = ""
    current: str = ""
    workflow: tuple[str, ...] = ()
    glossary: tuple[str, ...] = ()
    caution: str = ""
    references: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value) -> "FreeModeAssistantContext":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            glossary = value.get("glossary", ())
            if isinstance(glossary, str):
                glossary = (glossary,)
            workflow = value.get("workflow", value.get("steps", ()))
            if isinstance(workflow, str):
                workflow = (workflow,)
            references = value.get("references", value.get("docs", ()))
            if isinstance(references, str):
                references = (references,)
            return cls(
                location=str(value.get("location", "") or ""),
                goal=str(value.get("goal", "") or ""),
                current=str(value.get("current", value.get("state", "")) or ""),
                workflow=tuple(str(item or "") for item in workflow if str(item or "").strip()),
                glossary=tuple(str(item or "") for item in glossary if str(item or "").strip()),
                caution=str(value.get("caution", "") or ""),
                references=tuple(
                    str(item or "")
                    for item in references or ()
                    if str(item or "").strip()
                ),
            )
        return cls()

    def is_empty(self) -> bool:
        return not any((
            self.location,
            self.goal,
            self.current,
            self.workflow,
            self.glossary,
            self.caution,
            self.references,
        ))

    def render_body(self) -> str:
        return "\n".join(self.render_body_lines())

    def render_body_lines(self) -> tuple[str, ...]:
        lines = []
        if self.location:
            lines.append(f"Jesteś tutaj: {self.location}")
        if self.goal:
            lines.append(f"Co robisz: {self.goal}")
        if self.current:
            lines.append(f"Stan/kontekst: {self.current}")
        if self.workflow:
            lines.append("Kolejność pracy:")
            for index, item in enumerate(self.workflow, start=1):
                lines.append(f"{index}. {item}")
        if self.caution:
            lines.append(f"Uważaj: {self.caution}")
        if self.references:
            lines.append("Dokumenty:")
            for item in self.references:
                lines.append(f"- {item}")
        return tuple(lines)

    def render_glossary_lines(self) -> tuple[str, ...]:
        text_parts = [
            self.location,
            self.goal,
            self.current,
            self.caution,
            *self.workflow,
            *self.glossary,
            *self.references,
        ]
        searchable_text = " ".join(str(part or "") for part in text_parts)
        explicit_defs: dict[str, str] = {}
        ordered_keys: list[str] = []

        def add_key(key: str, definition: str = "") -> None:
            normalized = _normalize_glossary_key(key)
            if not normalized:
                return
            if definition:
                explicit_defs[normalized] = definition
            if normalized not in ordered_keys:
                ordered_keys.append(normalized)

        for entry in self.glossary:
            entry_text = str(entry or "").strip()
            if not entry_text:
                continue
            if "=" in entry_text:
                key, definition = entry_text.split("=", 1)
                add_key(key, definition.strip())
            else:
                add_key(entry_text)

        for key in FREE_MODE_ASSISTANT_GLOSSARY:
            if _contains_keyword(searchable_text, key):
                add_key(key)

        for alias, key in FREE_MODE_ASSISTANT_GLOSSARY_ALIASES.items():
            if _contains_keyword(searchable_text, alias):
                add_key(key)

        rendered: list[str] = []
        for key in ordered_keys:
            definition = explicit_defs.get(key) or FREE_MODE_ASSISTANT_GLOSSARY.get(key, "")
            if definition:
                rendered.append(f"- {key}: {definition}")
        return tuple(rendered)


class FreeModeAssistantOverlay:
    """Mały, pasywny HUD kontekstu dla trybu swobodnego."""

    def __init__(self, root: tk.Misc, on_close=None):
        self.root = root
        self._on_close = on_close
        self._visible = False
        self._context = FreeModeAssistantContext()
        self._palette = {}
        self._manual_position: tuple[int, int] | None = None
        self._drag_anchor: tuple[int, int, int, int] | None = None
        self._last_notebook: tk.Misc | None = None
        self._last_info_panel: tk.Misc | None = None
        self._glossary_expanded = False
        self._body_text = ""
        self._glossary_lines: tuple[str, ...] = ()
        self._glossary_text = ""
        self._last_width = 390
        self._last_height = 180
        self._estimated_content_height = 130
        self._content_window_id = None
        self._scrollregion_after_id = None
        self._notice_after_id = None

        self.frame = tk.Frame(
            root,
            bd=0,
            highlightthickness=1,
            padx=12,
            pady=10,
        )
        self.header_frame = tk.Frame(
            self.frame,
            bd=0,
            highlightthickness=0,
            cursor="fleur",
        )
        self.header_frame.pack(fill=tk.X)

        self.title_lbl = tk.Label(
            self.header_frame,
            text="Asystent kontekstu",
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9, "bold"),
            cursor="fleur",
        )
        self.title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.drag_hint_lbl = tk.Label(
            self.header_frame,
            text="GRAB",
            anchor="center",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 8, "bold"),
            cursor="fleur",
            padx=8,
        )
        self.drag_hint_lbl.pack(side=tk.LEFT, padx=(8, 4))

        self.close_btn = tk.Button(
            self.header_frame,
            text="X",
            command=self._request_close,
            width=2,
            bd=0,
            relief=tk.FLAT,
            highlightthickness=0,
            cursor="hand2",
            padx=4,
            pady=0,
            font=("Segoe UI", 8, "bold"),
        )
        self.close_btn.pack(side=tk.RIGHT)

        self.body_shell = tk.Frame(self.frame, bd=0, highlightthickness=0)
        self.body_shell.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self.body_canvas = tk.Canvas(self.body_shell, bd=0, highlightthickness=0, takefocus=0)
        self.body_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.body_scrollbar = WebSlimScrollbar(
            self.body_shell,
            orient=tk.VERTICAL,
            command=self.body_canvas.yview,
            auto_hide=True,
            thickness=8,
        )
        self.body_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        self.body_canvas.configure(yscrollcommand=self.body_scrollbar.set)

        self.content_frame = tk.Frame(self.body_canvas, bd=0, highlightthickness=0)
        self._content_window_id = self.body_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        self.notice_lbl = tk.Label(
            self.content_frame, text="", anchor="w", justify=tk.LEFT,
            bd=0, highlightthickness=0, font=("Segoe UI", 9), wraplength=330,
        )
        self.body_lbl = tk.Label(
            self.content_frame,
            text="",
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 9),
            wraplength=330,
        )
        self.body_lbl.pack(fill=tk.X, pady=(6, 0))

        self.glossary_toggle_btn = tk.Button(
            self.content_frame,
            text="Słownik pojęć (0) pokaż",
            command=self._toggle_glossary,
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            relief=tk.FLAT,
            highlightthickness=0,
            cursor="hand2",
            padx=0,
            pady=3,
            font=("Segoe UI", 8, "bold"),
        )
        self.glossary_toggle_btn.pack(fill=tk.X, pady=(8, 0))

        self.glossary_lbl = tk.Label(
            self.content_frame,
            text="",
            anchor="w",
            justify=tk.LEFT,
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 8),
            wraplength=330,
        )
        self.glossary_lbl.pack(fill=tk.X, pady=(4, 0))
        self.glossary_lbl.pack_forget()

        self.content_frame.bind("<Configure>", self._queue_scrollregion_refresh, add="+")
        self.body_canvas.bind("<Configure>", self._on_body_canvas_configure, add="+")
        for widget in (
            self.body_shell,
            self.body_canvas,
            self.content_frame,
            self.body_lbl,
            self.notice_lbl,
            self.glossary_toggle_btn,
            self.glossary_lbl,
            self.body_scrollbar,
        ):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                try:
                    widget.bind(sequence, self._on_mousewheel, add="+")
                except Exception:
                    pass

        for widget in (self.frame, self.header_frame, self.title_lbl, self.drag_hint_lbl):
            try:
                widget.bind("<ButtonPress-1>", self._begin_drag, add="+")
                widget.bind("<B1-Motion>", self._drag, add="+")
                widget.bind("<ButtonRelease-1>", self._end_drag, add="+")
            except Exception:
                pass

    def _request_close(self) -> str:
        callback = getattr(self, "_on_close", None)
        if callable(callback):
            try:
                callback()
                return "break"
            except Exception:
                pass
        self.hide()
        return "break"

    def refresh_theme(self, palette: dict | None = None) -> None:
        self._palette = dict(palette or self._palette or {})
        palette = self._palette
        panel = palette.get("panel", "#252526")
        panel_alt = palette.get("panel_alt", "#2d2d30")
        border = blend_hex_colors(
            palette.get("success", palette.get("accent", "#4ec9b0")),
            palette.get("panel_border", palette.get("border", "#3c3c3c")),
            0.55,
        )
        bg = blend_hex_colors(panel_alt, panel, 0.44)
        title_fg = palette.get("success", palette.get("accent", "#4ec9b0"))
        body_fg = palette.get("muted", palette.get("fg", "#f3f3f3"))

        themed_widgets = (
            self.frame,
            self.header_frame,
            self.title_lbl,
            self.drag_hint_lbl,
            self.close_btn,
            self.body_shell,
            self.body_canvas,
            self.content_frame,
            self.body_lbl,
            self.notice_lbl,
            self.glossary_toggle_btn,
            self.glossary_lbl,
        )
        for widget in themed_widgets:
            try:
                widget.configure(bg=bg)
            except Exception:
                pass
        try:
            self.frame.configure(highlightbackground=border, highlightcolor=border)
            self.title_lbl.configure(fg=title_fg)
            self.drag_hint_lbl.configure(fg=palette.get("muted", palette.get("fg", "#f3f3f3")))
            close_fg = palette.get("danger", palette.get("error", "#ff6b6b"))
            self.close_btn.configure(
                fg=close_fg,
                activeforeground=close_fg,
                activebackground=blend_hex_colors(close_fg, bg, 0.18),
            )
            self.body_lbl.configure(fg=body_fg)
            self.notice_lbl.configure(fg=title_fg)
            self.glossary_toggle_btn.configure(
                fg=title_fg,
                activeforeground=title_fg,
                activebackground=bg,
            )
            self.glossary_lbl.configure(fg=body_fg)
            self.body_scrollbar.configure_style(
                track_color=bg,
                thumb_color=blend_hex_colors(title_fg, bg, 0.40),
                thumb_hover_color=title_fg,
            )
        except Exception:
            pass

    def update_context(self, context, *, palette: dict | None = None) -> None:
        next_context = FreeModeAssistantContext.from_value(context)
        context_changed = next_context != self._context
        self._context = next_context
        if palette is not None:
            self.refresh_theme(palette)
        if context_changed:
            self._body_text = self._context.render_body()
            self._glossary_lines = self._context.render_glossary_lines()
            self._glossary_text = "\n".join(self._glossary_lines)
            self._estimated_content_height = self._estimate_content_height()
            try:
                self.body_canvas.yview_moveto(0.0)
            except Exception:
                pass
        try:
            if str(self.body_lbl.cget("text") or "") != self._body_text:
                self.body_lbl.configure(text=self._body_text)
        except Exception:
            pass
        self._refresh_glossary_display()
        self._queue_scrollregion_refresh()

    def show_notice(self, message: str, duration_ms: int = 2200) -> None:
        self._clear_notice()
        self.notice_lbl.configure(text=message)
        self.notice_lbl.pack(before=self.body_lbl, fill=tk.X, pady=(4, 0))
        self.body_canvas.yview_moveto(0.0)
        self._notice_after_id = self.frame.after(duration_ms, self._clear_notice)
        self._queue_scrollregion_refresh()

    def _clear_notice(self) -> None:
        if self._notice_after_id is None:
            return
        try:
            self.frame.after_cancel(self._notice_after_id)
            self.notice_lbl.pack_forget()
            self.notice_lbl.configure(text="")
        except tk.TclError:
            pass
        self._notice_after_id = None
        self._queue_scrollregion_refresh()

    def _toggle_glossary(self) -> None:
        self._glossary_expanded = not bool(self._glossary_expanded)
        self._estimated_content_height = self._estimate_content_height()
        self._refresh_glossary_display()
        self.place(
            notebook=self._last_notebook,
            info_panel=self._last_info_panel,
        )

    def _refresh_glossary_display(self) -> None:
        lines = self._glossary_lines
        count = len(lines)
        if count <= 0:
            try:
                self.glossary_toggle_btn.pack_forget()
                self.glossary_lbl.pack_forget()
            except Exception:
                pass
            return

        try:
            if not str(self.glossary_toggle_btn.winfo_manager()):
                self.glossary_toggle_btn.pack(fill=tk.X, pady=(8, 0))
            action = "ukryj" if self._glossary_expanded else "pokaż"
            self.glossary_toggle_btn.configure(text=f"Słownik pojęć ({count}) {action}")
            if self._glossary_expanded and str(self.glossary_lbl.cget("text") or "") != self._glossary_text:
                self.glossary_lbl.configure(text=self._glossary_text)
            if self._glossary_expanded:
                if not str(self.glossary_lbl.winfo_manager()):
                    self.glossary_lbl.pack(fill=tk.X, pady=(4, 0))
            elif str(self.glossary_lbl.winfo_manager()):
                self.glossary_lbl.pack_forget()
        except Exception:
            pass
        self._queue_scrollregion_refresh()

    def _estimate_content_height(self) -> int:
        body_lines = max(1, len(str(getattr(self, "_body_text", "") or "").splitlines()))
        glossary_lines = len(getattr(self, "_glossary_lines", ()) or ()) if self._glossary_expanded else 0
        return max(112, (body_lines * 18) + (glossary_lines * 17) + 42)

    def _on_body_canvas_configure(self, event=None) -> None:
        try:
            width = max(220, int(getattr(event, "width", self.body_canvas.winfo_width()) or 0))
            if self._content_window_id is not None:
                self.body_canvas.itemconfigure(self._content_window_id, width=width)
            wrap = max(220, width - 4)
            if int(float(self.body_lbl.cget("wraplength") or 0)) != wrap:
                self.body_lbl.configure(wraplength=wrap)
            if int(float(self.glossary_lbl.cget("wraplength") or 0)) != wrap:
                self.glossary_lbl.configure(wraplength=wrap)
            self.notice_lbl.configure(wraplength=wrap)
        except Exception:
            pass
        self._queue_scrollregion_refresh()

    def _queue_scrollregion_refresh(self, _event=None) -> None:
        if getattr(self, "_scrollregion_after_id", None):
            return
        try:
            self._scrollregion_after_id = self.root.after_idle(self._refresh_scrollregion)
        except Exception:
            self._scrollregion_after_id = None

    def _refresh_scrollregion(self) -> None:
        self._scrollregion_after_id = None
        try:
            self.body_canvas.configure(scrollregion=self.body_canvas.bbox("all"))
        except Exception:
            pass

    def _on_mousewheel(self, event) -> str:
        try:
            if getattr(event, "num", None) == 4:
                units = -3
            elif getattr(event, "num", None) == 5:
                units = 3
            else:
                delta = int(getattr(event, "delta", 0) or 0)
                units = -int(delta / 120) if delta else 0
                if units == 0 and delta:
                    units = -1 if delta > 0 else 1
                units *= 3
            if units:
                self.body_canvas.yview_scroll(units, "units")
        except Exception:
            pass
        return "break"

    def show(self) -> None:
        self._visible = True
        self.place()

    def hide(self) -> None:
        self._visible = False
        self._clear_notice()
        try:
            self.frame.place_forget()
        except Exception:
            pass

    def _begin_drag(self, event) -> str:
        try:
            self._drag_anchor = (
                int(event.x_root),
                int(event.y_root),
                int(self.frame.winfo_x()),
                int(self.frame.winfo_y()),
            )
            self.frame.configure(cursor="fleur")
        except Exception:
            self._drag_anchor = None
        return "break"

    def _drag(self, event) -> str:
        if self._drag_anchor is None:
            return "break"
        try:
            start_x, start_y, frame_x, frame_y = self._drag_anchor
            next_x = frame_x + int(event.x_root) - start_x
            next_y = frame_y + int(event.y_root) - start_y
            width = max(280, int(getattr(self, "_last_width", 390) or 390))
            height = max(120, int(getattr(self, "_last_height", 180) or 180))
            next_x, next_y = self._clamp_position(
                next_x,
                next_y,
                width=width,
                height=height,
                info_panel=self._last_info_panel,
            )
            self._manual_position = (next_x, next_y)
            self.frame.place(x=next_x, y=next_y, width=width, height=height)
            self.frame.lift()
        except Exception:
            pass
        return "break"

    def _end_drag(self, _event) -> str:
        self._drag_anchor = None
        try:
            self.frame.configure(cursor="")
        except Exception:
            pass
        return "break"

    def _clamp_position(
        self,
        x: int,
        y: int,
        *,
        width: int,
        height: int,
        info_panel: tk.Misc | None = None,
    ) -> tuple[int, int]:
        try:
            root_w = max(640, int(self.root.winfo_width() or 0))
        except Exception:
            root_w = 1024
        try:
            root_h = max(420, int(self.root.winfo_height() or 0))
        except Exception:
            root_h = 768

        bottom_limit = root_h - 12
        if info_panel is not None:
            try:
                bottom_limit -= int(info_panel.winfo_height() or 0)
            except Exception:
                pass

        min_x = 12
        min_y = 56
        max_x = max(min_x, root_w - width - 12)
        max_y = max(min_y, bottom_limit - height)
        return (
            max(min_x, min(int(x), max_x)),
            max(min_y, min(int(y), max_y)),
        )

    def place(self, *, notebook: tk.Misc | None = None, info_panel: tk.Misc | None = None) -> None:
        if not self._visible or self._context.is_empty():
            self.hide()
            return
        if notebook is not None:
            self._last_notebook = notebook
        if info_panel is not None:
            self._last_info_panel = info_panel

        try:
            root_w = max(640, int(self.root.winfo_width() or 0))
        except Exception:
            root_w = 1024
        try:
            root_h = max(420, int(self.root.winfo_height() or 0))
        except Exception:
            root_h = 768
        info_height = 0
        if info_panel is not None:
            try:
                info_height = max(0, int(info_panel.winfo_height() or 0))
            except Exception:
                info_height = 0

        width = min(390, max(320, int(root_w * 0.30)))
        wrap = max(260, width - 26)
        try:
            self.body_lbl.configure(wraplength=wrap)
            self.glossary_lbl.configure(wraplength=wrap)
            if self._content_window_id is not None:
                self.body_canvas.itemconfigure(self._content_window_id, width=max(220, width - 44))
        except Exception:
            pass

        title_h = 26
        content_req = 130
        try:
            title_h = max(
                22,
                int(self.header_frame.winfo_reqheight() or 0),
                int(self.title_lbl.winfo_reqheight() or title_h),
            )
            content_req = max(
                96,
                int(self.content_frame.winfo_reqheight() or content_req),
                int(getattr(self, "_estimated_content_height", content_req) or content_req),
            )
        except Exception:
            pass
        max_height = max(180, root_h - info_height - 88)
        height = min(max_height, max(154, title_h + content_req + 34))
        self._last_width = width
        self._last_height = height

        if self._manual_position is not None:
            x, y = self._manual_position
        else:
            x = max(12, root_w - width - 18)
            y = 78
            if notebook is not None:
                try:
                    y = max(56, int(notebook.winfo_y()) + 34)
                except Exception:
                    y = 78

        x, y = self._clamp_position(x, y, width=width, height=height, info_panel=info_panel)
        if self._manual_position is not None:
            self._manual_position = (x, y)

        try:
            self.frame.place(x=x, y=y, width=width, height=height)
            self.frame.lift()
            self._queue_scrollregion_refresh()
        except Exception:
            pass
