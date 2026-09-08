# ALPR Desktop

Aplikacja Python/Tkinter do anotacji obrazów, przygotowywania datasetów, treningu modeli
oraz analizy i importu raportów mobilnych ALPR.

## Uruchomienie w Windows / VS Code

Użyj Pythona 3.12 x64 z komponentem Tcl/Tk. Otwórz cały katalog projektu w VS Code.
W terminalu, w katalogu zawierającym `main.py`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

W VS Code wybierz **Python: Select Interpreter** → `.venv\Scripts\python.exe`,
a następnie otwórz nowy terminal. Po aktywacji środowiska uruchom:

```powershell
python main.py
```

Możesz również użyć F5 (konfiguracja jest dołączona) albo uruchomić interpreter bez aktywowania środowiska:

```powershell
.\.venv\Scripts\python.exe main.py
```

Samo pobranie repozytorium nie instaluje bibliotek — instalacja z `requirements.txt` jest
jednorazowym krokiem przygotowania środowiska. Bootstrap przy starcie wykrywa brakujące zależności.
Zależności nie narzucają wersji CUDA z poprzedniego komputera; konfiguracja GPU jest opcjonalna.

## Dane i modele

Repozytorium zawiera kod, testy, dokumentację i konfigurację edytora. Nie zawiera bibliotek,
modeli, zdjęć, datasetów, wyników, Workspace.
Pusty `Workspace/` wraz ze strukturą roboczą tworzy się automatycznie przy pierwszym uruchomieniu.
Ustawienia użytkownika aplikacja zapisuje w `~/.auto_annotation_tool/session.json`.
Na komputerze z wcześniejszą instalacją są one współdzielone; ten plik nie jest częścią kopii.

## Opcjonalny eksport mobilny i testy

Pakiety do konwersji modeli instaluje się osobno, gdy potrzebny jest eksport mobilny:

```powershell
python -m pip install -r requirements-mobile-export.txt
```

Zależności testowe:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_mobile_mt_final_hardening.py tests/test_mobile_mt_blind_review.py tests/test_mobile_human_review.py tests/test_mobile_report_full_rows.py -q
```

Testy związane z konwersją modeli mogą wymagać opcjonalnych pakietów eksportu.


## Nowe repozytorium

Katalog jest gotowy do inicjalizacji niezależnego repozytorium (`git init`).
Dołączony `.gitignore` pomija środowiska Pythona, Workspace, modele i artefakty pracy.





R.S.
