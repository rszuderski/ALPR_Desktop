"""User-local picker preferences, independent of reports and scientific data."""
from pathlib import Path
import os

from ..config import SESSION, logger

SECTION = 'mobile_report_import'
ENABLED = 'remember_folder'
FOLDER = 'last_successful_folder'


def remember_enabled():
    return SESSION is None or SESSION.get(SECTION, ENABLED, True) is not False


def set_remember_enabled(enabled):
    """Persist the switch immediately, even if the picker is later cancelled."""
    if SESSION is None:
        return
    try:
        SESSION.set(SECTION, ENABLED, bool(enabled))
        if not enabled:
            SESSION.data[SECTION].pop(FOLDER, None)
        SESSION.save_session()
    except Exception:
        logger.exception('Nie udało się zapisać preferencji folderu importu')


def readable_directory(value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    try:
        folder = Path(value).expanduser().resolve()
        # Opening the iterator checks directory access, including empty folders.
        with os.scandir(folder):
            pass
        return folder
    except (OSError, ValueError, RuntimeError):
        return None


def initial_directory(default):
    """A stale preference is never an import error and is not overwritten."""
    if SESSION is not None and remember_enabled():
        remembered = readable_directory(SESSION.get(SECTION, FOLDER, ''))
        if remembered is not None:
            return remembered
    return readable_directory(default) or Path.home()


def remember_successful_import(paths, *, folder=None):
    """Call only after the report store commits a non-cancelled import.

    A picker batch normally has one parent. If called with several parents,
    the last successfully imported source in the existing order wins.
    """
    if SESSION is None or not paths or not remember_enabled():
        return
    try:
        folder = Path(folder).expanduser().resolve() if folder is not None else Path(paths[-1]).expanduser().resolve().parent
        SESSION.set(SECTION, ENABLED, True)
        SESSION.set(SECTION, FOLDER, str(folder))
        SESSION.save_session()
    except Exception:
        # A preference failure must not relabel a successful report import.
        logger.exception('Nie udało się zapamiętać folderu udanego importu')
