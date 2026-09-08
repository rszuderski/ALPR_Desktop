#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zarządca archiwów ZIP (Import/Export do CVAT).
"""

import zipfile
import shutil
from pathlib import Path
from typing import Iterable, Tuple

from ..config import logger, CONFIG


class CVATZipManager:
    """Automatyzuje zrzut obrazów i pliku annotations.xml do jednej paczki ZIP."""

    @staticmethod
    def extract_cvat_export(zip_path: Path, output_workspace_dir: Path, subfolder_name: str) -> Tuple[bool, str, Path]:
        """
        Rozpakowuje ZIP pobrany z CVAT prosto do logicznego folderu.
        
        Args:
            zip_path: ścieżka do pobranego .zip z CVAT
            output_workspace_dir: Folder w którym utworzy się projekt (np. DIR_4_DATASETS)
            subfolder_name: Nazwa docelowa folderu np. "CVAT_Export_Paczka1"
            
        Returns:
            (Success, Komunikat, Ścieżka do rozpakowanego folderu)
        """
        target_dir = output_workspace_dir / subfolder_name
        
        if not zip_path.exists() or zip_path.suffix.lower() != '.zip':
            return False, "Nieprawidłowy plik ZIP lub nie istnieje", target_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Rozpakowywanie paczki: {zip_path.name} do {target_dir.name}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)

            xml_file = target_dir / "annotations.xml"
            if not xml_file.exists():
                return False, "Błąd: W pliku ZIP brakuje 'annotations.xml'. Czy to na pewno CVAT for Images 1.1?", target_dir

            return True, f"Pomyślnie rozpakowano zbiór.", target_dir

        except Exception as e:
            logger.error(f"Błąd rozpakowywania: {e}")
            return False, f"Błąd weryfikacji paczki ZIP: {e}", target_dir

    @staticmethod
    def create_cvat_import_zip(
        xml_path: Path,
        images_dir: Path,
        output_zip_path: Path,
        image_names_allowlist: Iterable[str] | None = None,
    ) -> Tuple[bool, str]:
        """
        Pakuje wskazany annotations.xml oraz wszystkie zdjęcia w nim użyte 
        w elegancką paczkę .zip, gotową do Drag & Drop w oknie przeglądarki z CVAT.
        
        Args:
            xml_path: Ścieżka do stworzonego annotations.xml
            images_dir: Folder skąd wziąć zdjęcia pojazdów/tablic
            output_zip_path: Miejsce zapisu gotowej paczki .zip
            image_names_allowlist: Opcjonalna lista nazw obrazów, które mają wejść do paczki
        """
        if not xml_path.exists():
            return False, "Brak pliku annotations.xml do spakowania!"
            
        try:
            output_zip_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Pakowanie zdjęć i XML dla CVAT: {output_zip_path.name}")
            allowed_names = {
                str(name or "").strip().lower()
                for name in (image_names_allowlist or [])
                if str(name or "").strip()
            }
            
            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                # Plik XML musi leżeć w głównym korzeniu pnia ZIP, a nie w folderach
                zip_ref.write(xml_path, arcname="annotations.xml")
                
                # Obrazy muszą leżeć w podfolderze 'images/', tak wymaga specyfikacja CVAT
                if images_dir and images_dir.exists():
                    added_images = 0
                    for img_file in images_dir.iterdir():
                        if img_file.suffix.lower() in CONFIG.IMAGE_EXTENSIONS:
                            if allowed_names and img_file.name.lower() not in allowed_names:
                                continue
                            zip_ref.write(img_file, arcname=f"images/{img_file.name}")
                            added_images += 1
                            
                    logger.debug(f"Spakowano {added_images} obrazów.")

            return True, f"Gotowa paczka do wgrania w przeglądarce: {output_zip_path.name}"

        except Exception as e:
            logger.error(f"Krytyczny błąd budowania archiwum ZIP: {e}")
            return False, str(e)
