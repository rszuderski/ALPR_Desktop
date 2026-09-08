#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator raportów.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional

from ..config import CONFIG, logger
from ..data_models import ImageAnnotation, AnnotationReport


class ReportGenerator:
    """Generator raportów z auto-anotacji."""
    
    @staticmethod
    def generate_text_report(report: AnnotationReport,
                              output_path: Optional[Path] = None) -> str:
        """
        Generuje tekstowy raport.
        
        Args:
            report: Raport z anotacji
            output_path: Opcjonalna ścieżka do zapisu
            
        Returns:
            Treść raportu
        """
        content = report.to_text()
        
        # Dodaj nagłówek
        header = f"""
================================================================================
                    RAPORT AUTO-ANOTACJI
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    {CONFIG.APP_NAME} v{CONFIG.VERSION}
================================================================================
"""
        content = header + content
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Zapisano raport: {output_path}")
        
        return content
    
    @staticmethod
    def generate_csv_report(annotations: List[ImageAnnotation],
                            output_path: Path) -> bool:
        """
        Generuje raport CSV.
        
        Args:
            annotations: Lista anotacji
            output_path: Ścieżka wyjściowa
            
        Returns:
            True jeśli sukces
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Nagłówek
                f.write("filename,status,num_vehicles,num_plates,message\n")
                
                for ann in annotations:
                    status = ann.status.value
                    msg = ann.status_message.replace(',', ';').replace('\n', ' ')
                    f.write(f"{ann.filename},{status},{ann.num_vehicles},{ann.num_plates},{msg}\n")
            
            logger.info(f"Zapisano CSV: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd zapisu CSV: {e}")
            return False
    
    @staticmethod
    def generate_failed_images_list(report: AnnotationReport,
                                     output_path: Path) -> bool:
        """
        Generuje listę obrazów z błędami/brakami.
        
        Args:
            report: Raport z anotacji
            output_path: Ścieżka wyjściowa
            
        Returns:
            True jeśli sukces
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Lista obrazów wymagających ręcznej anotacji\n")
                f.write(f"# Wygenerowano: {datetime.now().isoformat()}\n\n")
                
                if report.no_vehicle_images:
                    f.write("# === BRAK WYKRYTEGO POJAZDU ===\n")
                    for img in report.no_vehicle_images:
                        f.write(f"{img}\n")
                    f.write("\n")
                
                if report.no_plate_images:
                    f.write("# === BRAK WYKRYTEJ TABLICY ===\n")
                    for img in report.no_plate_images:
                        f.write(f"{img}\n")
                    f.write("\n")
                
                if report.partial_plate_images:
                    f.write("# === TABLICA CZĘŚCIOWO WIDOCZNA ===\n")
                    for img in report.partial_plate_images:
                        f.write(f"{img}\n")
                    f.write("\n")
                
                if report.error_images:
                    f.write("# === BŁĘDY ===\n")
                    for img in report.error_images:
                        f.write(f"{img}\n")
            
            logger.info(f"Zapisano listę: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd: {e}")
            return False