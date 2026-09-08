#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modele danych aplikacji.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


class AnnotationStatus(Enum):
    """Status anotacji dla obrazu."""
    SUCCESS = "success"
    NO_VEHICLE = "no_vehicle"
    NO_PLATE = "no_plate"
    PARTIAL_PLATE = "partial_plate"
    MULTIPLE_VEHICLES = "multiple_vehicles"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class Detection:
    """Pojedyncza detekcja obiektu."""
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: Optional[List[Tuple[float, float, float]]] = None
    polygon: Optional[List[Tuple[float, float]]] = None
    
    text: Optional[str] = None  # Rozpoznany tekst (ABC 1234)
    text_confidence: float = 0.0  # Pewność OCR (0.0-1.0)
    attributes: Dict[str, str] = field(default_factory=dict)  # format, is_valid, itp
    
    def is_inside(self, other_bbox: Tuple[float, float, float, float], 
                  threshold: float = 0.95) -> bool:
        """Sprawdza czy ta detekcja jest wewnątrz innego bbox."""
        x1, y1, x2, y2 = self.bbox
        ox1, oy1, ox2, oy2 = other_bbox
        
        # Część wspólna
        inter_x1 = max(x1, ox1)
        inter_y1 = max(y1, oy1)
        inter_x2 = min(x2, ox2)
        inter_y2 = min(y2, oy2)
        
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return False
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        self_area = (x2 - x1) * (y2 - y1)
        
        if self_area == 0:
            return False
        
        return (inter_area / self_area) >= threshold
    
    def get_area(self) -> float:
        """Oblicza powierzchnię bbox."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def has_ocr_text(self) -> bool:
        """Sprawdza czy detekcja ma rozpoznany tekst."""
        return self.text is not None and len(self.text) > 0
    
    def get_plate_format(self) -> Optional[str]:
        """Zwraca format tablicy (PL, EU, UNKNOWN)."""
        return self.attributes.get('format', None)


@dataclass
class ImageAnnotation:
    """Anotacje dla pojedynczego obrazu."""
    filename: str
    width: int
    height: int
    detections: List[Detection] = field(default_factory=list)
    status: AnnotationStatus = AnnotationStatus.SUCCESS
    status_message: str = ""
    
    @property
    def vehicles(self) -> List[Detection]:
        return [d for d in self.detections if d.label.lower() == "vehicle"]
    
    @property
    def plates(self) -> List[Detection]:
        return [d for d in self.detections if d.label.lower() == "plate"]
    
    @property
    def num_vehicles(self) -> int:
        return len(self.vehicles)
    
    @property
    def num_plates(self) -> int:
        return len(self.plates)
    
    @property
    def plates_with_ocr(self) -> List[Detection]:
        """Zwraca tablice, które mają rozpoznany tekst."""
        return [p for p in self.plates if p.has_ocr_text()]
    
    @property
    def num_plates_with_ocr(self) -> int:
        return len(self.plates_with_ocr)
    
    @property
    def ocr_success_rate(self) -> float:
        """Procent tablic z pomyślnym OCR."""
        if self.num_plates == 0:
            return 0.0
        return (self.num_plates_with_ocr / self.num_plates) * 100
    
    @property
    def is_successful(self) -> bool:
        return self.status == AnnotationStatus.SUCCESS


@dataclass
class AnnotationReport:
    """Raport z auto-anotacji."""
    total_images: int = 0
    successful: int = 0
    no_vehicle: int = 0
    no_plate: int = 0
    partial_plate: int = 0
    errors: int = 0
    skipped: int = 0
    
    # Liczniki detekcji
    total_vehicles: int = 0
    total_plates: int = 0
    
    total_plates_with_ocr: int = 0
    plates_with_valid_format: int = 0
    
    # Listy obrazów
    successful_images: List[str] = field(default_factory=list)
    no_vehicle_images: List[str] = field(default_factory=list)
    no_plate_images: List[str] = field(default_factory=list)
    partial_plate_images: List[str] = field(default_factory=list)
    error_images: List[str] = field(default_factory=list)
    skipped_images: List[str] = field(default_factory=list)
    
    # Szczegóły błędów
    error_details: Dict[str, str] = field(default_factory=dict)
    
    def add_result(self, annotation: 'ImageAnnotation'):
        """Dodaje wynik dla obrazu."""
        self.total_images += 1
        self.total_vehicles += annotation.num_vehicles
        self.total_plates += annotation.num_plates
        
        self.total_plates_with_ocr += annotation.num_plates_with_ocr
        
        # Policz tablice z poprawnym formatem
        for plate in annotation.plates_with_ocr:
            if plate.attributes.get('is_valid') == 'True':
                self.plates_with_valid_format += 1
        
        status = annotation.status
        filename = annotation.filename
        
        if status == AnnotationStatus.SUCCESS:
            self.successful += 1
            self.successful_images.append(filename)
        elif status == AnnotationStatus.NO_VEHICLE:
            self.no_vehicle += 1
            self.no_vehicle_images.append(filename)
        elif status == AnnotationStatus.NO_PLATE:
            self.no_plate += 1
            self.no_plate_images.append(filename)
        elif status == AnnotationStatus.PARTIAL_PLATE:
            self.partial_plate += 1
            self.partial_plate_images.append(filename)
        elif status == AnnotationStatus.SKIPPED:
            self.skipped += 1
            self.skipped_images.append(filename)
        else:
            self.errors += 1
            self.error_images.append(filename)
            if annotation.status_message:
                self.error_details[filename] = annotation.status_message
    
    @property
    def success_rate(self) -> float:
        if self.total_images == 0:
            return 0.0
        return (self.successful / self.total_images) * 100
    
    @property
    def ocr_success_rate(self) -> float:
        """Procent tablic z rozpoznanym tekstem."""
        if self.total_plates == 0:
            return 0.0
        return (self.total_plates_with_ocr / self.total_plates) * 100
    
    @property
    def ocr_format_validity_rate(self) -> float:
        """Procent tablic ze zwalidowanym formatem."""
        if self.total_plates_with_ocr == 0:
            return 0.0
        return (self.plates_with_valid_format / self.total_plates_with_ocr) * 100
    
    def to_text(self) -> str:
        """Generuje tekstowy raport."""
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           RAPORT AUTO-ANOTACJI                              ║
╠══════════════════════════════════════════════════════════════════════════════╣

  PODSUMOWANIE OBRAZÓW
  ─────────────────────────
  Obrazów ogółem:              {self.total_images:5d}
  Udane anotacje:              {self.successful:5d}  ({self.success_rate:.1f}%)
  Przerwane:                   {self.skipped:5d}
  Brak pojazdu:                {self.no_vehicle:5d}
  Brak tablicy:                {self.no_plate:5d}
  Tablica częściowa:           {self.partial_plate:5d}
  Błędy:                       {self.errors:5d}

  STATYSTYKA DETEKCJI
  ─────────────────────────
  Pojazdów wykrytych:          {self.total_vehicles:5d}
  Tablic wykrytych:            {self.total_plates:5d}

  STATYSTYKA OCR
  ─────────────────────────
  Tablic z tekstem (OCR):      {self.total_plates_with_ocr:5d}  ({self.ocr_success_rate:.1f}%)
  Tablic ze zwal. formatem:    {self.plates_with_valid_format:5d}  ({self.ocr_format_validity_rate:.1f}%)

╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        if self.no_vehicle_images:
            report += "\nOBRAZY BEZ WYKRYTEGO POJAZDU:\n"
            report += "─" * 50 + "\n"
            for img in self.no_vehicle_images[:20]:
                report += f"  • {img}\n"
            if len(self.no_vehicle_images) > 20:
                report += f"  ... i {len(self.no_vehicle_images) - 20} więcej\n"
        
        if self.no_plate_images:
            report += "\nOBRAZY BEZ WYKRYTEJ TABLICY:\n"
            report += "─" * 50 + "\n"
            for img in self.no_plate_images[:20]:
                report += f"  • {img}\n"
            if len(self.no_plate_images) > 20:
                report += f"  ... i {len(self.no_plate_images) - 20} więcej\n"
        
        if self.partial_plate_images:
            report += "\nOBRAZY Z CZĘŚCIOWO WIDOCZNĄ TABLICĄ:\n"
            report += "─" * 50 + "\n"
            for img in self.partial_plate_images[:20]:
                report += f"  • {img}\n"
            if len(self.partial_plate_images) > 20:
                report += f"  ... i {len(self.partial_plate_images) - 20} więcej\n"
        
        if self.skipped_images:
            report += "\nOBRAZY PRZERWANE:\n"
            report += "─" * 50 + "\n"
            for img in self.skipped_images[:10]:
                report += f"  • {img}\n"
            if len(self.skipped_images) > 10:
                report += f"  ... i {len(self.skipped_images) - 10} więcej\n"
        
        if self.error_images:
            report += "\nOBRAZY Z BŁĘDAMI:\n"
            report += "─" * 50 + "\n"
            for img in self.error_images[:10]:
                error_msg = self.error_details.get(img, "Nieznany błąd")
                report += f"  • {img}: {error_msg}\n"
        
        return report
