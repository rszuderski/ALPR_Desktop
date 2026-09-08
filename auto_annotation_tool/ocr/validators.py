#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walidacja formatów tablic rejestracyjnych (PL, EU, itd).
"""

import re
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass


class LicensePlateFormat(Enum):
    """Obsługiwane formaty tablic."""
    POLISH = "PL"
    EU = "EU"
    UNKNOWN = "UNKNOWN"


@dataclass
class PlateValidationResult:
    """Wynik walidacji tablicy."""
    is_valid: bool
    format: LicensePlateFormat
    text: str
    normalized: str  # Znormalizowany format (bez spacji)
    confidence: float
    details: str


class LicensePlateValidator:
    """Walidator formatów tablic rejestracyjnych."""
    
    # Wzorce dla różnych formatów
    PATTERNS = {
        # Polska: ABC 1234 (3 litery + 4 cyfry)
        LicensePlateFormat.POLISH: r'^[A-Z]{3}\s?\d{4}$',
        
        # EU standard: ABC 1234 (różne warianty)
        LicensePlateFormat.EU: r'^[A-Z]{2,3}\s?\d{3,4}$',
    }
    
    # Polskie województwa (pierwszy element numeru rejestracyjnego)
    POLISH_PROVINCES = {
        'A': 'Dolnośląskie', 'B': 'Kujawsko-Pomorskie', 'C': 'Warmińsko-Mazurskie',
        'D': 'Łódzkie', 'E': 'Małopolskie', 'F': 'Świętokrzyskie',
        'G': 'Lubelskie', 'H': 'Podlaskie', 'I': 'Podkarpackie',
        'J': 'Silesia', 'K': 'Opolskie', 'L': 'Greater Poland',
        'M': 'Masovia', 'N': 'Lublin Voivodeship', 'O': 'Pomerania',
        'P': 'Śląskie', 'R': 'Zachodniopomorskie', 'S': 'West Pomerania',
        'T': 'Warmian-Masurian', 'U': 'Subcarpathian', 'V': 'Greater Poland',
        'W': 'Masovia', 'X': 'Greater Poland', 'Y': 'Lublin',
        'Z': 'Silesia',
    }
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Czyści tekst z OCR."""
        # Usuń znaki specjalne poza literami i cyframi
        text = re.sub(r'[^A-Z0-9\s]', '', text.upper())
        # Usuń zbędne spacje
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    @classmethod
    def validate(cls, ocr_text: str) -> PlateValidationResult:
        """
        Waliduje tekst tablicy i określa format.
        
        Args:
            ocr_text: Tekst z OCR
            
        Returns:
            PlateValidationResult z informacjami o walidacji
        """
        cleaned = cls._clean_text(ocr_text)
        normalized = cleaned.replace(' ', '')
        
        # Próbuj polskie
        if re.match(cls.PATTERNS[LicensePlateFormat.POLISH], cleaned):
            return PlateValidationResult(
                is_valid=True,
                format=LicensePlateFormat.POLISH,
                text=cleaned,
                normalized=normalized,
                confidence=0.95,
                details=f"Polski format: {cleaned}"
            )
        
        # Próbuj EU
        if re.match(cls.PATTERNS[LicensePlateFormat.EU], cleaned):
            return PlateValidationResult(
                is_valid=True,
                format=LicensePlateFormat.EU,
                text=cleaned,
                normalized=normalized,
                confidence=0.85,
                details=f"Format EU: {cleaned}"
            )
        
        # Jeśli nie spełnia żadnego formatu
        return PlateValidationResult(
            is_valid=False,
            format=LicensePlateFormat.UNKNOWN,
            text=cleaned,
            normalized=normalized,
            confidence=0.0,
            details=f"Format nieznany: {cleaned} (próbuj ręcznej korekty)"
        )
    
    @classmethod
    def get_province(cls, plate_text: str) -> Optional[str]:
        """
        Zwraca województwo z polskiej tablicy.
        
        Args:
            plate_text: Tekst tablicy (np. "ABC 1234")
            
        Returns:
            Nazwa województwa lub None
        """
        plate_text = plate_text.upper().replace(' ', '')
        if len(plate_text) >= 1:
            first_letter = plate_text[0]
            return cls.POLISH_PROVINCES.get(first_letter)
        return None