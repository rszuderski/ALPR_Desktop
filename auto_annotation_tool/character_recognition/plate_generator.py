#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator wyciętych, prostowanych i ZNORMALIZOWANYCH ROZMIAROWO tablic.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import numpy as np

from ..config import logger, CV2_AVAILABLE, cv2
from ..rectification import PlateRectifier
from ..data_models import ImageAnnotation, Detection


class PlateGenerator:
    """Generuje wycięte i znormalizowane rozmiarowo tablice z detections."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.metadata_file = self.output_dir / "metadata.json"
        
        self.images_dir.mkdir(exist_ok=True)
        
        self.metadata = {}  
        self.plate_counter = 0

    @staticmethod
    def _safe_int(value, default=None):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_json_list(value):
        if isinstance(value, list):
            return [str(item or "").strip().upper() for item in value if str(item or "").strip()]
        raw_value = str(value or "").strip()
        if not raw_value:
            return []
        try:
            parsed = json.loads(raw_value)
        except Exception:
            parsed = []
        if not isinstance(parsed, list):
            return []
        return [str(item or "").strip().upper() for item in parsed if str(item or "").strip()]

    def generate_from_annotations(self,
                                 source_image_path: Path,
                                 annotation: ImageAnnotation,
                                 rectify: bool = True,
                                 do_deskew: bool = True,
                                 enhance_contrast: bool = True,
                                 interpolation: str = "lanczos4",
                                 sharpen: float = 0.0,
                                 progress_callback: Optional[callable] = None
                                 ) -> List[Tuple[Path, Detection]]:
        """
        Generuje wycięte tablice z anotacji pojazdu (Inteligentna normalizacja).
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV niedostępny")
            return []
        
        try:
            source_image = cv2.imread(str(source_image_path))
            if source_image is None:
                logger.error(f"Nie można załadować: {source_image_path}")
                return []
            
            results = []
            plates = annotation.plates
            
            interp_map = {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "cubic": cv2.INTER_CUBIC,
                "lanczos4": cv2.INTER_LANCZOS4
            }
            cv2_interp = interp_map.get(interpolation.lower(), cv2.INTER_LANCZOS4)

            # Sztywne wymiary docelowe dla datasetu (Normalizacja!)
            STANDARD_WIDTH = 256
            STANDARD_HEIGHT_LONG = 64     # Tablice podłużne (jednorzędowe)
            STANDARD_HEIGHT_SQUARE = 128  # Tablice kwadratowe (dwurzędowe)

            for plate_idx, plate_detection in enumerate(plates):
                if progress_callback:
                    progress_callback(plate_idx + 1, len(plates), source_image_path.name)
                
                x1, y1, x2, y2 = plate_detection.bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(source_image.shape[1], x2)
                y2 = min(source_image.shape[0], y2)
                
                if y2 <= y1 or x2 <= x1:
                    continue
                    
                plate_image = source_image[y1:y2, x1:x2].copy()
                if plate_image.size == 0:
                    continue
                
                is_square = False

                if rectify and plate_detection.polygon:
                    try:
                        local_polygon = [
                            (p[0] - x1, p[1] - y1) 
                            for p in plate_detection.polygon[:4]
                        ]
                        
                        w_est, h_est = PlateRectifier.polygon_wh_px(local_polygon)
                        # Ważne: o typie tablicy decydujemy na podstawie NATURALNYCH
                        # proporcji polygonu, a nie na podstawie wysokości podbitej
                        # do minimalnego rozmiaru roboczego. Wcześniej clamp `h>=40`
                        # sztucznie obniżał aspect ratio małych, ale długich tablic
                        # i błędnie klasyfikował je jako "square".
                        native_h = max(1.0, float(h_est))
                        aspect_ratio = float(w_est) / native_h
                        h = max(40, int(round(native_h)))
                        
                        # --- 1. PROSTOWANIE Z ZACHOWANIEM NATURALNYCH PROPORCJI ---
                        if aspect_ratio < 2.5:
                            w = int(h * 1.5)  # Dwurzędowa
                            is_square = True
                        else:
                            w = int(h * 4.56) # Jednorzędowa
                            is_square = False
                        
                        plate_image = PlateRectifier.rectify(
                            plate_image,
                            local_polygon,
                            out_w_px=w,
                            out_h_px=h,
                            interpolation=interpolation, 
                            enhance_contrast=enhance_contrast,
                            do_deskew=do_deskew,
                            manual_angle=0.0,
                            sharpen=sharpen
                        )
                        
                        # --- 2. ZACHOWAJ DOKLADNY ZAKRES ANOTACJI ---
                        # Historycznie po prostowaniu ucinalismy lewy/prawy margines,
                        # zeby usunac niebieski wyroznik kraju i skrajne artefakty.
                        # Przy obecnym schemacie anotacji polygon obejmuje juz tylko
                        # wlasciwa tablice, wiec dodatkowe docinanie obcinaloby znaki.
                        # Zachowujemy wiec caly wycinek zwrocony przez rectifier.
                            
                    except Exception as e:
                        logger.debug(f"Błąd prostowania tablicy: {e}")
                        # Fallback jeśli prostowanie zawiedzie (traktujemy jak zwykły wycinek)
                        aspect_ratio = plate_image.shape[1] / plate_image.shape[0]
                        is_square = aspect_ratio < 2.5

                else:
                    # Traktujemy niewyprostowany wycinek jako źródło
                    aspect_ratio = plate_image.shape[1] / plate_image.shape[0]
                    is_square = aspect_ratio < 2.5

                # --- 3. TWARDA NORMALIZACJA ROZMIARU (RESIZE DO STANDARDU) ---
                # Nieważne czy tablica była z 10 czy ze 100 metrów, teraz każda
                # w katalogu wyjściowym będzie miała identyczny, wyraźny rozmiar!
                target_w = STANDARD_WIDTH
                target_h = STANDARD_HEIGHT_SQUARE if is_square else STANDARD_HEIGHT_LONG
                
                plate_image = cv2.resize(plate_image, (target_w, target_h), interpolation=cv2_interp)

                # =========================================================
                # 4. ZAPIS I METADANE
                # =========================================================
                plate_id = f"plate_{self.plate_counter:06d}"
                plate_filename = f"{plate_id}.jpg"
                plate_path = self.images_dir / plate_filename
                
                cv2.imwrite(str(plate_path), plate_image)
                attributes = dict(plate_detection.attributes or {})
                source_expected_text = str(
                    attributes.get("source_expected_text")
                    or attributes.get("expected_text")
                    or ""
                ).strip().upper()
                source_expected_texts = self._safe_json_list(attributes.get("source_expected_texts"))
                source_plate_index = self._safe_int(attributes.get("source_plate_index"), plate_idx)
                source_plate_count = self._safe_int(attributes.get("source_plate_count"), len(plates))
                
                self.metadata[plate_id] = {
                    'source_image': str(source_image_path),
                    'source_image_name': source_image_path.name,
                    'source_bbox': [float(x) for x in plate_detection.bbox],
                    'source_polygon': (
                        [[float(px), float(py)] for px, py in list(plate_detection.polygon or [])[:4]]
                        if plate_detection.polygon else None
                    ),
                    'is_square': bool(is_square),  # Zapisujemy typ, może się przydać do YOLO
                    'source_plate_index': source_plate_index,
                    'source_plate_count': source_plate_count,
                    'source_expected_text': source_expected_text or None,
                    'source_expected_texts': source_expected_texts,
                    'source_expected_text_source': str(
                        attributes.get("source_expected_text_source") or ""
                    ).strip() or None,
                    'ocr_text': str(plate_detection.text) if plate_detection.text else None,
                    'ocr_confidence': float(plate_detection.text_confidence) if plate_detection.text_confidence else 0.0,
                    'detection_confidence': float(plate_detection.confidence),
                    'plate_layout': 'two_row_candidate' if is_square else 'single_row',
                    'layout_row_count': 0 if is_square else 1,
                    'layout_confidence': 0.35 if is_square else 0.55,
                    'layout_source': 'plate_aspect',
                    'plate_attributes': attributes,
                }
                
                plate_detection.plate_id = plate_id
                plate_detection.plate_path = str(plate_path)
                
                results.append((plate_path, plate_detection))
                self.plate_counter += 1
            
            return results
        except Exception as e:
            logger.error(f"Błąd generowania tablic: {e}")
            return []
    
    def save_metadata(self) -> bool:
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Błąd zapisu metadanych: {e}")
            return False
    
    def get_summary(self) -> Dict:
        return {
            'total_plates_generated': self.plate_counter,
            'output_dir': str(self.output_dir),
            'images_count': len(list(self.images_dir.glob('*.jpg'))),
            'metadata_file': str(self.metadata_file),
        }
