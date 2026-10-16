#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Install: python -m pip install PyQt5==5.15.11 opencv-python-headless==4.13.0.92 numpy==2.4.4 Pillow==12.2.0 reportlab==4.4.10
# Optional local U2-Net: python -m pip install -r requirements-ai.txt
# Run: python iraqi_id_printer.py
# Copyright (c) 2026 حسن وضاح (Hasan Wadhah). All rights reserved.
"""مستمسكاتي: offline Iraqi document preparation, A4 layout and actual-size printing."""
from __future__ import annotations

import math
import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas
from PyQt5.QtCore import Qt, QThread, QTimer, QRectF, QSizeF, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget,
)
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog

APP_NAME = "مستمسكاتي"
COPYRIGHT = "© 2026 حسن وضاح — جميع الحقوق محفوظة"
A4_MM = (210.0, 297.0)
PAGE_MARGIN_MM = 10.0
MAX_IMAGE_PIXELS = 40_000_000
PRESETS = {
    "البطاقة الوطنية الموحدة": (8.6, 5.6),
    "بطاقة السكن": (8.6, 5.6),
    "البطاقة التموينية الإلكترونية": (8.6, 5.6),
    "بطاقة الناخب": (8.6, 5.6),
    "كي كارد / ماستر كارد": (8.6, 5.6),
    "هوية الأحوال المدنية القديمة": (6.5, 9.5),
    "شهادة الجنسية العراقية": (13.0, 18.0),
    "جواز السفر العراقي": (12.5, 8.8),
}


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


def read_image(path: str | Path) -> np.ndarray:
    """Decode Unicode paths, normalize EXIF and composite transparent images on white."""
    path = Path(path)
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("حجم الملف يتجاوز 100 ميغابايت.")
    with Image.open(path) as header:
        if header.width * header.height > MAX_IMAGE_PIXELS:
            raise ValueError("الصورة كبيرة جدًا. استخدم صورة لا تتجاوز 40 مليون بكسل.")
        orientation = header.getexif().get(274, 1)
    data = np.fromfile(str(path), dtype=np.uint8)
    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("تعذر قراءة الصورة. استخدم ملف JPG أو PNG أو TIFF أو BMP صالحًا.")
    if decoded.dtype == np.uint16:
        decoded = (decoded / 257).astype(np.uint8)
    if decoded.ndim == 2:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    elif decoded.shape[2] == 4:
        rgba = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
        foreground = Image.fromarray(rgba)
        background = Image.new("RGBA", foreground.size, "white")
        rgb = np.asarray(Image.alpha_composite(background, foreground).convert("RGB"))
    else:
        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    transpose = {
        2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    if orientation in transpose:
        image = image.transpose(transpose[orientation])
    return np.array(image, dtype=np.uint8)


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all():
        raise ValueError("إحداثيات القص غير صالحة.")
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) != 4 or abs(cv2.contourArea(hull)) < 25:
        raise ValueError("اختر أربع زوايا تشكل مستطيلاً واضحًا دون تقاطع.")
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    if cv2.contourArea(points, oriented=True) < 0:
        points = points[[0, 3, 2, 1]]
    return points.astype(np.float32)


def rectangle_points(image: np.ndarray, inset: float = 0.0) -> np.ndarray:
    height, width = image.shape[:2]
    x, y = (width - 1) * inset, (height - 1) * inset
    return np.array([[x, y], [width - 1 - x, y], [width - 1 - x, height - 1 - y],
                     [x, height - 1 - y]], dtype=np.float32)


def local_model_path() -> Path:
    if os.environ.get("U2NET_HOME"):
        return Path(os.environ["U2NET_HOME"]).expanduser() / "u2net.onnx"
    bundled = Path(__file__).resolve().parent / "models" / "u2net.onnx"
    return bundled if bundled.is_file() else Path.home() / ".u2net" / "u2net.onnx"


def mask_quad(mask: np.ndarray) -> np.ndarray | None:
    mask = ((mask > 127) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if not 0.04 * mask.size < area < 0.98 * mask.size:
            continue
        hull = cv2.convexHull(contour)
        for epsilon in (0.015, 0.025, 0.04):
            approx = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True)
            if len(approx) == 4:
                return order_points(approx)
        return order_points(cv2.boxPoints(cv2.minAreaRect(hull)))
    return None


_AI_SESSION = None
_AI_MODEL_KEY = None
U2NET_MD5 = "60024c5c889badc19c04ad937298a77b"  # Published by rembg's U2netSession.


def detect_document(rgb: np.ndarray, target_ratio: float = 8.6 / 5.6, *, use_ai: bool = True) -> tuple[np.ndarray, str]:
    """Never download a model at runtime. Fall back to geometry when unavailable."""
    global _AI_SESSION, _AI_MODEL_KEY
    scale = min(1.0, 1400.0 / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    reason = "النموذج المحلي غير متوفر" if use_ai else "وضع الويب الخفيف"
    if use_ai and local_model_path().is_file():
        try:
            model_stat = local_model_path().stat()
            model_key = (str(local_model_path().resolve()), model_stat.st_size, model_stat.st_mtime_ns)
            if _AI_SESSION is None or _AI_MODEL_KEY != model_key:
                model_file = local_model_path()
                with model_file.open("rb") as model_stream:
                    if hashlib.file_digest(model_stream, "md5").hexdigest() != U2NET_MD5:
                        raise ValueError("فشل التحقق من سلامة النموذج المحلي.")
                from rembg.sessions.u2net import U2netSession
                import onnxruntime as ort

                class LocalSession(U2netSession):
                    @classmethod
                    def download_models(cls, *args, **kwargs):
                        # Explicit local-only model loading; no network/download fallback.
                        return str(model_file)

                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                _AI_SESSION = LocalSession("u2net", options, providers=["CPUExecutionProvider"])
                _AI_MODEL_KEY = model_key
            from rembg import remove
            masked = np.asarray(remove(Image.fromarray(small), session=_AI_SESSION))
            quad = mask_quad(masked[:, :, 3])
            if quad is not None:
                quad /= scale
                quad[:, 0] = np.clip(quad[:, 0], 0, rgb.shape[1] - 1)
                quad[:, 1] = np.clip(quad[:, 1], 0, rgb.shape[0] - 1)
                return order_points(quad), "تم الكشف بالنموذج المحلي — راجع الزوايا قبل الاعتماد"
            reason = "لم يعثر النموذج على حدود واضحة"
        except Exception:
            reason = "تعذر تشغيل النموذج المحلي"
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 60, 60)
    edges = cv2.Canny(filtered, 45, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = small.shape[0] * small.shape[1]
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.04 or area > image_area * 0.96:
            continue
        perimeter = cv2.arcLength(contour, True)
        for epsilon in (0.015, 0.025, 0.04):
            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = order_points(approx)
            width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
            height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
            ratio_error = abs(math.log(max(width / max(height, 1), 0.01) / target_ratio))
            score = area / image_area / (1 + ratio_error)
            candidates.append((score, quad))
            break
    if candidates:
        quad = max(candidates, key=lambda item: item[0])[1] / scale
        quad[:, 0] = np.clip(quad[:, 0], 0, rgb.shape[1] - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, rgb.shape[0] - 1)
        return order_points(quad), f"كشف الحواف • {reason} — راجع الزوايا يدويًا"
    # A rotated rectangle is only a suggestion; never claim reliable ID recognition.
    eligible = [c for c in contours if image_area * 0.04 < cv2.contourArea(c) < image_area * 0.96]
    if eligible:
        quad = cv2.boxPoints(cv2.minAreaRect(max(eligible, key=cv2.contourArea))) / scale
        quad[:, 0] = np.clip(quad[:, 0], 0, rgb.shape[1] - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, rgb.shape[0] - 1)
        try:
            return order_points(quad), "حدود تقريبية فقط — اضبط الزوايا الأربع يدويًا"
        except ValueError:
            pass
    return rectangle_points(rgb, 0.08), "لم تُكتشف حدود موثوقة — حدد الزوايا الأربع يدويًا"


def warp_document(rgb: np.ndarray, points: np.ndarray, ratio: float) -> np.ndarray:
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("نسبة أبعاد المستمسك غير صالحة.")
    ordered = order_points(points)
    # Reject crossed/concave handles instead of silently reinterpreting an invalid selection.
    polygon = np.asarray(points, np.float32).reshape(4, 2)
    if not cv2.isContourConvex(polygon):
        raise ValueError("زوايا القص متقاطعة. رتّب المقابض حول حواف المستمسك.")
    if (ordered[:, 0] < 0).any() or (ordered[:, 0] >= rgb.shape[1]).any() or (ordered[:, 1] < 0).any() or (ordered[:, 1] >= rgb.shape[0]).any():
        raise ValueError("زوايا القص يجب أن تبقى داخل الصورة.")
    width = max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3]))
    height = max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1]))
    output_w = max(width, height * ratio, 64)
    output_h = output_w / ratio
    factor = min(1.0, 6000 / max(output_w, output_h), math.sqrt(25_000_000 / (output_w * output_h)))
    w, h = max(2, round(output_w * factor)), max(2, round(output_h * factor))
    dest = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dest)
    return cv2.warpPerspective(rgb, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def enhance_image(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "color":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        light, a, b = cv2.split(lab)
        light = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(light)
        light = cv2.add(light, 10)
        enhanced = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2RGB)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        return cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)
    if mode == "bw":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 45, 45)
        gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 12)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return rgb.copy()


def deskew_image(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 1800,
                            threshold=50, minLineLength=max(25, rgb.shape[1] // 6), maxLineGap=20)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = (angle + 90) % 180 - 90
            if abs(angle) <= 12:
                angles.append(angle)
    if len(angles) < 3:
        return rgb.copy(), 0.0
    angle = float(np.median(angles))
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)), angle


@dataclass
class Card:
    width_cm: float = 8.6
    height_cm: float = 5.6
    front: Image.Image | None = None
    back: Image.Image | None = None


@dataclass
class Placement:
    image: Image.Image
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    card_index: int
    side: str


def layout_pages(cards: list[Card], horizontal_mm: float = 4, vertical_mm: float = 4) -> list[list[Placement]]:
    if not all(math.isfinite(v) and 0 <= v <= 30 for v in (horizontal_mm, vertical_mm)):
        raise ValueError("التباعد يجب أن يكون بين 0 و30 مم.")
    pages: list[list[Placement]] = [[]]
    y = PAGE_MARGIN_MM
    usable_width = A4_MM[0] - 2 * PAGE_MARGIN_MM
    for index, card in enumerate(cards):
        if card.front is None and card.back is None:
            continue
        width, height = card.width_cm * 10, card.height_cm * 10
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError(f"أبعاد البطاقة {index + 1} غير صالحة.")
        if width > usable_width or height > A4_MM[1] - 2 * PAGE_MARGIN_MM:
            raise ValueError(f"البطاقة {index + 1} أكبر من المساحة المتاحة على A4 بهامش 10 مم. قلّل الأبعاد.")
        sides = [("back", card.back), ("front", card.front)]
        sides = [(side, image) for side, image in sides if image is not None]
        pair_fits = len(sides) == 2 and 2 * width + horizontal_mm <= usable_width + 1e-7
        rows = [sides] if pair_fits else [[item] for item in reversed(sides)]
        for row in rows:
            if y + height > A4_MM[1] - PAGE_MARGIN_MM + 1e-7:
                pages.append([])
                y = PAGE_MARGIN_MM
            row_width = width * len(row) + horizontal_mm * (len(row) - 1)
            x = (A4_MM[0] - row_width) / 2
            for side, image in row:
                pages[-1].append(Placement(image, x, y, width, height, index, side))
                x += width + horizontal_mm
            y += height + vertical_mm
    return pages


def fitted_image(image: Image.Image, width: int, height: int) -> Image.Image:
    # Preserve proportions if the user changes dimensions after cropping.
    return ImageOps.pad(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS,
                        color="white", centering=(0.5, 0.5))


def render_page(placements: list[Placement], dpi: int = 300) -> Image.Image:
    if dpi not in (96, 300, 600):
        raise ValueError("دقة التصدير المدعومة: 300 أو 600 dpi.")
    page = Image.new("RGB", tuple(mm_to_px(mm, dpi) for mm in A4_MM), "white")
    draw = ImageDraw.Draw(page)
    for p in placements:
        x, y = mm_to_px(p.x_mm, dpi), mm_to_px(p.y_mm, dpi)
        width, height = mm_to_px(p.width_mm, dpi), mm_to_px(p.height_mm, dpi)
        page.paste(fitted_image(p.image, width, height), (x, y))
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline="#cccccc", width=1)
    return page


def _atomic_output(path: Path, writer: Callable[[Path], None]) -> None:
    fd, name = tempfile.mkstemp(prefix=".mostamskaty-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_pdf(path: str | Path, pages: list[list[Placement]], dpi: int = 300) -> None:
    if dpi not in (300, 600):
        raise ValueError("اختر 300 أو 600 dpi.")
    path = Path(path)
    unit = 72 / 25.4

    def write(temporary):
        pdf = pdf_canvas.Canvas(str(temporary), pagesize=(210 * unit, 297 * unit), pageCompression=1)
        pdf.setTitle("Mostamskaty - Iraqi Documents")
        pdf.setAuthor("Hasan Wadhah | حسن وضاح")
        for page in pages:
            for p in page:
                image = fitted_image(p.image, mm_to_px(p.width_mm, dpi), mm_to_px(p.height_mm, dpi))
                pdf.drawImage(ImageReader(image), p.x_mm * unit,
                              (297 - p.y_mm - p.height_mm) * unit,
                              width=p.width_mm * unit, height=p.height_mm * unit)
                pdf.setStrokeColorRGB(0.8, 0.8, 0.8)
                pdf.setLineWidth(72 / dpi)
                pdf.rect(p.x_mm * unit, (297 - p.y_mm - p.height_mm) * unit,
                         p.width_mm * unit, p.height_mm * unit)
            pdf.showPage()
        pdf.save()
    _atomic_output(path, write)


def image_export_paths(path: str | Path, count: int) -> list[Path]:
    path = Path(path)
    if count == 1:
        return [path]
    return [path.with_name(f"{path.stem}_{index + 1:02d}{path.suffix}") for index in range(count)]


def export_images(path: str | Path, pages: list[list[Placement]], dpi: int = 300) -> list[Path]:
    if dpi not in (300, 600):
        raise ValueError("اختر 300 أو 600 dpi.")
    paths = image_export_paths(path, len(pages))
    for target, page in zip(paths, pages):
        image = render_page(page, dpi)
        options = {"dpi": (dpi, dpi)}
        if target.suffix.lower() in (".jpg", ".jpeg"):
            options.update(quality=98, subsampling=0)
        _atomic_output(target, lambda temporary: image.save(temporary, **options))
    return paths


def qimage_from_array(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()


def pil_pixmap(image: Image.Image) -> QPixmap:
    return QPixmap.fromImage(qimage_from_array(np.asarray(image.convert("RGB"))))


class Task(QThread):
    """Keep all widget updates on the GUI thread; retain worker until finished."""
    def __init__(self, job: Callable, parent=None):
        super().__init__(parent)
        self.job = job
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.job()
        except Exception as exc:
            self.error = str(exc) or exc.__class__.__name__


class CornerHandle(QGraphicsEllipseItem):
    def __init__(self, canvas, index: int, position: QPointF):
        super().__init__(-9, -9, 18, 18)
        self.canvas = canvas
        self.index = index
        self.setBrush(QColor("#16a765"))
        self.setPen(QPen(QColor("white"), 2))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges |
                      QGraphicsItem.ItemIgnoresTransformations)
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(10)
        self.setPos(position)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            bounds = self.canvas.image_rect
            return QPointF(max(0, min(value.x(), bounds.width() - 1)),
                           max(0, min(value.y(), bounds.height() - 1)))
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.canvas.viewport().update()
            self.canvas.pointsChanged.emit()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, str(self.index + 1))


class CropCanvas(QGraphicsView):
    pointsChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#172337"))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setLayoutDirection(Qt.LeftToRight)
        self.image_rect = QRectF(0, 0, 1, 1)
        self.handles = []
        self.setMinimumSize(420, 300)
        self._zoom = 1.0

    def set_image(self, rgb: np.ndarray, points: np.ndarray | None = None):
        self.handles = []
        self.scene().clear()
        h, w = rgb.shape[:2]
        self.image_rect = QRectF(0, 0, w, h)
        self.scene().addPixmap(QPixmap.fromImage(qimage_from_array(rgb)))
        self.scene().setSceneRect(self.image_rect)
        self.set_points(rectangle_points(rgb) if points is None else points)
        self.fit_image()

    def set_points(self, points: np.ndarray):
        for handle in self.handles:
            self.scene().removeItem(handle)
        self.handles = []
        for index, (x, y) in enumerate(points):
            handle = CornerHandle(self, index, QPointF(float(x), float(y)))
            self.scene().addItem(handle)
            self.handles.append(handle)
        self.viewport().update()

    def points(self) -> np.ndarray:
        return np.array([[h.pos().x(), h.pos().y()] for h in self.handles], dtype=np.float32)

    def fit_image(self):
        self.resetTransform()
        self.fitInView(self.image_rect.adjusted(-18, -18, 18, 18), Qt.KeepAspectRatio)
        self._zoom = 1.0

    def drawForeground(self, painter, rect):
        if len(self.handles) == 4:
            polygon = QPolygonF([handle.pos() for handle in self.handles])
            pen = QPen(QColor("#39d6ed"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QColor(57, 214, 237, 18))
            painter.drawPolygon(polygon)

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        if 0.25 <= self._zoom * factor <= 12:
            self.scale(factor, factor)
            self._zoom *= factor
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom == 1.0:
            self.fit_image()


class EditorDialog(QDialog):
    def __init__(self, path: str, ratio: float, title: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.ratio = ratio
        self.original = None
        self.current = None
        self.result_image = None
        self.cropped = False
        self.filter_base = None
        self.filter_mode = 0
        self.worker = None
        self._updating = False
        self.setWindowTitle(f"تحرير وقص الصورة — {title}")
        self.setLayoutDirection(Qt.RightToLeft)
        self.resize(1100, 850)
        layout = QVBoxLayout(self)
        heading = QLabel("حدد زوايا المستمسك الأربع، ثم اعتمد الصورة")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        hint = QLabel("اسحب المقابض الخضراء • عجلة الفأرة للتكبير • اسحب الصورة للتحريك")
        hint.setObjectName("muted")
        layout.addWidget(hint)
        self.canvas = CropCanvas()
        self.canvas.pointsChanged.connect(self._points_changed)
        layout.addWidget(self.canvas, 1)
        self.toolbar = QWidget()
        grid = QGridLayout(self.toolbar)
        grid.setContentsMargins(0, 4, 0, 4)
        actions = [
            ("تعرف ذكي على الهوية", self.auto_detect), ("قص يدوي", self.manual_crop),
            ("تطبيق القص", self.apply_crop), ("تحسين الألوان", self.enhance),
            ("ضبط الميلان", self.deskew), ("تدوير 90°", self.rotate),
            ("ضبط الحواف", self.border), ("ملاءمة الصورة", self.canvas.fit_image),
            ("استعادة الأصل", self.restore),
        ]
        for index, (label, callback) in enumerate(actions):
            button = QPushButton(label)
            button.clicked.connect(callback)
            grid.addWidget(button, index // 3, index % 3)
        layout.addWidget(self.toolbar)
        self.status = QLabel("جارٍ فتح الصورة…")
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        footer = QHBoxLayout()
        self.confirm_button = QPushButton("اعتماد وتثبيت")
        self.confirm_button.setObjectName("primary")
        self.confirm_button.clicked.connect(self.confirm)
        self.cancel_button = QPushButton("إلغاء")
        self.cancel_button.clicked.connect(self.reject)
        footer.addWidget(self.confirm_button)
        footer.addWidget(self.cancel_button)
        layout.addLayout(footer)
        self._busy(True)
        QTimer.singleShot(0, self.load)

    def _busy(self, busy: bool):
        self.toolbar.setEnabled(not busy and self.current is not None)
        self.canvas.setEnabled(not busy and self.current is not None)
        self.confirm_button.setEnabled(not busy and self.current is not None)
        self.cancel_button.setEnabled(not busy)

    def run_task(self, job: Callable, callback: Callable, message: str):
        if self.worker is not None:
            return
        self.status.setText(message)
        self._busy(True)
        self.worker = Task(job, self)
        worker = self.worker

        def finish():
            self.worker = None
            self._busy(False)
            if worker.error:
                self.status.setText("تعذر إكمال العملية. يمكنك المحاولة مجددًا.")
                QMessageBox.warning(self, "تعذر معالجة الصورة", worker.error)
            else:
                try:
                    callback(worker.result)
                except Exception as exc:
                    QMessageBox.warning(self, "تعذر معالجة الصورة", str(exc))
            self._busy(False)
            worker.deleteLater()
        worker.finished.connect(finish)
        worker.start()

    def load(self):
        def job():
            image = read_image(self.path)
            points, message = detect_document(image, self.ratio)
            return image, points, message

        def loaded(result):
            image, points, message = result
            self.original = image.copy()
            self.set_image(image, cropped=False, points=points)
            self.status.setText(message)
        self.run_task(job, loaded, "جارٍ فتح الصورة والكشف المحلي عن الحواف…")

    def set_image(self, image: np.ndarray, cropped: bool = True, points=None):
        self.current = image
        self._updating = True
        self.canvas.set_image(image, points)
        self._updating = False
        self.cropped = cropped
        self.filter_base = None
        self.filter_mode = 0

    def _points_changed(self):
        if not self._updating:
            self.cropped = False
            self.filter_base = None
            self.filter_mode = 0

    def crop_job(self) -> Callable:
        image, points, cropped, ratio = self.current, self.canvas.points(), self.cropped, self.ratio
        return lambda: image.copy() if cropped else warp_document(image, points, ratio)

    def auto_detect(self):
        image = self.current

        def done(result):
            points, message = result
            self._updating = True
            self.canvas.set_points(points)
            self._updating = False
            self.cropped = False
            self.filter_base = None
            self.status.setText(message)
        self.run_task(lambda: detect_document(image, self.ratio), done, "جارٍ الكشف عن الحواف…")

    def manual_crop(self):
        self.canvas.set_points(rectangle_points(self.current, 0.08))
        self.cropped = False
        self.status.setText("اسحب الزوايا حتى تطابق حدود المستمسك، ثم اضغط تطبيق القص.")

    def apply_crop(self):
        job = self.crop_job()

        def done(image):
            self.set_image(image)
            self.status.setText("تم تصحيح المنظور وقص المستمسك.")
        self.run_task(job, done, "جارٍ تصحيح المنظور…")

    def enhance(self):
        mode = (self.filter_mode + 1) % 3
        base = self.filter_base
        job = self.crop_job()

        def process():
            original = job() if base is None else base
            return original, enhance_image(original, {0: "original", 1: "color", 2: "bw"}[mode])

        def done(result):
            original, image = result
            self.set_image(image)
            self.filter_base = original
            self.filter_mode = mode
            self.status.setText({0: "ألوان ما قبل التحسين", 1: "ماسح ملون — تباين محلي وحدّة معتدلة",
                                 2: "ماسح ليزر — أبيض وأسود"}[mode])
        self.run_task(process, done, "جارٍ تحسين وضوح المستمسك…")

    def transform_image(self, transform: Callable, message: str):
        job = self.crop_job()

        def done(image):
            self.set_image(image)
            self.status.setText(message)
        self.run_task(lambda: transform(job()), done, "جارٍ معالجة الصورة…")

    def deskew(self):
        job = self.crop_job()

        def done(result):
            image, angle = result
            self.set_image(image)
            self.status.setText(f"تم تصحيح الميلان بمقدار {angle:.2f}°" if abs(angle) > 0.05
                                else "لم يُكتشف ميلان نصي واضح؛ يمكنك ضبط الزوايا يدويًا.")
        self.run_task(lambda: deskew_image(job()), done, "جارٍ قياس ميلان النص…")

    def rotate(self):
        image, points, cropped = self.current, self.canvas.points(), self.cropped
        # Rotate the original selection too, allowing sideways photos to be fixed before cropping.
        rotated_points = np.column_stack((image.shape[0] - 1 - points[:, 1], points[:, 0]))

        def done(rotated):
            self.set_image(rotated, cropped=cropped,
                           points=None if cropped else order_points(rotated_points))
            self.status.setText("تم التدوير 90° مع عقارب الساعة.")
        self.run_task(lambda: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), done, "جارٍ تدوير الصورة…")

    def border(self):
        def add_border(rgb):
            image = rgb.copy()
            cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), (204, 204, 204), 1)
            return image
        self.transform_image(add_border, "تم ضبط الحواف بخط رمادي رفيع داخل حدود المستمسك.")

    def restore(self):
        self.set_image(self.original.copy(), cropped=False, points=rectangle_points(self.original, 0.08))
        self.status.setText("تمت استعادة الصورة الأصلية. اضبط الزوايا أو شغّل التعرف الذكي.")

    def confirm(self):
        def done(image):
            self.result_image = Image.fromarray(image).copy()
            self.accept()
        self.run_task(self.crop_job(), done, "جارٍ اعتماد الصورة…")

    def reject(self):
        if self.worker is None:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)


class PagePreview(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#dfe7ef"))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumSize(380, 450)
        self.setLayoutDirection(Qt.LeftToRight)

    def set_page(self, page: Image.Image):
        self.scene().clear()
        item = self.scene().addPixmap(pil_pixmap(page))
        self.scene().setSceneRect(item.boundingRect().adjusted(-16, -16, 16, 16))
        self.fit_page()

    def fit_page(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_page()


class SpacingControl(QWidget):
    valueChanged = pyqtSignal()

    def __init__(self, title: str, initial: float = 4):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 300)
        self.slider.setValue(round(initial * 10))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0, 30)
        self.spin.setDecimals(1)
        self.spin.setSingleStep(0.5)
        self.spin.setSuffix(" mm")
        self.spin.setValue(initial)
        self.spin.setMaximumWidth(105)
        self.slider.valueChanged.connect(lambda v: self.spin.setValue(v / 10))
        self.spin.valueChanged.connect(self._changed)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        layout.addLayout(row)

    def _changed(self, value):
        self.slider.setValue(round(value * 10))
        self.valueChanged.emit()

    def value(self):
        return self.spin.value()


class CardSlot(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, index: int, card: Card, parent=None):
        super().__init__(f"البطاقة {index + 1}", parent)
        self.index, self.card = index, card
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.front_button = QPushButton("رفع الوجه")
        self.back_button = QPushButton("رفع الظهر")
        self.front_button.clicked.connect(lambda: self.upload("front"))
        self.back_button.clicked.connect(lambda: self.upload("back"))
        row.addWidget(self.front_button)
        row.addWidget(self.back_button)
        layout.addLayout(row)
        thumbnails = QHBoxLayout()
        self.thumbnails = {}
        for side, caption in (("front", "الوجه"), ("back", "الظهر")):
            label = QLabel(caption)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedHeight(62)
            label.setObjectName("thumbnail")
            thumbnails.addWidget(label)
            self.thumbnails[side] = label
        layout.addLayout(thumbnails)
        dimensions = QGridLayout()
        self.width = QDoubleSpinBox()
        self.height = QDoubleSpinBox()
        for label, widget, initial, column in (
            ("العرض (cm)", self.width, card.width_cm, 0),
            ("الارتفاع (cm)", self.height, card.height_cm, 1),
        ):
            widget.setRange(1.0, 30.0)
            widget.setDecimals(2)
            widget.setSingleStep(0.1)
            widget.setValue(initial)
            dimensions.addWidget(QLabel(label), 0, column)
            dimensions.addWidget(widget, 1, column)
            widget.valueChanged.connect(self.dimensions_changed)
        layout.addLayout(dimensions)
        clear_button = QPushButton("حذف البطاقة")
        clear_button.setObjectName("danger")
        clear_button.clicked.connect(self.clear)
        layout.addWidget(clear_button)

    def dimensions_changed(self):
        self.card.width_cm, self.card.height_cm = self.width.value(), self.height.value()
        self.changed.emit()

    def set_dimensions(self, width: float, height: float):
        self.width.blockSignals(True)
        self.height.blockSignals(True)
        self.width.setValue(width)
        self.height.setValue(height)
        self.width.blockSignals(False)
        self.height.blockSignals(False)
        self.dimensions_changed()

    def refresh(self):
        for side, button, caption in (("front", self.front_button, "الوجه"),
                                       ("back", self.back_button, "الظهر")):
            image = getattr(self.card, side)
            button.setText(f"✓ تغيير {caption}" if image is not None else f"رفع {caption}")
            thumbnail = self.thumbnails[side]
            if image is not None:
                thumbnail.setPixmap(pil_pixmap(image.copy().resize((150, 90), Image.Resampling.LANCZOS))
                                    .scaled(110, 54, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                thumbnail.clear()
                thumbnail.setText(caption)
        self.changed.emit()

    def upload(self, side: str):
        path, _ = QFileDialog.getOpenFileName(self, "اختر صورة المستمسك", "",
                                              "صور (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;كل الملفات (*)")
        if not path:
            return
        dialog = EditorDialog(path, self.width.value() / self.height.value(),
                              f"البطاقة {self.index + 1} • {'الوجه' if side == 'front' else 'الظهر'}", self)
        if dialog.exec_() == QDialog.Accepted and dialog.result_image is not None:
            setattr(self.card, side, dialog.result_image)
            self.refresh()
        dialog.deleteLater()

    def clear(self):
        if self.card.front is None and self.card.back is None:
            return
        answer = QMessageBox.question(self, "حذف البطاقة", "هل تريد إزالة صور الوجه والظهر من هذه البطاقة؟",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.card.front = self.card.back = None
            self.refresh()


def paint_print_pages(printer: QPrinter, pages: list[list[Placement]], dpi: int = 300,
                      first: int = 1, last: int | None = None) -> None:
    """Use millimetre device coordinates, never fit-to-page scaling."""
    if dpi not in (300, 600):
        raise ValueError("اختر 300 أو 600 dpi.")
    last = len(pages) if last is None else last
    if not 1 <= first <= last <= len(pages):
        raise ValueError("نطاق صفحات الطباعة غير صالح.")
    paper = printer.paperRect(QPrinter.Millimeter)
    if abs(paper.width() - 210) > 1 or abs(paper.height() - 297) > 1:
        raise ValueError("اختر ورق A4 بالاتجاه العمودي في إعدادات الطابعة.")
    printer.setFullPage(True)
    painter = QPainter()
    if not painter.begin(printer):
        raise RuntimeError("تعذر الاتصال بالطابعة. تحقق من تعريفها وإعداداتها.")
    try:
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.scale(printer.resolution() / 25.4, printer.resolution() / 25.4)
        for index in range(first - 1, last):
            if index > first - 1 and not printer.newPage():
                raise RuntimeError("تعذر بدء صفحة جديدة في الطابعة.")
            for p in pages[index]:
                image = fitted_image(p.image, mm_to_px(p.width_mm, dpi), mm_to_px(p.height_mm, dpi))
                rect = QRectF(p.x_mm, p.y_mm, p.width_mm, p.height_mm)
                painter.drawImage(rect, qimage_from_array(np.asarray(image)))
                pen = QPen(QColor("#cccccc"))
                pen.setWidthF(25.4 / dpi)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
    finally:
        painter.end()
    if printer.printerState() in (QPrinter.Error, QPrinter.Aborted):
        raise RuntimeError("توقفت مهمة الطباعة أو رفضتها الطابعة.")


STYLE = """
QWidget { font-family: 'Segoe UI', 'Noto Sans Arabic', 'DejaVu Sans'; font-size: 13px; color: #233750; }
QMainWindow, QDialog { background: #f2f5fa; }
QLabel#brand { color: #14395b; font-size: 29px; font-weight: bold; }
QLabel#muted { color: #62768b; font-size: 12px; }
QLabel#sectionTitle { color: #14395b; font-size: 16px; font-weight: bold; }
QLabel#badge { color: #08765c; background: #e3f4ed; border-radius: 8px; padding: 8px 14px; }
QLabel#notice { color: #6e561b; background: #fff5dc; border-radius: 8px; padding: 9px; }
QGroupBox { background: white; border: 1px solid #dce4ed; border-radius: 10px; margin-top: 14px; padding: 16px 12px 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top right; right: 14px; padding: 0 7px; }
QPushButton { background: white; border: 1px solid #cdd9e6; border-radius: 7px; padding: 9px 12px; min-height: 18px; }
QPushButton:hover { background: #eef5ff; border-color: #78a5d3; }
QPushButton:pressed { background: #dfedff; }
QPushButton:disabled { color: #99a6b5; background: #e9eef4; border-color: #dce4ed; }
QPushButton#primary { background: #1766ad; color: white; border-color: #1766ad; font-weight: bold; }
QPushButton#primary:hover { background: #10548f; }
QPushButton#primary:disabled { background: #a7bed5; border-color: #a7bed5; color: #edf2f7; }
QPushButton#danger { color: #b43e43; background: #fff5f5; border-color: #f0d8d9; }
QDoubleSpinBox, QComboBox { background: white; border: 1px solid #cdd9e6; border-radius: 6px; padding: 7px; min-height: 21px; }
QComboBox::drop-down { width: 24px; border: none; }
QLabel#thumbnail { background: #f5f8fc; border: 1px dashed #cbd7e5; border-radius: 6px; color: #899aab; }
QScrollArea { border: none; background: transparent; }
QSplitter::handle { background: #f2f5fa; width: 14px; }
QSlider::groove:horizontal { height: 5px; border-radius: 2px; background: #d7e2ef; }
QSlider::handle:horizontal { width: 15px; margin: -5px 0; border-radius: 7px; background: #1766ad; }
QToolTip { background: #14395b; color: white; border: none; padding: 6px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cards = [Card() for _ in range(4)]
        self.pages = [[]]
        self.current_page = 0
        self.layout_error = None
        self.worker = None
        self.setWindowTitle(f"{APP_NAME} | إعداد وطباعة المستمسكات العراقية — حسن وضاح")
        self.setLayoutDirection(Qt.RightToLeft)
        self.resize(1300, 920)
        self.setMinimumSize(1020, 720)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 14, 22, 12)
        root.setSpacing(12)
        header = QHBoxLayout()
        brand_column = QVBoxLayout()
        brand = QLabel(APP_NAME)
        brand.setObjectName("brand")
        brand_column.addWidget(brand)
        subtitle = QLabel("قصّ وتحسين وترتيب المستمسكات العراقية للطباعة")
        subtitle.setObjectName("muted")
        brand_column.addWidget(subtitle)
        header.addLayout(brand_column)
        header.addStretch()
        badge = QLabel("● معالجة محلية • دون إنترنت")
        badge.setObjectName("badge")
        badge.setMaximumHeight(40)
        header.addWidget(badge)
        about = QPushButton("عن البرنامج")
        about.clicked.connect(self.about)
        header.addWidget(about)
        root.addLayout(header)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setLayoutDirection(Qt.LeftToRight)
        splitter.setChildrenCollapsible(False)
        controls = QWidget()
        controls.setLayoutDirection(Qt.RightToLeft)
        controls.setMinimumWidth(350)
        controls.setMaximumWidth(470)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 5, 0)
        group = QGroupBox("إعداد الصفحة")
        settings = QVBoxLayout(group)
        settings.addWidget(QLabel("النمط — يطبّق المقاسات على البطاقات الأربع"))
        self.preset = QComboBox()
        self.preset.addItems(PRESETS)
        settings.addWidget(self.preset)
        self.horizontal = SpacingControl("التباعد الأفقي • الجوانب")
        self.vertical = SpacingControl("التباعد العمودي • أعلى وأسفل")
        settings.addWidget(self.horizontal)
        settings.addWidget(self.vertical)
        controls_layout.addWidget(group)
        self.slots = []
        for index, card in enumerate(self.cards):
            slot = CardSlot(index, card)
            slot.changed.connect(self.schedule_refresh)
            controls_layout.addWidget(slot)
            self.slots.append(slot)
        controls_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(370)
        splitter.addWidget(scroll)
        preview_panel = QWidget()
        preview_panel.setLayoutDirection(Qt.RightToLeft)
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(6, 0, 0, 0)
        navigation = QHBoxLayout()
        self.previous = QPushButton("السابق")
        self.next = QPushButton("التالي")
        self.page_label = QLabel()
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setObjectName("sectionTitle")
        self.previous.clicked.connect(lambda: self.navigate(-1))
        self.next.clicked.connect(lambda: self.navigate(1))
        navigation.addWidget(self.previous)
        navigation.addWidget(self.page_label, 1)
        navigation.addWidget(self.next)
        preview_layout.addLayout(navigation)
        self.preview = PagePreview()
        preview_layout.addWidget(self.preview, 1)
        self.empty_label = QLabel("ابدأ برفع الوجه أو الظهر من البطاقة 1 — الصور تبقى على جهازك")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setObjectName("muted")
        self.empty_label.setWordWrap(True)
        preview_layout.addWidget(self.empty_label)
        caption = QLabel("معاينة A4 • 210 × 297 mm • جودة التصدير والطباعة 300/600 dpi")
        caption.setAlignment(Qt.AlignCenter)
        caption.setObjectName("muted")
        preview_layout.addWidget(caption)
        self.notice = QLabel()
        self.notice.setObjectName("notice")
        self.notice.setWordWrap(True)
        preview_layout.addWidget(self.notice)
        splitter.addWidget(preview_panel)
        splitter.setSizes([400, 820])
        root.addWidget(splitter, 1)
        actions = QHBoxLayout()
        self.print_button = QPushButton("طباعة مباشرة")
        self.pdf_button = QPushButton("تصدير PDF")
        self.image_button = QPushButton("تصدير صورة")
        self.action_buttons = [self.print_button, self.pdf_button, self.image_button]
        for button in self.action_buttons:
            button.setObjectName("primary")
            actions.addWidget(button)
        self.print_button.clicked.connect(self.print_documents)
        self.pdf_button.clicked.connect(self.save_pdf)
        self.image_button.clicked.connect(self.save_images)
        self.image_format = QComboBox()
        self.image_format.addItems(["PNG", "JPG"])
        self.image_format.setToolTip("صيغة تصدير الصور")
        actions.addWidget(self.image_format)
        actions.addStretch()
        actions.addWidget(QLabel("الدقة"))
        self.dpi = QComboBox()
        self.dpi.addItems(["300 dpi", "600 dpi"])
        actions.addWidget(self.dpi)
        root.addLayout(actions)
        footer = QLabel(f"تطوير: حسن وضاح  |  {COPYRIGHT}")
        footer.setObjectName("muted")
        footer.setAlignment(Qt.AlignCenter)
        root.addWidget(footer)
        self.statusBar().showMessage("جاهز • لا تُرفع الصور إلى أي خدمة خارجية")
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self.refresh_preview)
        self.preset.currentTextChanged.connect(self.apply_preset)
        self.horizontal.valueChanged.connect(self.schedule_refresh)
        self.vertical.valueChanged.connect(self.schedule_refresh)
        self.refresh_preview()

    def schedule_refresh(self):
        self.refresh_timer.start()

    def apply_preset(self, name: str):
        width, height = PRESETS[name]
        for slot in self.slots:
            slot.set_dimensions(width, height)
        self.schedule_refresh()

    def active(self):
        return any(card.front is not None or card.back is not None for card in self.cards)

    def refresh_preview(self):
        try:
            pages = layout_pages(self.cards, self.horizontal.value(), self.vertical.value())
            self.layout_error = None
            self.pages = pages
            self.current_page = min(self.current_page, len(pages) - 1)
            self.preview.set_page(render_page(pages[self.current_page], 96))
            self.notice.setText("الوجهان الكبيران يُرتّبان عموديًا دون تصغير. المقاسات بالسنتيمتر؛ هامش الصفحة 10 مم.")
            self.statusBar().showMessage(f"{len(pages)} صفحة • المقاس الفعلي 100% • الوجه يمينًا والظهر يسارًا")
        except ValueError as exc:
            self.layout_error = str(exc)
            self.pages = [[]]
            self.current_page = 0
            self.preview.set_page(render_page([], 96))
            self.notice.setText(str(exc))
            self.statusBar().showMessage("راجع أبعاد المستمسكات قبل التصدير أو الطباعة")
        self.empty_label.setVisible(not self.active())
        self.page_label.setText(f"الصفحة {self.current_page + 1} / {len(self.pages)} — A4")
        self.previous.setEnabled(self.current_page > 0)
        self.next.setEnabled(self.current_page < len(self.pages) - 1)
        for button in self.action_buttons:
            button.setEnabled(self.active() and self.layout_error is None and self.worker is None)

    def navigate(self, delta: int):
        self.current_page = max(0, min(self.current_page + delta, len(self.pages) - 1))
        self.refresh_preview()

    def ready_pages(self) -> list[list[Placement]] | None:
        self.refresh_timer.stop()
        self.refresh_preview()
        if self.layout_error:
            QMessageBox.warning(self, "أبعاد غير مناسبة", self.layout_error)
            return None
        if not self.active():
            QMessageBox.information(self, "لا توجد صور", "ارفع صورة وجه أو ظهر واحدة على الأقل.")
            return None
        return self.pages

    def selected_dpi(self) -> int:
        return 600 if self.dpi.currentIndex() == 1 else 300

    def run_export(self, job: Callable, success: str):
        self.centralWidget().setEnabled(False)
        self.statusBar().showMessage("جارٍ التصدير…")
        self.worker = Task(job, self)
        worker = self.worker

        def finished():
            self.worker = None
            self.centralWidget().setEnabled(True)
            self.refresh_preview()
            if worker.error:
                QMessageBox.warning(self, "تعذر التصدير", worker.error)
            else:
                self.statusBar().showMessage(success)
                QMessageBox.information(self, "تم التصدير", success)
            worker.deleteLater()
        worker.finished.connect(finished)
        worker.start()

    def save_pdf(self):
        pages = self.ready_pages()
        if pages is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "تصدير PDF — جميع الصفحات", "مستمسكاتي.pdf", "PDF (*.pdf)")
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
                if Path(path).exists() and not self.confirm_overwrite():
                    return
            dpi = self.selected_dpi()
            self.run_export(lambda: export_pdf(path, pages, dpi), f"تم حفظ {len(pages)} صفحة PDF بالمقاسات الفعلية.")

    def save_images(self):
        pages = self.ready_pages()
        if pages is None:
            return
        extension = self.image_format.currentText().lower()
        path, _ = QFileDialog.getSaveFileName(self, "تصدير صور A4 — جميع الصفحات", f"مستمسكاتي.{extension}",
                                              f"{extension.upper()} (*.{extension})")
        if not path:
            return
        if not path.lower().endswith(f".{extension}"):
            path += f".{extension}"
            if len(pages) == 1 and Path(path).exists() and not self.confirm_overwrite():
                return
        paths = image_export_paths(path, len(pages))
        if len(paths) > 1 and any(target.exists() for target in paths):
            answer = QMessageBox.question(self, "استبدال الملفات", "بعض صور الصفحات موجودة. هل تريد استبدالها؟",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        dpi = self.selected_dpi()
        self.run_export(lambda: export_images(path, pages, dpi),
                        f"تم حفظ {len(pages)} صورة بدقة {dpi} dpi. تُرقّم أسماء الملفات عند تعدد الصفحات.")

    def confirm_overwrite(self):
        return QMessageBox.question(self, "استبدال الملف", "الملف موجود. هل تريد استبداله؟",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def print_documents(self):
        pages = self.ready_pages()
        if pages is None:
            return
        printer = QPrinter(QPrinter.HighResolution)
        printer.setResolution(self.selected_dpi())
        printer.setPageSize(QPrinter.A4)
        printer.setOrientation(QPrinter.Portrait)
        printer.setFullPage(True)
        printer.setDocName("Mostamskaty - Iraqi Documents")
        dialog = QPrintDialog(printer, self)
        dialog.setWindowTitle("طباعة A4 بالمقاس الفعلي — 100%")
        dialog.setMinMax(1, len(pages))
        dialog.setOption(QPrintDialog.PrintPageRange, True)
        if dialog.exec_() != QDialog.Accepted:
            return
        try:
            first = printer.fromPage() or 1
            last = printer.toPage() or len(pages)
            paint_print_pages(printer, pages, self.selected_dpi(), first, last)
            self.statusBar().showMessage("تم إرسال المستمسكات إلى نظام الطباعة بالمقاس الفعلي.")
        except Exception as exc:
            QMessageBox.warning(self, "تعذر الطباعة", str(exc))

    def about(self):
        QMessageBox.about(self, "عن مستمسكاتي",
                          f"<h2>مستمسكاتي</h2><p>إعداد وطباعة المستمسكات العراقية محليًا.</p>"
                          f"<p>المطور: <b>حسن وضاح</b></p><p>{COPYRIGHT}</p>"
                          "<p>كشف الحواف تقريبي؛ راجع القص والمقاسات قبل الاعتماد.</p>"
                          "<p>عند طباعة PDF اختر الحجم الفعلي / 100% وألغِ الملاءمة التلقائية.</p>"
                          "<p>لا يحفظ البرنامج الصور الأصلية أو سجل المستمسكات تلقائيًا.</p>")

    def closeEvent(self, event):
        if self.worker is not None:
            event.ignore()
            return
        if self.active():
            answer = QMessageBox.question(self, "إغلاق مستمسكاتي",
                                           "إغلاق البرنامج سيزيل الصور من الجلسة. هل صدّرت ما تحتاجه وتريد الإغلاق؟",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Hasan Wadhah")
    app.setLayoutDirection(Qt.RightToLeft)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
