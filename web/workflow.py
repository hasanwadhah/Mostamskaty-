"""Session-local browser workflow, sharing the desktop imaging and A4 engines."""
from __future__ import annotations

import base64
import io
import math
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps

from iraqi_id_printer import (
    Card, detect_document, deskew_image, enhance_image, export_images, export_pdf,
    layout_pages, order_points, rectangle_points, warp_document,
)

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_EDIT_EDGE = 2200


def decode_upload(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("اختر صورة صالحة بحجم لا يتجاوز 12 ميغابايت.")
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > 40_000_000 or min(source.size) < 16:
            raise ValueError("الصورة يجب أن تكون بين 16 بكسل و40 مليون بكسل.")
        image = ImageOps.exif_transpose(source).convert("RGBA")
        image.thumbnail((MAX_EDIT_EDGE, MAX_EDIT_EDGE), Image.Resampling.LANCZOS)
        white = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(white, image).convert("RGB")
        return np.array(image)


@dataclass
class WebEditor:
    original: np.ndarray
    current: np.ndarray
    points: np.ndarray
    ratio: float
    card_index: int
    side: str
    message: str = ""
    cropped: bool = False
    revision: str = field(default_factory=lambda: uuid.uuid4().hex)
    identity: str = field(default_factory=lambda: uuid.uuid4().hex)
    filter_base: np.ndarray | None = None
    filter_mode: int = 0

    @classmethod
    def from_upload(cls, data: bytes, ratio: float, card_index: int, side: str):
        image = decode_upload(data)
        points, message = detect_document(image, ratio, use_ai=False)
        return cls(image.copy(), image, points, ratio, card_index, side, message)

    @classmethod
    def from_card(cls, image: Image.Image, ratio: float, card_index: int, side: str):
        rgb = np.array(image.convert("RGB"))
        return cls(rgb.copy(), rgb, rectangle_points(rgb), ratio, card_index, side,
                   "تحرير الصورة المعتمدة. لاستعادة صورة الهاتف، أعد رفع ملفها.", cropped=True)

    def update_points(self, value: dict | None):
        if not value or value.get("revision") != self.revision:
            return
        points = np.asarray(value.get("points"), dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("إحداثيات الزوايا غير صالحة.")
        h, w = self.current.shape[:2]
        if (points < 0).any() or (points[:, 0] > w - 1).any() or (points[:, 1] > h - 1).any():
            raise ValueError("الزوايا يجب أن تبقى داخل الصورة.")
        if not np.allclose(points, self.points, atol=0.01):
            self.points = points
            self.cropped = False
            self.filter_base = None
            self.filter_mode = 0

    def set_current(self, rgb: np.ndarray, cropped: bool = True, points=None):
        self.current = rgb
        self.points = rectangle_points(rgb) if points is None else points
        self.cropped = cropped
        self.filter_base = None
        self.filter_mode = 0
        self.revision = uuid.uuid4().hex

    def cropped_image(self):
        return self.current.copy() if self.cropped else warp_document(self.current, self.points, self.ratio)

    def apply(self, action: str):
        if action == "detect":
            points, self.message = detect_document(self.current, self.ratio, use_ai=False)
            self.points = points
            self.cropped = False
            self.filter_base = None
            self.revision = uuid.uuid4().hex
        elif action == "manual":
            self.points = rectangle_points(self.current, 0.08)
            self.cropped = False
            self.filter_base = None
            self.revision = uuid.uuid4().hex
            self.message = "اسحب المقابض الأربعة إلى حدود المستمسك."
        elif action == "crop":
            self.set_current(self.cropped_image())
            self.message = "تم القص وتصحيح المنظور."
        elif action == "enhance":
            mode = (self.filter_mode + 1) % 3
            original = self.cropped_image() if self.filter_base is None else self.filter_base
            self.set_current(enhance_image(original, {0: "original", 1: "color", 2: "bw"}[mode]))
            self.filter_base, self.filter_mode = original, mode
            self.message = {0: "ألوان ما قبل التحسين", 1: "ماسح ملون", 2: "ماسح أبيض وأسود"}[mode]
        elif action == "deskew":
            rgb, angle = deskew_image(self.cropped_image())
            self.set_current(rgb)
            self.message = f"تصحيح الميلان: {angle:.2f}°" if abs(angle) > .05 else "لم يُكتشف ميلان واضح."
        elif action == "rotate":
            points = np.column_stack((self.current.shape[0] - 1 - self.points[:, 1], self.points[:, 0]))
            self.set_current(cv2.rotate(self.current, cv2.ROTATE_90_CLOCKWISE),
                             cropped=self.cropped, points=order_points(points))
            self.message = "تم التدوير 90°."
        elif action == "border":
            rgb = self.cropped_image()
            cv2.rectangle(rgb, (0, 0), (rgb.shape[1] - 1, rgb.shape[0] - 1), (204, 204, 204), 1)
            self.set_current(rgb)
            self.message = "تم ضبط الحواف."
        elif action == "reset":
            self.set_current(self.original.copy(), cropped=False, points=rectangle_points(self.original, .08))
            self.message = "تمت استعادة الصورة الأصلية."
        else:
            raise ValueError("أمر تحرير غير معروف.")

    def accept(self) -> Image.Image:
        return Image.fromarray(self.cropped_image()).copy()

    def image_url(self) -> str:
        output = io.BytesIO()
        Image.fromarray(self.current).save(output, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def prepare_export(cards: list[Card], horizontal: float, vertical: float,
                   dpi: int, format_name: str) -> tuple[bytes, str, str]:
    if not any(card.front is not None or card.back is not None for card in cards):
        raise ValueError("ارفع واعتمد صورة واحدة على الأقل قبل التصدير.")
    pages = layout_pages(cards, horizontal, vertical)
    if format_name not in ("PDF", "PNG", "JPG"):
        raise ValueError("صيغة تصدير غير مدعومة.")
    with tempfile.TemporaryDirectory(prefix="mostamskaty-export-") as folder:
        root = Path(folder)
        if format_name == "PDF":
            target = root / "mostamskaty.pdf"
            export_pdf(target, pages, dpi)
            return target.read_bytes(), "application/pdf", target.name
        extension = format_name.lower()
        paths = export_images(root / f"mostamskaty.{extension}", pages, dpi)
        if len(paths) == 1:
            return paths[0].read_bytes(), "image/png" if extension == "png" else "image/jpeg", paths[0].name
        output = io.BytesIO()
        with ZipFile(output, "w", ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, path.name)
        return output.getvalue(), "application/zip", "mostamskaty-pages.zip"
