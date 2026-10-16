import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/mostamskaty-test-cache')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/mostamskaty-numba')

import numpy as np
import pytest
from PIL import Image
from PyQt5.QtCore import QPointF, Qt, QEvent
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrinter
from pypdf import PdfReader

import iraqi_id_printer as m


@pytest.fixture(scope='session')
def app():
    instance = QApplication.instance() or QApplication([])
    instance.setStyleSheet(m.STYLE)
    yield instance


def wait_for(app, predicate, timeout=15):
    until = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        QTest.qWait(10)
        if time.monotonic() > until:
            pytest.fail('GUI worker did not finish')
    app.processEvents()


def test_window_slots_spacing_presets_and_pagination(app, monkeypatch):
    window = m.MainWindow()
    window.show()
    app.processEvents()
    assert window.layoutDirection() == Qt.RightToLeft
    assert len(window.slots) == 4
    assert not window.pdf_button.isEnabled()
    window.horizontal.slider.setValue(75)
    assert window.horizontal.spin.value() == 7.5
    window.vertical.spin.setValue(6.5)
    assert window.vertical.slider.value() == 65
    for card in window.cards:
        card.front = Image.new('RGB', (130, 180), 'red')
        card.back = Image.new('RGB', (130, 180), 'blue')
    window.preset.setCurrentText('شهادة الجنسية العراقية')
    window.refresh_preview()
    assert len(window.pages) == 8
    assert window.pdf_button.isEnabled()
    QTest.mouseClick(window.next, Qt.LeftButton)
    assert window.current_page == 1
    window.slots[0].width.setValue(25)
    window.refresh_preview()
    assert window.layout_error is not None
    assert not window.pdf_button.isEnabled()
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.Yes)
    QTest.mouseClick(window.slots[0].findChildren(m.QPushButton)[2], Qt.LeftButton)
    assert window.cards[0].front is None and window.cards[0].back is None
    window.close()


def test_crop_handles_drag_and_clamp(app):
    canvas = m.CropCanvas()
    canvas.resize(650, 450)
    canvas.set_image(np.full((300, 500, 3), 200, np.uint8),
                     np.array([[50, 50], [450, 50], [450, 250], [50, 250]], np.float32))
    canvas.show()
    app.processEvents()
    handle = canvas.handles[0]
    start = canvas.mapFromScene(handle.pos())
    finish = canvas.mapFromScene(QPointF(85, 75))
    QTest.mousePress(canvas.viewport(), Qt.LeftButton, pos=start)
    # Offscreen QTest.mouseMove omits held-button state; send a real drag event.
    move = QMouseEvent(QEvent.MouseMove, QPointF(finish),
                       QPointF(canvas.viewport().mapToGlobal(finish)),
                       Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(canvas.viewport(), move)
    QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, pos=finish)
    app.processEvents()
    assert handle.pos().x() == pytest.approx(85, abs=3)
    assert handle.pos().y() == pytest.approx(75, abs=3)
    handle.setPos(-100, 9999)
    assert handle.pos() == QPointF(0, 299)
    canvas.close()


def test_editor_load_crop_filters_rotate_restore_and_accept(app, tmp_path, monkeypatch):
    monkeypatch.setenv('U2NET_HOME', str(tmp_path / 'no-model'))
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[2]))
    image = Image.new('RGB', (600, 390), '#bad2cc')
    path = tmp_path / 'مستمسك تجريبي.png'
    image.save(path)
    editor = m.EditorDialog(str(path), 8.6 / 5.6, 'اختبار')
    editor.show()
    wait_for(app, lambda: editor.current is not None and editor.worker is None)
    assert not editor.cropped
    editor.manual_crop()
    editor.apply_crop()
    wait_for(app, lambda: editor.worker is None)
    assert editor.cropped
    before = editor.current.copy()
    editor.enhance()
    wait_for(app, lambda: editor.worker is None)
    assert editor.filter_mode == 1
    editor.enhance()
    wait_for(app, lambda: editor.worker is None)
    assert editor.filter_mode == 2 and set(np.unique(editor.current)) <= {0, 255}
    editor.enhance()
    wait_for(app, lambda: editor.worker is None)
    np.testing.assert_array_equal(editor.current, before)
    editor.rotate()
    wait_for(app, lambda: editor.worker is None)
    assert editor.current.shape[:2] == before.shape[:2][::-1]
    editor.restore()
    np.testing.assert_array_equal(editor.current, editor.original)
    editor.confirm()
    wait_for(app, lambda: editor.worker is None)
    assert editor.result() == QDialog.Accepted and editor.result_image is not None
    assert not messages
    editor.deleteLater()


def test_editor_bad_file_is_recoverable(app, tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: messages.append(a[2]))
    editor = m.EditorDialog(str(tmp_path / 'missing.png'), 1.5, 'خطأ')
    editor.show()
    wait_for(app, lambda: editor.worker is None and bool(messages))
    assert editor.current is None
    assert editor.cancel_button.isEnabled()
    assert not editor.toolbar.isEnabled()
    assert not editor.confirm_button.isEnabled()
    editor.reject()
    editor.deleteLater()


def test_print_engine_a4_pdf_multiple_pages(app, tmp_path):
    path = tmp_path / 'qt-printer.pdf'
    printer = QPrinter(QPrinter.HighResolution)
    printer.setResolution(300)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(str(path))
    printer.setPageSize(QPrinter.A4)
    printer.setOrientation(QPrinter.Portrait)
    pages = m.layout_pages([m.Card(13, 18, Image.new('RGB', (130, 180), 'red'),
                                  Image.new('RGB', (130, 180), 'blue'))])
    m.paint_print_pages(printer, pages, 300)
    reader = PdfReader(path)
    assert len(reader.pages) == 2
    assert float(reader.pages[0].mediabox.width) == pytest.approx(210 * 72 / 25.4, abs=.5)
    assert float(reader.pages[0].mediabox.height) == pytest.approx(297 * 72 / 25.4, abs=.5)
    assert len(reader.pages[0].images) == 1
    assert len(reader.pages[1].images) == 1


def test_print_rejects_non_a4(app, tmp_path):
    printer = QPrinter()
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(str(tmp_path / 'wrong-size.pdf'))
    printer.setPageSize(QPrinter.Letter)
    with pytest.raises(ValueError, match='A4'):
        m.paint_print_pages(printer, [[]])


def test_rotate_uncropped_photo_preserves_pixels_and_selection(app, tmp_path, monkeypatch):
    monkeypatch.setenv('U2NET_HOME', str(tmp_path / 'no-model'))
    path = tmp_path / 'صورة جانبية.png'
    image = Image.new('RGB', (300, 200), '#b4c6dc')
    image.save(path)
    editor = m.EditorDialog(str(path), 1.5, 'تدوير قبل القص')
    editor.show()
    wait_for(app, lambda: editor.current is not None and editor.worker is None)
    editor.rotate()
    wait_for(app, lambda: editor.worker is None)
    assert editor.current.shape == (300, 200, 3)
    assert not editor.cropped
    np.testing.assert_array_equal(editor.current, m.cv2.rotate(editor.original, m.cv2.ROTATE_90_CLOCKWISE))
    assert len(editor.canvas.points()) == 4
    editor.reject()
    editor.deleteLater()
