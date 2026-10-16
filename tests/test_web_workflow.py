import io
import os
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from iraqi_id_printer import Card, detect_document, rectangle_points
from web.workflow import WebEditor, decode_upload, prepare_export

APP_PATH = Path(__file__).resolve().parents[1] / 'web' / 'streamlit_app.py'


def sample_bytes(size=(600, 390), color='#bad2cc'):
    output = io.BytesIO()
    Image.new('RGB', size, color).save(output, format='PNG')
    return output.getvalue()


def test_web_decode_caps_resolution_and_handles_alpha_exif():
    assert max(decode_upload(sample_bytes((3000, 2400))).shape[:2]) == 2200
    output = io.BytesIO()
    image = Image.new('RGBA', (80, 50), (255, 0, 0, 128))
    exif = image.getexif(); exif[274] = 6
    image.save(output, format='PNG', exif=exif)
    decoded = decode_upload(output.getvalue())
    assert decoded.shape == (80, 50, 3)
    assert tuple(decoded[30, 20]) == (255, 127, 127)


def test_web_decode_rejects_invalid_and_large_payload():
    for data in (b'bad image', b'x' * (12 * 1024 * 1024 + 1)):
        with pytest.raises(Exception):
            decode_upload(data)


def test_web_mode_never_loads_ai_even_if_model_present(monkeypatch):
    monkeypatch.setattr('iraqi_id_printer.local_model_path', lambda: pytest.fail('AI model lookup'))
    points, message = detect_document(np.full((200, 300, 3), 255, np.uint8), use_ai=False)
    assert points.shape == (4, 2)
    editor = WebEditor.from_upload(sample_bytes(), 8.6 / 5.6, 0, 'front')
    assert editor.current.shape == (390, 600, 3)


def test_editor_revision_prevents_stale_browser_coordinates():
    editor = WebEditor.from_upload(sample_bytes(), 8.6 / 5.6, 0, 'front')
    old_revision = editor.revision
    editor.apply('rotate')
    before = editor.points.copy()
    editor.update_points({'revision': old_revision, 'points': [[0, 0]] * 4})
    np.testing.assert_array_equal(editor.points, before)
    assert not editor.cropped
    assert editor.current.shape == (600, 390, 3)


def test_web_editor_crop_enhance_reset_and_accept():
    editor = WebEditor.from_upload(sample_bytes(), 8.6 / 5.6, 1, 'back')
    points = rectangle_points(editor.current, .1)
    editor.update_points({'revision': editor.revision, 'points': points.tolist()})
    editor.apply('crop')
    base = editor.current.copy()
    for mode in (1, 2, 0):
        editor.apply('enhance')
        assert editor.filter_mode == mode
    np.testing.assert_array_equal(editor.current, base)
    assert isinstance(editor.accept(), Image.Image)
    editor.apply('reset')
    np.testing.assert_array_equal(editor.current, editor.original)
    assert not editor.cropped


def test_web_rejects_crossed_and_out_of_bounds_points():
    editor = WebEditor.from_upload(sample_bytes(), 8.6 / 5.6, 0, 'front')
    crossed = [[20, 20], [550, 350], [550, 20], [20, 350]]
    editor.update_points({'revision': editor.revision, 'points': crossed})
    with pytest.raises(ValueError):
        editor.accept()
    with pytest.raises(ValueError):
        editor.update_points({'revision': editor.revision, 'points': [[-1, 0]] * 4})


def test_web_pdf_and_multi_page_zip_have_exact_sizes():
    card = Card(13, 18, Image.new('RGB', (130, 180), 'red'), Image.new('RGB', (130, 180), 'blue'))
    data, mime, name = prepare_export([card], 4, 4, 300, 'PDF')
    assert mime == 'application/pdf' and name.endswith('.pdf')
    pdf = PdfReader(io.BytesIO(data))
    assert len(pdf.pages) == 2
    assert float(pdf.pages[0].mediabox.width) == pytest.approx(210 * 72 / 25.4, abs=.001)
    data, mime, name = prepare_export([card], 4, 4, 300, 'PNG')
    assert mime == 'application/zip'
    with ZipFile(io.BytesIO(data)) as archive:
        assert len(archive.namelist()) == 2
        for entry in archive.namelist():
            with Image.open(io.BytesIO(archive.read(entry))) as image:
                assert image.size == (2480, 3508)
                assert image.info['dpi'][0] == pytest.approx(300, abs=.1)


def test_empty_web_export_is_rejected():
    with pytest.raises(ValueError):
        prepare_export([Card()], 4, 4, 300, 'PDF')


def test_streamlit_editor_accept_preview_export_and_clear():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not app.exception
    assert next(button for button in app.button if button.label == 'تجهيز ملف التنزيل').disabled
    app.session_state['editor'] = WebEditor.from_upload(sample_bytes(), 8.6 / 5.6, 0, 'front')
    app.run()
    assert not app.exception
    app.button(key='tool_crop').click().run()
    assert not app.exception and app.session_state['editor'].cropped
    next(button for button in app.button if button.label == 'اعتماد وتثبيت').click().run()
    assert not app.exception
    assert app.session_state['editor'] is None
    assert app.session_state['cards'][0].front is not None
    next(button for button in app.button if button.label == 'تجهيز ملف التنزيل').click().run()
    assert not app.exception
    assert len(app.get('download_button')) == 1
    assert 'PDF' in app.session_state['exports']
    app.selectbox(key='preset').select('شهادة الجنسية العراقية').run()
    assert not app.exception
    assert app.session_state['cards'][0].width_cm == 13
    assert not app.session_state['exports']
    next(button for button in app.button if button.label == 'مسح الجلسة بالكامل').click().run()
    assert not app.exception
    assert all(c.front is None and c.back is None for c in app.session_state['cards'])


def test_streamlit_invalid_dimensions_block_export():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.session_state['cards'][0].front = Image.new('RGB', (100, 60), 'red')
    app.run()
    app.number_input(key='width_0').set_value(25).run()
    assert not app.exception
    assert any('أكبر من المساحة' in error.value for error in app.error)
    assert not any(button.label == 'تجهيز ملف التنزيل' for button in app.button)
