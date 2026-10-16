import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader
from pypdf.generic import ContentStream

import iraqi_id_printer as m


def pair(width=8.6, height=5.6):
    return m.Card(width, height, Image.new('RGB', (172, 112), 'red'),
                  Image.new('RGB', (172, 112), 'blue'))


def test_four_pairs_actual_size_and_order():
    pages = m.layout_pages([pair() for _ in range(4)])
    assert len(pages) == 1 and len(pages[0]) == 8
    for index in range(4):
        back, front = pages[0][index * 2:index * 2 + 2]
        assert (back.side, front.side) == ('back', 'front')
        assert back.x_mm == 17 and front.x_mm == 107
        assert front.width_mm == 86 and front.height_mm == 56
        assert front.y_mm == back.y_mm == 10 + 60 * index


def test_single_side_centered_and_empty_slots_ignored():
    pages = m.layout_pages([m.Card(), m.Card(front=Image.new('RGB', (100, 60)))])
    assert len(pages[0]) == 1
    assert pages[0][0].x_mm == 62
    assert pages[0][0].y_mm == 10
    assert pages[0][0].card_index == 1


def test_legacy_pagination():
    pages = m.layout_pages([pair(6.5, 9.5) for _ in range(4)])
    assert len(pages) == 2
    assert [len(p) for p in pages] == [4, 4]
    assert [p[0].card_index for p in pages] == [0, 2]


def test_large_pairs_stack_without_shrinking():
    pages = m.layout_pages([pair(13, 18)])
    assert len(pages) == 2
    assert pages[0][0].side == 'front' and pages[1][0].side == 'back'
    assert pages[0][0].width_mm == 130 and pages[0][0].height_mm == 180
    assert pages[0][0].x_mm == 40


@pytest.mark.parametrize('width,height', [(20, 5), (5, 29), (float('nan'), 5), (-3, 5)])
def test_oversize_or_invalid_dimensions_rejected(width, height):
    with pytest.raises(ValueError):
        m.layout_pages([pair(width, height)])


@pytest.mark.parametrize('gap', [-1, 31, float('nan')])
def test_invalid_spacing_rejected(gap):
    with pytest.raises(ValueError):
        m.layout_pages([pair()], gap, 4)


def test_pixel_sizes_and_front_back_pixels():
    placements = m.layout_pages([pair()])[0]
    image = m.render_page(placements, 300)
    assert image.size == (2480, 3508)
    assert image.getpixel((m.mm_to_px(150, 300), m.mm_to_px(30, 300))) == (255, 0, 0)
    assert image.getpixel((m.mm_to_px(50, 300), m.mm_to_px(30, 300))) == (0, 0, 255)
    assert image.getpixel((5, 5)) == (255, 255, 255)
    assert m.render_page(placements, 600).size == (4961, 7016)


def test_pdf_a4_physical_image_matrices_and_author(tmp_path):
    path = tmp_path / 'مستمسكاتي.pdf'
    pages = m.layout_pages([pair(13, 18)])
    m.export_pdf(path, pages, 300)
    reader = PdfReader(path)
    assert len(reader.pages) == 2
    assert 'Hasan Wadhah' in reader.metadata.author
    for page in reader.pages:
        assert float(page.mediabox.width) == pytest.approx(210 / 25.4 * 72, abs=.001)
        assert float(page.mediabox.height) == pytest.approx(297 / 25.4 * 72, abs=.001)
        transforms = [args for args, op in ContentStream(page.get_contents(), reader).operations
                      if op == b'cm' and abs(float(args[0]) - 130 / 25.4 * 72) < .01]
        assert len(transforms) == 1
        assert float(transforms[0][3]) == pytest.approx(180 / 25.4 * 72, abs=.001)


@pytest.mark.parametrize('suffix,dpi', [('.png', 300), ('.jpg', 600)])
def test_image_export_numbering_metadata_and_size(tmp_path, suffix, dpi):
    paths = m.export_images(tmp_path / ('هويات' + suffix), m.layout_pages([pair(13, 18)]), dpi)
    assert [path.name for path in paths] == ['هويات_01' + suffix, 'هويات_02' + suffix]
    for path in paths:
        with Image.open(path) as image:
            assert image.size == tuple(m.mm_to_px(mm, dpi) for mm in m.A4_MM)
            assert image.info['dpi'][0] == pytest.approx(dpi, abs=.1)


def test_atomic_export_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / 'keep.pdf'
    path.write_bytes(b'previous export')
    monkeypatch.setattr(m.pdf_canvas, 'Canvas', lambda *a, **k: (_ for _ in ()).throw(OSError('disk failure')))
    with pytest.raises(OSError):
        m.export_pdf(path, m.layout_pages([pair()]))
    assert path.read_bytes() == b'previous export'
    assert not list(tmp_path.glob('.mostamskaty-*'))


def test_layout_never_places_content_outside_margins():
    for width, height in m.PRESETS.values():
        for horizontal in (0, 4, 30):
            for page in m.layout_pages([pair(width, height) for _ in range(4)], horizontal, 30):
                for p in page:
                    assert p.x_mm >= 10 and p.y_mm >= 10
                    assert p.x_mm + p.width_mm <= 200.00001
                    assert p.y_mm + p.height_mm <= 287.00001
