import itertools
import socket

import cv2
import numpy as np
import pytest
from PIL import Image

import iraqi_id_printer as m


@pytest.fixture
def scene():
    card = np.full((390, 600, 3), (228, 238, 220), np.uint8)
    cv2.rectangle(card, (20, 20), (580, 370), (30, 70, 95), 3)
    cv2.circle(card, (75, 75), 25, (240, 30, 20), -1)
    cv2.circle(card, (525, 315), 25, (20, 40, 240), -1)
    for y in range(140, 320, 32):
        cv2.line(card, (160, y), (520, y), (20, 25, 30), 3)
    quad = np.array([[140, 130], [750, 80], [800, 480], [100, 510]], np.float32)
    transform = cv2.getPerspectiveTransform(m.rectangle_points(card), quad)
    projected = cv2.warpPerspective(card, transform, (900, 650))
    mask = cv2.warpPerspective(np.full(card.shape[:2], 255, np.uint8), transform, (900, 650))
    background = np.full((650, 900, 3), (39, 48, 58), np.uint8)
    background[mask > 0] = projected[mask > 0]
    return background, quad


def test_unicode_path_and_transparency(tmp_path):
    path = tmp_path / 'هوية حسن وضاح.png'
    Image.new('RGBA', (80, 50), (255, 0, 0, 128)).save(path)
    decoded = m.read_image(path)
    assert decoded.shape == (50, 80, 3)
    assert tuple(decoded[20, 20]) == (255, 127, 127)


def test_exif_orientation(tmp_path):
    path = tmp_path / 'تدوير.jpg'
    image = Image.new('RGB', (80, 50), 'red')
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    assert m.read_image(path).shape == (80, 50, 3)


def test_invalid_file(tmp_path):
    path = tmp_path / 'bad.png'
    path.write_bytes(b'not an image')
    with pytest.raises(Exception):
        m.read_image(path)


def test_point_order_all_permutations():
    expected = np.array([[100, 50], [430, 80], [410, 300], [80, 270]], np.float32)
    for permutation in itertools.permutations(expected):
        np.testing.assert_array_equal(m.order_points(permutation), expected)


def test_edge_detection_offline(scene, monkeypatch, tmp_path):
    monkeypatch.setenv('U2NET_HOME', str(tmp_path))
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('Network used'))
    rgb, expected = scene
    quad, message = m.detect_document(rgb)
    assert 'كشف الحواف' in message
    assert np.max(np.linalg.norm(quad - expected, axis=1)) < 15
    warped = m.warp_document(rgb, quad, 8.6 / 5.6)
    assert abs(warped.shape[1] / warped.shape[0] - 8.6 / 5.6) < 0.01
    # The top-left red and bottom-right blue markers survive in correct orientation.
    assert warped[round(warped.shape[0] * .19), round(warped.shape[1] * .125), 0] > 180
    assert warped[round(warped.shape[0] * .81), round(warped.shape[1] * .875), 2] > 180


def test_blank_detection_warns(monkeypatch, tmp_path):
    monkeypatch.setenv('U2NET_HOME', str(tmp_path))
    quad, message = m.detect_document(np.full((200, 300, 3), 255, np.uint8))
    assert 'يدويًا' in message
    assert quad.shape == (4, 2)


def test_mask_contour():
    mask = np.zeros((500, 700), np.uint8)
    cv2.rectangle(mask, (80, 80), (600, 420), 255, -1)
    quad = m.mask_quad(mask)
    np.testing.assert_allclose(quad, [[80, 80], [600, 80], [600, 420], [80, 420]], atol=1)


@pytest.mark.parametrize('points', [
    [[0, 0], [0, 0], [20, 20], [30, 30]],
    [[0, 0], [99, 99], [99, 0], [0, 99]],
    [[-3, 0], [99, 0], [99, 99], [0, 99]],
    [[0, 0], [99, 0], [99, float('nan')], [0, 99]],
])
def test_invalid_crops_rejected(points):
    with pytest.raises(ValueError):
        m.warp_document(np.zeros((100, 100, 3), np.uint8), np.array(points, np.float32), 1.5)


def test_enhancement_modes_do_not_mutate(scene):
    rgb = scene[0]
    original = rgb.copy()
    color = m.enhance_image(rgb, 'color')
    bw = m.enhance_image(rgb, 'bw')
    np.testing.assert_array_equal(rgb, original)
    assert color.dtype == np.uint8 and color.shape == rgb.shape
    assert not np.array_equal(color, rgb)
    assert set(np.unique(bw)) <= {0, 255}
    np.testing.assert_array_equal(bw[:, :, 0], bw[:, :, 1])


def test_deskew_corrects_text_lines():
    image = np.full((500, 800, 3), 255, np.uint8)
    for y in range(70, 430, 35):
        cv2.line(image, (80, y), (720, y), (0, 0, 0), 3)
    matrix = cv2.getRotationMatrix2D((400, 250), -5, 1)
    tilted = cv2.warpAffine(image, matrix, (800, 500), borderValue=(255, 255, 255))
    corrected, angle = m.deskew_image(tilted)
    assert abs(angle - 5) < 0.7
    _, residual = m.deskew_image(corrected)
    assert abs(residual) < 0.7


def test_corrupt_local_model_falls_back_without_network(scene, tmp_path, monkeypatch):
    (tmp_path / 'u2net.onnx').write_bytes(b'corrupt model')
    monkeypatch.setenv('U2NET_HOME', str(tmp_path))
    monkeypatch.setenv('NUMBA_CACHE_DIR', str(tmp_path / 'numba'))
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('Network used'))
    quad, message = m.detect_document(scene[0])
    assert 'كشف الحواف' in message
    assert 'تعذر تشغيل' in message
    assert quad.shape == (4, 2)
