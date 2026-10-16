"""Optional integration test: executes real local inference, without downloading weights."""
import socket
import urllib.request

import cv2
import numpy as np
import pytest

import iraqi_id_printer as m


def test_real_u2net_is_offline_and_detects_quad(monkeypatch):
    if not m.local_model_path().is_file():
        pytest.skip('Optional U2-Net weights not installed')
    pytest.importorskip('rembg')

    def no_network(*args, **kwargs):
        pytest.fail('Image processing attempted network access')

    monkeypatch.setattr(socket, 'create_connection', no_network)
    monkeypatch.setattr(urllib.request, 'urlopen', no_network)
    image = np.full((650, 900, 3), (39, 48, 58), np.uint8)
    expected = np.array([[140, 130], [750, 80], [800, 480], [100, 510]], np.float32)
    cv2.fillConvexPoly(image, expected.astype(np.int32), (220, 235, 220))
    quad, message = m.detect_document(image)
    assert 'بالنموذج المحلي' in message
    assert m._AI_SESSION is not None
    assert np.max(np.linalg.norm(quad - expected, axis=1)) < 18
