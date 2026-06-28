"""Round-trip tests for the calibration core (no image deps needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import HexGrid, fit_from_anchors, parse_ccrr, to_ccrr


def test_ccrr_roundtrip():
    assert parse_ccrr("3115") == (31, 15)
    assert parse_ccrr("0108") == (1, 8)
    assert to_ccrr(31, 15) == "3115"
    assert to_ccrr(1, 8) == "0108"


def test_center_formula():
    g = HexGrid(image_full=(6518, 5139), col_pitch_x=104.08, row_pitch_y=120.19,
                x_intercept_col0=69.73, y_intercept_row0=522.3,
                even_col_y_offset=60.13, web_scale=0.5)
    # odd column: no down-shift
    x, y = g.center(1, 8)
    assert abs(x - (69.73 + 104.08)) < 0.01
    assert abs(y - (522.3 + 8 * 120.19)) < 0.01
    # even column: shifted down by even_col_y_offset
    _, ye = g.center(32, 8)
    assert abs(ye - (522.3 + 8 * 120.19 + 60.13)) < 0.01
    # web scale halves the coordinates
    xw, yw = g.center_web(1, 8)
    assert abs(xw - x * 0.5) < 0.01 and abs(yw - y * 0.5) < 0.01


def test_fit_recovers_known_model():
    truth = HexGrid(image_full=(6518, 5139), col_pitch_x=104.08, row_pitch_y=120.19,
                    x_intercept_col0=69.73, y_intercept_row0=522.3,
                    even_col_y_offset=60.13)
    # generate consistent anchors straight from the model (odd + even columns)
    anchors = []
    for col, row in [(1, 8), (3, 8), (29, 8), (32, 8), (47, 24), (48, 25), (50, 26)]:
        x, y = truth.center(col, row)
        anchors.append({"col": col, "row": row, "x": x, "y": y})
    fit = fit_from_anchors(anchors, image_full=(6518, 5139))
    assert abs(fit.col_pitch_x - 104.08) < 0.1
    assert abs(fit.row_pitch_y - 120.19) < 0.1
    assert abs(fit.x_intercept_col0 - 69.73) < 0.1
    assert abs(fit.even_col_y_offset - 60.13) < 0.5


if __name__ == "__main__":
    test_ccrr_roundtrip()
    test_center_formula()
    test_fit_recovers_known_model()
    print("all tests passed")
