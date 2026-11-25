"""Plot image -> measured outline -> layout that fills it exactly."""
import base64
from pathlib import Path
import sys
import unittest

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_layout
import plot_outline
from plan_postprocess import RoomPolygon
from plot_outline import PlotError, building_outline, detect_outline, measure_plot, solve_plot

L_SHAPE = [(0, 0), (90, 0), (90, 60), (150, 60), (150, 140), (0, 140)]
L_LENGTHS = [9, 6, 6, 8, 15, 14]


def drawing(points, filled=False, dark=False, frame=False, size=(600, 800)):
    import cv2
    image = np.full((*size, 3), 30 if dark else 255, np.uint8)
    ring = np.asarray(points, np.int32)
    if filled:
        cv2.fillPoly(image, [ring], (90, 170, 120))
    cv2.polylines(image, [ring], True, (240, 240, 240) if dark else (20, 20, 20), 4, cv2.LINE_AA)
    if frame:
        cv2.rectangle(image, (1, 1), (size[1] - 2, size[0] - 2), (0, 0, 0), 2)
    cv2.putText(image, "PLOT 12", (20, size[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (120, 120, 120), 2)
    return cv2.imencode(".png", image)[1].tobytes()


class Detection(unittest.TestCase):
    shapes = {
        "rectangle": [(150, 100), (650, 100), (650, 480), (150, 480)],
        "L": [(120, 80), (420, 80), (420, 260), (680, 260), (680, 500), (120, 500)],
        "trapezoid": [(200, 90), (560, 90), (700, 490), (120, 490)],
        "triangle": [(400, 80), (700, 480), (120, 480)],
        "pentagon": [(400, 70), (680, 250), (590, 500), (200, 500), (110, 250)],
    }

    def test_shapes_are_found_in_every_drawing_style(self):
        for name, points in self.shapes.items():
            for style in ({}, {"filled": True}, {"dark": True}, {"frame": True}):
                with self.subTest(shape=name, style=style):
                    found = detect_outline(drawing(points, **style))["points"]
                    self.assertEqual(len(found), len(points))
                    a, b = Polygon(found), Polygon(points)
                    self.assertGreater(a.intersection(b).area / a.union(b).area, 0.95)

    def test_large_images_report_original_pixels(self):
        big = [(x * 4, y * 4) for x, y in self.shapes["L"]]
        result = detect_outline(drawing(big, filled=True, size=(2400, 3200)))
        self.assertEqual((result["width"], result["height"]), (3200, 2400))
        self.assertGreater(Polygon(result["points"]).intersection(Polygon(big)).area / Polygon(big).area, .95)

    def test_blank_and_unreadable_files_are_explained(self):
        import cv2
        blank = cv2.imencode(".png", np.full((300, 300, 3), 255, np.uint8))[1].tobytes()
        with self.assertRaisesRegex(PlotError, "No closed plot outline"):
            detect_outline(blank)
        with self.assertRaisesRegex(PlotError, "not a readable image"):
            detect_outline(b"not an image")


class Lengths(unittest.TestCase):
    def test_typed_lengths_are_exact(self):
        corners = np.asarray(solve_plot([(0, 0), (100, 3), (98, 120), (0, 118)], [14, 15, 14, 15])["corners"])
        np.testing.assert_allclose(np.hypot(*(np.roll(corners, -1, 0) - corners).T), [14, 15, 14, 15], atol=1e-4)
        np.testing.assert_allclose(Polygon(corners).area, 210, rtol=1e-3)

    def test_slanted_lengths_close_by_turning(self):
        corners = np.asarray(solve_plot([(0, 0), (120, 0), (150, 140), (0, 140)], [12, 14.5, 15, 14])["corners"])
        np.testing.assert_allclose(np.hypot(*(np.roll(corners, -1, 0) - corners).T), [12, 14.5, 15, 14], atol=1e-3)

    def test_missing_or_impossible_lengths_are_rejected(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for lengths, message in (([14, None, 14, 15], "edge B"), ([14, 15, 14], "every edge"),
                                 ([100, 10, 10, 10], "longer than all"), ([14, 0, 14, 15], "between"),
                                 ([14, True, 14, 15], "edge B"), ([14, float("nan"), 14, 15], "edge B")):
            with self.subTest(lengths=lengths), self.assertRaisesRegex(PlotError, message):
                solve_plot(square, lengths)

    def test_right_angled_mismatch_names_the_edges(self):
        with self.assertRaisesRegex(PlotError, r"A\+C = 15 m but E = 18 m"):
            solve_plot(L_SHAPE, [9, 6, 6, 8, 18, 14])

    def test_small_measuring_error_is_shared_out(self):
        result = solve_plot(L_SHAPE, [9, 6, 6, 8, 15.2, 14])
        self.assertTrue(plot_outline.is_right_angled(Polygon(result["corners"])))
        self.assertTrue(result["notes"])


class Building(unittest.TestCase):
    def test_right_angled_plot_is_filled_to_its_edges(self):
        plot = measure_plot(L_SHAPE, L_LENGTHS)
        self.assertAlmostEqual(plot["footprint_m2"], plot["plot_area_m2"], places=3)
        self.assertEqual(len(plot["outline"]), 6)

    def test_slanted_plot_gets_a_right_angled_building_inside(self):
        for points in ([(0, 0), (12, 0), (15, 14), (0, 14)], [(0, 0), (20, 0), (8, 18)],
                       [(7, 0), (14, 5), (11, 14), (3, 14), (0, 5)]):
            built = building_outline(points)
            outline = Polygon(built["outline"])
            self.assertTrue(plot_outline.is_right_angled(outline))
            self.assertTrue(Polygon(built["plot"]).buffer(1e-6).contains(outline.buffer(.2, join_style=2)))
            self.assertGreater(built["coverage"], 0.4)

    def test_unusable_plots(self):
        for size, message in ((3, "At least"), (60, "up to")):
            with self.assertRaisesRegex(PlotError, message):
                building_outline([(0, 0), (size, 0), (size, size), (0, size)])


class Fitting(unittest.TestCase):
    def room(self, number, kind, *bounds):
        return RoomPolygon(room_id=number, type_name=kind, polygon=box(*bounds), name=f"{kind}_{number}")

    def test_notches_are_the_part_that_is_not_building(self):
        outline = Polygon(measure_plot(L_SHAPE, L_LENGTHS)["outline"])
        (x0, y0, x1, y1), = plot_layout.notches(outline)
        self.assertAlmostEqual(y0, -1)
        self.assertAlmostEqual(x1, 1)
        self.assertEqual(plot_layout.notches(box(0, 0, 5, 5)), [])

    def test_rooms_fill_the_outline_exactly(self):
        outline = Polygon(measure_plot(L_SHAPE, L_LENGTHS)["outline"])
        # Gaps, an overlap and spill into the missing corner.
        rooms = [self.room(0, "living", -.9, -.9, .1, .3), self.room(1, "bedroom", -.95, .35, -.1, .9),
                 self.room(2, "bedroom", -.1, .3, .5, .9), self.room(3, "bathroom", .45, .3, .9, .8),
                 self.room(4, "kitchen", .12, -.4, .9, .25)]
        report = plot_layout.fit_rooms(rooms, outline)
        self.assertGreater(report["spill_before_fit"], 0)
        self.assertLess(unary_union([r.polygon for r in rooms]).symmetric_difference(outline).area, 1e-6)
        for i, a in enumerate(rooms):
            self.assertTrue(plot_outline.is_right_angled(a.polygon))
            for b in rooms[i + 1:]:
                self.assertLess(a.polygon.intersection(b.polygon).area, 1e-9)

    def test_a_squeezed_room_gets_usable_space(self):
        outline = Polygon(measure_plot(L_SHAPE, L_LENGTHS)["outline"])
        rooms = [self.room(0, "living", -1, -1, 1, 1), self.room(1, "bathroom", .6, -.95, .7, -.9)]
        plot_layout.fit_rooms(rooms, outline)
        self.assertGreaterEqual(rooms[1].polygon.area, plot_layout.MIN_AREA_M2["bathroom"])
        self.assertLess(unary_union([r.polygon for r in rooms]).symmetric_difference(outline).area, 1e-6)

    def test_too_many_rooms_for_the_plot_are_rejected(self):
        rooms = [self.room(0, "living", -1, -1, 0, 0)] + [
            self.room(i, "bedroom", -.9 + .2 * i, -.9, -.6 + .2 * i, .9) for i in range(1, 8)]
        with self.assertRaises(plot_layout.PlotFitError):
            plot_layout.fit_rooms(rooms, box(0, 0, 5, 5.4))

    def test_guide_moves_rooms_out_of_the_missing_corner(self):
        outline = Polygon(measure_plot(L_SHAPE, L_LENGTHS)["outline"])
        guide = plot_layout.Guide(outline, {"room_ids": np.zeros(4, int), "corner_valid": np.ones(4, bool)})
        moved = guide.push(np.array([(.5, -.9), (.9, -.9), (.9, -.5), (.5, -.5)]))
        left, top, right, bottom = guide.notches[0]
        shape = Polygon(moved)
        self.assertLess(shape.intersection(box(left, top, right, bottom)).area, 1e-9)
        self.assertAlmostEqual(shape.area, .16, places=6)

    def test_ensuite_names_the_bedroom_it_opens_into(self):
        rooms = [self.room(0, "living", 0, 0, 1, 1), self.room(1, "bedroom", 0, 0, 1, 1),
                 self.room(2, "bedroom", 0, 0, 1, 1), self.room(3, "bathroom", 0, 0, 1, 1)]
        labels = {0: "Hall", 1: "Master bedroom", 2: "Study", 3: "En-suite bathroom"}
        sleeping = {"Bedroom", "Master bedroom"}
        new, lost = plot_layout.settle_ensuites(rooms, {(0, 1), (0, 2), (2, 3)}, labels, "En-suite bathroom", sleeping)
        self.assertEqual((new[1], new[2], lost), ("Study", "Master bedroom", 0))
        new, lost = plot_layout.settle_ensuites(rooms, {(0, 1), (0, 2), (0, 3)}, labels, "En-suite bathroom", sleeping)
        self.assertEqual((new[1], new[3], lost), ("Master bedroom", "Bathroom", 1))

    def test_canvas_scale_returns_the_typed_size(self):
        outline = box(0, 0, 13.6, 14.6)
        rooms = [RoomPolygon(room_id=0, type_name="living", polygon=outline, name="living_0")]
        scale, _ = plot_layout.to_canvas(rooms, outline)
        x0, y0, x1, y1 = rooms[0].polygon.bounds
        self.assertAlmostEqual((x1 - x0) * scale + .4, 14)
        self.assertAlmostEqual((y1 - y0) * scale + .4, 15)


class ServerImages(unittest.TestCase):
    def test_data_urls_and_bad_base64(self):
        from web_editor.server import read_image
        self.assertEqual(read_image("data:image/png;base64," + base64.b64encode(b"abc").decode()), b"abc")
        for bad in ("data:image/png;base64,%%%", None):
            with self.assertRaises(ValueError):
                read_image(bad)


if __name__ == "__main__":
    unittest.main()
