import importlib.util
import struct
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "velend_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE, package)
spec = importlib.util.spec_from_file_location(
	PACKAGE + ".surface_output", ROOT / "surface_output.py"
)
surface_output = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = surface_output
spec.loader.exec_module(surface_output)

PNG_SIGNATURE = surface_output.PNG_SIGNATURE
PNGWriter = surface_output.PNGWriter
linear_rgba_to_srgb8 = surface_output.linear_rgba_to_srgb8
output_size = surface_output.output_size
read_velend_metadata = surface_output.read_velend_metadata
uv_output_size = surface_output.uv_output_size


def png_chunks(path):
	with open(path, "rb") as file:
		assert file.read(8) == PNG_SIGNATURE
		while True:
			length_data = file.read(4)
			if not length_data:
				return
			length = struct.unpack(">I", length_data)[0]
			kind = file.read(4)
			data = file.read(length)
			file.read(4)
			yield kind, data
			if kind == b"IEND":
				return


class SurfaceOutputTests(unittest.TestCase):
	def test_physical_pixel_size_sets_dimensions(self):
		self.assertEqual(output_size((100, 200), (2, 4), 8, 4), (200, 50))
		self.assertEqual(output_size((11, 17), (1, 1), 9.362, 9.362), (17, 11))

	def test_invalid_physical_values_are_rejected(self):
		for args in (
			((0, 2), (1, 1), 1, 1),
			((2, 2), (0, 1), 1, 1),
			((2, 2), (1, 1), 1, 0),
		):
			with self.assertRaises(ValueError):
				output_size(*args)

	def test_uv_parameterization_sets_physical_texel_density(self):
		positions = np.array([
			[0, 0, 0], [2, 0, 0], [2, 1, 0],
			[0, 0, 0], [2, 1, 0], [0, 1, 0],
		], dtype=np.float32)
		uvs = np.array([
			[0, 0], [1, 0], [1, 1],
			[0, 0], [1, 1], [0, 1],
		], dtype=np.float32)
		(size, physical) = uv_output_size(positions, uvs, np.eye(4), 1000.0, 100.0)
		self.assertEqual(size, (20, 10))
		np.testing.assert_allclose(physical, (2000.0, 1000.0))

	def test_linear_framebuffer_is_encoded_as_srgb(self):
		linear = np.array([[[0.0, 0.0031308, 0.5, 1.0]]], dtype=np.float32)
		encoded = linear_rgba_to_srgb8(linear)
		np.testing.assert_array_equal(encoded, [[[0, 10, 188, 255]]])

	def test_png_rows_and_unicode_metadata_round_trip(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "surface.png"
			metadata = {"schema": "velend.surface_render.v1", "surface": "ψηφίδες"}
			pixels = np.array([
				[[255, 0, 0, 255], [0, 255, 0, 0]],
				[[0, 0, 255, 255], [255, 255, 255, 255]],
			], dtype=np.uint8)
			writer = PNGWriter(path, 2, 2, metadata)
			writer.write_rows(pixels[:1])
			writer.write_rows(pixels[1:])
			writer.close()

			self.assertEqual(read_velend_metadata(path), metadata)
			chunks = list(png_chunks(path))
			self.assertEqual(chunks[0][0], b"IHDR")
			self.assertEqual(chunks[-1][0], b"IEND")
			encoded = b"".join(data for kind, data in chunks if kind == b"IDAT")
			raw = zlib.decompress(encoded)
			rows = np.frombuffer(raw, dtype=np.uint8).reshape(2, 9)
			np.testing.assert_array_equal(rows[:, 0], 0)
			np.testing.assert_array_equal(rows[:, 1:].reshape(2, 2, 4), pixels)


if __name__ == "__main__":
	unittest.main()
