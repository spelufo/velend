"""Physical image sizing and streaming PNG output, without Blender dependencies."""

import binascii
import json
import math
import struct
import zlib

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_MAX_DIMENSION = (1 << 31) - 1


def output_size(shape, scale, source_voxel_um, pixel_size_um):
	"""PNG (width, height) for a tifxyz grid at a physical pixel size."""
	height, width = (int(value) for value in shape)
	scale_x, scale_y = (float(value) for value in scale)
	values = (width, height, scale_x, scale_y, source_voxel_um, pixel_size_um)
	if width < 1 or height < 1 or not all(math.isfinite(float(value)) for value in values):
		raise ValueError("the tifxyz grid size and scale must be finite and positive")
	if min(scale_x, scale_y, source_voxel_um, pixel_size_um) <= 0:
		raise ValueError("the tifxyz grid scale and pixel size must be positive")
	result = (
		max(1, round(width * source_voxel_um / scale_x / pixel_size_um)),
		max(1, round(height * source_voxel_um / scale_y / pixel_size_um)),
	)
	if max(result) > PNG_MAX_DIMENSION:
		raise ValueError("the requested pixel size makes an image too large for PNG")
	return result


def uv_output_size(positions, uvs, object_matrix, um_per_unit, pixel_size_um):
	"""Image size from the RMS physical stretch of the mesh's UV parameterization."""
	positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3, 3)
	uvs = np.asarray(uvs, dtype=np.float64).reshape(-1, 3, 2)
	matrix = np.asarray(object_matrix, dtype=np.float64)
	if len(positions) != len(uvs) or matrix.shape != (4, 4):
		raise ValueError("mesh positions, UV triangles, or object transform are invalid")
	if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
		raise ValueError("pixel size must be positive")

	world = positions @ matrix[:3, :3].T + matrix[:3, 3]
	duv1 = uvs[:, 1] - uvs[:, 0]
	duv2 = uvs[:, 2] - uvs[:, 0]
	edge1 = world[:, 1] - world[:, 0]
	edge2 = world[:, 2] - world[:, 0]
	determinant = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
	valid = np.isfinite(determinant) & (np.abs(determinant) > 1e-12)
	if not valid.any():
		raise ValueError("the active UV map has no non-degenerate triangles")

	determinant = determinant[valid]
	duv1, duv2 = duv1[valid], duv2[valid]
	edge1, edge2 = edge1[valid], edge2[valid]
	du = (
		edge1 * duv2[:, 1, None] - edge2 * duv1[:, 1, None]
	) / determinant[:, None]
	dv = (
		-edge1 * duv2[:, 0, None] + edge2 * duv1[:, 0, None]
	) / determinant[:, None]
	weights = np.abs(determinant)
	u_um = math.sqrt(np.average(np.einsum('ij,ij->i', du, du), weights=weights))
	v_um = math.sqrt(np.average(np.einsum('ij,ij->i', dv, dv), weights=weights))
	u_um *= float(um_per_unit)
	v_um *= float(um_per_unit)
	result = (max(1, round(u_um / pixel_size_um)), max(1, round(v_um / pixel_size_um)))
	if not all(math.isfinite(value) for value in (u_um, v_um)) or max(result) > PNG_MAX_DIMENSION:
		raise ValueError("the UV parameterization makes an image too large for PNG")
	return result, (u_um, v_um)


def linear_rgba_to_srgb8(rgba):
	"""Encode a floating-point linear framebuffer as straight-alpha sRGB bytes."""
	rgba = np.asarray(rgba, dtype=np.float32)
	rgb = np.clip(rgba[..., :3], 0.0, 1.0)
	rgb = np.where(
		rgb <= 0.0031308,
		12.92 * rgb,
		1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
	)
	encoded = np.empty(rgba.shape, dtype=np.uint8)
	encoded[..., :3] = np.rint(rgb * 255.0).astype(np.uint8)
	encoded[..., 3] = np.rint(np.clip(rgba[..., 3], 0.0, 1.0) * 255.0).astype(np.uint8)
	return encoded


def _png_chunk(kind, data):
	payload = kind + data
	return struct.pack(">I", len(data)) + payload + struct.pack(">I", binascii.crc32(payload))


class PNGWriter:
	"""Incrementally write top-to-bottom RGBA rows and one JSON iTXt field."""

	def __init__(self, path, width, height, metadata):
		self.path = path
		self.width = int(width)
		self.height = int(height)
		self.rows = 0
		self.file = open(path, "wb")
		self.compressor = zlib.compressobj(level=6)
		self.file.write(PNG_SIGNATURE)
		self.file.write(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
		text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
		# keyword, compression flag/method, language, translated keyword, UTF-8 text.
		self.file.write(_png_chunk(b"iTXt", b"velend\0\0\0\0\0" + text))

	def write_rows(self, rgba):
		rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
		if rgba.ndim != 3 or rgba.shape[1:] != (self.width, 4):
			raise ValueError("PNG rows do not match its width")
		if self.rows + len(rgba) > self.height:
			raise ValueError("too many PNG rows")
		filtered = np.empty((len(rgba), 1 + self.width * 4), dtype=np.uint8)
		filtered[:, 0] = 0
		filtered[:, 1:] = rgba.reshape(len(rgba), self.width * 4)
		data = self.compressor.compress(filtered.tobytes())
		if data:
			self.file.write(_png_chunk(b"IDAT", data))
		self.rows += len(rgba)

	def close(self):
		if self.file is None:
			return
		if self.rows != self.height:
			raise ValueError("PNG ended after %d of %d rows" % (self.rows, self.height))
		data = self.compressor.flush()
		if data:
			self.file.write(_png_chunk(b"IDAT", data))
		self.file.write(_png_chunk(b"IEND", b""))
		self.file.close()
		self.file = None

	def abort(self):
		if self.file is not None:
			self.file.close()
			self.file = None


def read_velend_metadata(path):
	"""Read the embedded Velend JSON, primarily for validation and tests."""
	with open(path, "rb") as file:
		if file.read(8) != PNG_SIGNATURE:
			raise ValueError("not a PNG")
		while True:
			length_data = file.read(4)
			if not length_data:
				break
			length = struct.unpack(">I", length_data)[0]
			kind = file.read(4)
			data = file.read(length)
			file.read(4)
			if kind == b"iTXt" and data.startswith(b"velend\0"):
				parts = data.split(b"\0", 5)
				return json.loads(parts[5].decode("utf-8"))
			if kind == b"IEND":
				break
	return None
