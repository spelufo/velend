"""GPU side of the brick streaming system.

Everything here needs an active GPU context, so it may only be touched from
`view_draw`. The CPU side lives in `bricks.py`.

Two textures back each streamed level:

	  atlas      a big R8 3D texture holding padded bricks.
  pageTable  one R32F texel per chunk of the whole volume, holding `slot + 1`
	             of the brick that chunk lives in, or 0 when it is not resident.

Blender's `GPUTexture` has no sub-region upload, so bricks reach the atlas by
way of a small staging texture and a compute shader that copies it into place.
The page table is small enough to just be rebuilt whole whenever it changes.
"""

import os

import gpu
import numpy as np

from .bricks import BRICK_SIZE


_SHADERS_DIR = os.path.join(os.path.dirname(__file__), "shaders")
COPY_SHADER_PATH = os.path.join(_SHADERS_DIR, "brick_copy.comp")

# The copy shader guards against overrun, so the group size need not divide
# BRICK_SIZE evenly.
_COPY_GROUP = 4


def texture_data(array_zyx):
	"""Pack a numpy (Z, Y, X) array into a Buffer for a (width, height, depth) texture.

	`GPUTexture(size)` calls `GPU_texture_create_3d(width, height, depth)` and
	consumes the data with x varying fastest, so a C-order (Z, Y, X) array maps
	straight across: texel (x, y, z) holds `array_zyx[z, y, x]`.
	"""
	data = np.ascontiguousarray(array_zyx, dtype=np.float32)
	return gpu.types.Buffer('FLOAT', data.size, data)


def slot_origin(slot, slots_per_axis):
	"""Texel coordinate of a slot's corner within the atlas."""
	return (
		(slot % slots_per_axis) * BRICK_SIZE,
		((slot // slots_per_axis) % slots_per_axis) * BRICK_SIZE,
		(slot // (slots_per_axis * slots_per_axis)) * BRICK_SIZE,
	)


class BrickAtlas:
	def __init__(self, page_dims_xyz, slots_per_axis):
		self.page_dims = tuple(int(d) for d in page_dims_xyz)
		self.slots_per_axis = int(slots_per_axis)
		self.atlas_dim = self.slots_per_axis * BRICK_SIZE
		self.texture = gpu.types.GPUTexture(
			(self.atlas_dim, self.atlas_dim, self.atlas_dim), format='R8'
		)
		self.texture.filter_mode(True)
		self.texture.clear(format='FLOAT', value=(0.0, 0.0, 0.0, 1.0))
		self.page_texture = None
		self.copy_shader = None
		self.copy_mtime = None
		self.copy_failed = False

	def ensure_copy_shader(self):
		"""Compile the brick copy compute shader, reloading it when the file changes."""
		try:
			mtime = os.stat(COPY_SHADER_PATH).st_mtime_ns
		except OSError as error:
			print("vlend: cannot stat brick copy shader:", error)
			return self.copy_shader
		if self.copy_shader is not None and mtime == self.copy_mtime:
			return self.copy_shader
		if self.copy_failed and mtime == self.copy_mtime:
			return None

		self.copy_mtime = mtime
		try:
			with open(COPY_SHADER_PATH, encoding="utf-8") as source_file:
				source = source_file.read()

			info = gpu.types.GPUShaderCreateInfo()
			info.define("BRICK_SIZE", str(BRICK_SIZE))
			info.local_group_size(_COPY_GROUP, _COPY_GROUP, _COPY_GROUP)
			info.push_constant('IVEC3', "dstOrigin")
			info.sampler(0, 'FLOAT_3D', "brick")
			info.image(0, 'R8', 'FLOAT_3D', "atlas", qualifiers={'WRITE'})
			info.compute_source(source)
			self.copy_shader = gpu.shader.create_from_info(info)
			self.copy_failed = False
			print("vlend: loaded brick copy shader")
		except Exception as error:
			print("vlend: brick copy shader failed:", error)
			self.copy_shader = None
			self.copy_failed = True
		return self.copy_shader

	def upload(self, slot, brick):
		"""Copy one padded brick into its atlas slot. `brick` is (Z, Y, X) float32."""
		shader = self.ensure_copy_shader()
		if shader is None:
			return False

		staging = gpu.types.GPUTexture(
			(BRICK_SIZE, BRICK_SIZE, BRICK_SIZE),
			format='R8',
			data=texture_data(brick),
		)
		staging.filter_mode(False)

		shader.bind()
		shader.uniform_sampler("brick", staging)
		shader.image("atlas", self.texture)
		shader.uniform_int("dstOrigin", slot_origin(slot, self.slots_per_axis))
		groups = -(-BRICK_SIZE // _COPY_GROUP)
		gpu.compute.dispatch(shader, groups, groups, groups)
		return True

	def sync_page(self, page_zyx):
		"""Rebuild the page table texture from its CPU mirror.

		`page_zyx` is the (Z, Y, X) page table holding `slot + 1`, or 0 where the
		chunk is not resident.

		R32F rather than an integer format because `GPUTexture` only accepts
		`Buffer('FLOAT')` as initial data. Slot indices are small integers, so
		they survive the float round trip exactly.
		"""
		texture = gpu.types.GPUTexture(
			self.page_dims,
			format='R32F',
			data=texture_data(page_zyx),
		)
		texture.filter_mode(False)
		self.page_texture = texture
