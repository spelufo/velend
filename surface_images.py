"""Blender-native image layers attached to tifxyz surface materials."""

import os

import bpy

from .renderer import VolumeSamplerRenderEngine as Engine


IMAGE_TAG = "velend_surface_image"
IMAGE_ROLE = "velend_surface_image_role"
ROLES = (
	('RENDER', "Render", "A flattened render of the scan"),
	('PREDICTION', "Prediction", "An ink prediction or other model output"),
	('ANNOTATION', "Annotation", "An image intended for texture painting"),
	('OTHER', "Other", "Another image sharing the surface UV domain"),
)
MODES = (
	('AUTO', "Auto", "Use a supported material image connection, otherwise sample the volume"),
	('VOLUME', "Volume", "Always sample the live volume"),
	('TEXTURE', "Texture", "Render the image connected to the material output"),
	('OVERLAY', "Overlay", "Composite the connected image over the sampled volume"),
)


def is_surface(obj):
	return obj is not None and obj.type == 'MESH' and "velend_tifxyz_shape" in obj


def active_material(obj):
	if obj is None or obj.type != 'MESH':
		return None
	return obj.active_material


def managed_nodes(material):
	if material is None or not material.use_nodes:
		return []
	return [
		node for node in material.node_tree.nodes
		if node.bl_idname == 'ShaderNodeTexImage' and node.get(IMAGE_TAG, False)
	]


def selected_node(material):
	if material is None or not material.use_nodes:
		return None
	node = material.node_tree.nodes.active
	return node if node in managed_nodes(material) else None


def _follow_input(socket):
	while socket is not None and socket.is_linked:
		node = socket.links[0].from_node
		if node.bl_idname != 'NodeReroute':
			return node
		socket = node.inputs[0]
	return None


def _socket(node, name):
	return node.inputs.get(name) if node is not None else None


def resolved_image(material):
	"""Return (image, uv layer, node), or (None, reason, None).

	The deliberately small graph contract starts at the active Material Output,
	then accepts Principled Base Color or Emission Color through reroutes.
	"""
	if material is None or not material.use_nodes or material.node_tree is None:
		return None, "the object has no node material", None
	nodes = material.node_tree.nodes
	output = next((
		node for node in nodes
		if node.bl_idname == 'ShaderNodeOutputMaterial' and node.is_active_output
	), None)
	if output is None:
		return None, "the material has no active Material Output", None
	shader = _follow_input(_socket(output, "Surface"))
	if shader is None:
		return None, "Material Output Surface is not connected", None
	if shader.bl_idname == 'ShaderNodeBsdfPrincipled':
		color = _socket(shader, "Base Color")
	elif shader.bl_idname == 'ShaderNodeEmission':
		color = _socket(shader, "Color")
	else:
		return None, "Material Output must use Principled BSDF or Emission", None
	image_node = _follow_input(color)
	if image_node is None or image_node.bl_idname != 'ShaderNodeTexImage':
		return None, "the shader color is not connected directly to an Image Texture", None
	if image_node.image is None:
		return None, "the connected Image Texture has no image", None
	uv_node = _follow_input(_socket(image_node, "Vector"))
	if uv_node is None or uv_node.bl_idname != 'ShaderNodeUVMap' or not uv_node.uv_map:
		return None, "the connected Image Texture needs an explicit UV Map node", None
	return image_node.image, uv_node.uv_map, image_node


def render_source(material):
	"""Resolve a material to (mode, image, uv layer, node, warning)."""
	mode = getattr(material, "velend_render_mode", 'AUTO') if material else 'AUTO'
	image, detail, node = resolved_image(material)
	if mode == 'AUTO':
		mode = 'TEXTURE' if image is not None else 'VOLUME'
	if mode in {'TEXTURE', 'OVERLAY'} and image is None:
		return 'VOLUME', None, None, None, detail
	return mode, image, detail if image is not None else None, node, None


def _surface_material(obj):
	material = active_material(obj)
	if material is None:
		material = bpy.data.materials.new(obj.name + " Surface")
		material.use_nodes = True
		obj.data.materials.append(material)
		obj.active_material_index = len(obj.data.materials) - 1
	elif material.users > 1:
		copy = material.copy()
		copy.name = obj.name + " Surface"
		obj.material_slots[obj.active_material_index].material = copy
		material = copy
	material.use_nodes = True
	return material


def _uv_node(material, uv_name):
	nodes = material.node_tree.nodes
	for node in nodes:
		if node.bl_idname == 'ShaderNodeUVMap' and node.uv_map == uv_name:
			return node
	node = nodes.new('ShaderNodeUVMap')
	node.name = "Velend Surface UV"
	node.label = "Velend Surface UV"
	node.uv_map = uv_name
	node.location = (-700, 0)
	return node


def layout_image_nodes(material):
	"""Place managed image nodes loose beside the material shader."""
	nodes = material.node_tree.nodes
	output = next((
		node for node in nodes
		if node.bl_idname == 'ShaderNodeOutputMaterial' and node.is_active_output
	), None)
	shader = _follow_input(output.inputs.get("Surface")) if output is not None else None
	anchor_x = (shader.location.x if shader is not None else 0.0) - 360.0
	anchor_y = (shader.location.y if shader is not None else 0.0) + 280.0
	images = managed_nodes(material)
	for index, node in enumerate(images):
		node.parent = None
		node.location = (anchor_x, anchor_y - index * 280.0)
	uv_nodes = {
		link.from_node
		for node in images for link in node.inputs["Vector"].links
		if link.from_node.bl_idname == 'ShaderNodeUVMap'
	}
	for index, node in enumerate(sorted(uv_nodes, key=lambda item: item.name)):
		node.parent = None
		node.location = (anchor_x - 220.0, anchor_y - index * 180.0)
	frame = nodes.get("Velend Surface Images")
	if frame is not None:
		nodes.remove(frame)


@bpy.app.handlers.persistent
def _migrate_image_layout(_file_path=None):
	for material in bpy.data.materials:
		if managed_nodes(material):
			layout_image_nodes(material)


def _migrate_when_registered():
	"""Wait until every module, including the Scene settings, is registered."""
	if any(not hasattr(scene, "velend") for scene in bpy.data.scenes):
		return 0.1
	_migrate_image_layout()
	return None


def attach_image(obj, image, role='PREDICTION'):
	material = _surface_material(obj)
	uv = obj.data.uv_layers.active
	if uv is None:
		raise ValueError("the surface has no active UV map")
	for node in managed_nodes(material):
		if node.image == image:
			node[IMAGE_ROLE] = role
			node.label = "%s: %s" % (role.title(), image.name)
			material.node_tree.nodes.active = node
			node.select = True
			layout_image_nodes(material)
			return node
	nodes = material.node_tree.nodes
	tex = nodes.new('ShaderNodeTexImage')
	tex.name = "Velend %s: %s" % (role.title(), image.name)
	tex.label = "%s: %s" % (role.title(), image.name)
	tex.image = image
	tex[IMAGE_TAG] = True
	tex[IMAGE_ROLE] = role
	material.node_tree.links.new(_uv_node(material, uv.name).outputs["UV"], tex.inputs["Vector"])
	for node in nodes:
		node.select = False
	tex.select = True
	nodes.active = tex
	layout_image_nodes(material)
	Engine.request_redraw()
	return tex


def connect_for_render(material, image_node):
	nodes = material.node_tree.nodes
	output = next((
		node for node in nodes
		if node.bl_idname == 'ShaderNodeOutputMaterial' and node.is_active_output
	), None)
	if output is None:
		output = nodes.new('ShaderNodeOutputMaterial')
	shader = _follow_input(output.inputs["Surface"])
	if shader is not None and shader.bl_idname == 'ShaderNodeEmission':
		material.node_tree.links.new(image_node.outputs["Color"], shader.inputs["Color"])
		layout_image_nodes(material)
		Engine.request_redraw()
		return
	if shader is None or shader.bl_idname != 'ShaderNodeBsdfPrincipled':
		shader = nodes.new('ShaderNodeBsdfPrincipled')
		material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
	material.node_tree.links.new(image_node.outputs["Color"], shader.inputs["Base Color"])
	if "Alpha" in shader.inputs:
		material.node_tree.links.new(image_node.outputs["Alpha"], shader.inputs["Alpha"])
		if hasattr(material, "surface_render_method"):
			material.surface_render_method = 'DITHERED'
	layout_image_nodes(material)
	Engine.request_redraw()


class velend_OT_attach_surface_image(bpy.types.Operator):
	bl_idname = "velend.attach_surface_image"
	bl_label = "Attach Surface Image"
	bl_description = "Attach an existing image to this tifxyz surface and its texture-paint slots"
	bl_options = {'REGISTER', 'UNDO'}

	filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
	filter_glob: bpy.props.StringProperty(
		default="*.png;*.tif;*.tiff;*.jpg;*.jpeg;*.exr", options={'HIDDEN', 'SKIP_SAVE'}
	)
	role: bpy.props.EnumProperty(name="Role", items=ROLES, default='PREDICTION')

	@classmethod
	def poll(cls, context):
		return is_surface(context.active_object)

	def invoke(self, context, event):
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		path = os.path.expanduser(bpy.path.abspath(self.filepath))
		try:
			image = bpy.data.images.load(path, check_existing=True)
			attach_image(context.active_object, image, self.role)
		except Exception as error:
			self.report({'ERROR'}, "Could not attach image: %s" % error)
			return {'CANCELLED'}
		return {'FINISHED'}


class velend_OT_new_surface_annotation(bpy.types.Operator):
	bl_idname = "velend.new_surface_annotation"
	bl_label = "New Paint Annotation"
	bl_description = "Create a transparent external PNG and attach it as a texture-paint slot"
	bl_options = {'REGISTER', 'UNDO'}

	filepath: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
	filter_glob: bpy.props.StringProperty(default="*.png", options={'HIDDEN', 'SKIP_SAVE'})

	@classmethod
	def poll(cls, context):
		return is_surface(context.active_object)

	def invoke(self, context, event):
		obj = context.active_object
		self.filepath = os.path.join(bpy.path.abspath("//"), obj.name + "-annotation.png")
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		obj = context.active_object
		path = os.path.expanduser(bpy.path.abspath(self.filepath))
		if not path.lower().endswith(".png"):
			path += ".png"
		material = active_material(obj)
		node = selected_node(material)
		if node is not None and node.image is not None:
			width, height = node.image.size[:]
		else:
			height, width = tuple(obj.get(
				"velend_tifxyz_source_shape", obj.get("velend_tifxyz_shape", (1, 1))
			))
		try:
			image = bpy.data.images.new(
				os.path.basename(path), width=max(1, width), height=max(1, height), alpha=True
			)
			image.generated_color = (0.0, 0.0, 0.0, 0.0)
			image.filepath_raw = path
			image.file_format = 'PNG'
			image.save()
			attach_image(obj, image, 'ANNOTATION')
		except Exception as error:
			self.report({'ERROR'}, "Could not create annotation: %s" % error)
			return {'CANCELLED'}
		return {'FINISHED'}


class velend_OT_use_surface_image(bpy.types.Operator):
	bl_idname = "velend.use_surface_image"
	bl_label = "Use for Render"
	bl_description = "Connect the selected surface image to Principled Base Color and Alpha"
	bl_options = {'REGISTER', 'UNDO'}

	@classmethod
	def poll(cls, context):
		material = active_material(context.active_object)
		return selected_node(material) is not None

	def execute(self, context):
		material = active_material(context.active_object)
		connect_for_render(material, selected_node(material))
		return {'FINISHED'}


class velend_OT_save_surface_image(bpy.types.Operator):
	bl_idname = "velend.save_surface_image"
	bl_label = "Save Surface Image"
	bl_description = "Save the selected surface image to its external file"

	@classmethod
	def poll(cls, context):
		node = selected_node(active_material(context.active_object))
		return node is not None and node.image is not None and bool(node.image.filepath)

	def execute(self, context):
		image = selected_node(active_material(context.active_object)).image
		try:
			image.save()
		except Exception as error:
			self.report({'ERROR'}, "Could not save image: %s" % error)
			return {'CANCELLED'}
		return {'FINISHED'}


class velend_OT_select_surface_image(bpy.types.Operator):
	bl_idname = "velend.select_surface_image"
	bl_label = "Select Surface Image"
	bl_options = {'INTERNAL'}

	node_name: bpy.props.StringProperty(options={'HIDDEN'})

	def execute(self, context):
		material = active_material(context.active_object)
		if material is None or not material.use_nodes:
			return {'CANCELLED'}
		node = material.node_tree.nodes.get(self.node_name)
		if node is None:
			return {'CANCELLED'}
		for other in material.node_tree.nodes:
			other.select = False
		node.select = True
		material.node_tree.nodes.active = node
		# Blender derives the active texture-paint slot from the active image node.
		return {'FINISHED'}


class MATERIAL_PT_velend_surface_images(bpy.types.Panel):
	bl_label = "Velend Surface Images"
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "material"

	@classmethod
	def poll(cls, context):
		return is_surface(context.active_object)

	def draw(self, context):
		obj = context.active_object
		material = active_material(obj)
		layout = self.layout
		if material is None:
			layout.operator("velend.attach_surface_image", icon='FILE_IMAGE')
			layout.operator("velend.new_surface_annotation", icon='ADD')
			return
		column = layout.column()
		column.use_property_split = True
		column.prop(material, "velend_render_mode")
		if material.velend_render_mode == 'OVERLAY':
			column.prop(material, "velend_texture_opacity")
		image, detail, _node = resolved_image(material)
		if material.velend_render_mode in {'TEXTURE', 'OVERLAY'} and image is None:
			layout.label(text=detail, icon='ERROR')
		layout.separator()
		buttons = layout.row(align=True)
		buttons.operator("velend.attach_surface_image", icon='FILE_IMAGE')
		buttons.operator("velend.new_surface_annotation", icon='BRUSH_DATA')
		layout.separator()
		active = selected_node(material)
		for node in managed_nodes(material):
			row = layout.row(align=True)
			op = row.operator(
				"velend.select_surface_image", text=node.label or node.name,
				icon='RADIOBUT_ON' if node == active else 'RADIOBUT_OFF',
			)
			op.node_name = node.name
			row.label(text=node.get(IMAGE_ROLE, 'OTHER').title())
		use = layout.row()
		use.enabled = active is not None
		use.operator("velend.use_surface_image", icon='NODE_MATERIAL')
		if active is not None and active.image is not None:
			row = layout.row()
			row.enabled = bool(active.image.filepath)
			row.operator(
				"velend.save_surface_image",
				text="Save Image" + (" *" if active.image.is_dirty else ""),
				icon='FILE_TICK',
			)


_CLASSES = (
	velend_OT_attach_surface_image,
	velend_OT_new_surface_annotation,
	velend_OT_use_surface_image,
	velend_OT_save_surface_image,
	velend_OT_select_surface_image,
	MATERIAL_PT_velend_surface_images,
)


def register():
	for cls in _CLASSES:
		bpy.utils.register_class(cls)
	bpy.types.Material.velend_render_mode = bpy.props.EnumProperty(
		name="Surface Source",
		items=MODES,
		default='AUTO',
		update=lambda self, context: Engine.request_redraw(),
	)
	bpy.types.Material.velend_texture_opacity = bpy.props.FloatProperty(
		name="Image Opacity", min=0.0, max=1.0, default=1.0,
		update=lambda self, context: Engine.request_redraw(),
	)
	if not bpy.app.timers.is_registered(_migrate_when_registered):
		bpy.app.timers.register(_migrate_when_registered, first_interval=0.0)
	if _migrate_image_layout not in bpy.app.handlers.load_post:
		bpy.app.handlers.load_post.append(_migrate_image_layout)


def unregister():
	if bpy.app.timers.is_registered(_migrate_when_registered):
		bpy.app.timers.unregister(_migrate_when_registered)
	if _migrate_image_layout in bpy.app.handlers.load_post:
		bpy.app.handlers.load_post.remove(_migrate_image_layout)
	del bpy.types.Material.velend_texture_opacity
	del bpy.types.Material.velend_render_mode
	for cls in reversed(_CLASSES):
		bpy.utils.unregister_class(cls)
