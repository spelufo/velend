import bpy

from .renderer import VolumeSamplerRenderEngine


class velend_OT_load_hires(bpy.types.Operator):
	bl_idname = "velend.load_hires"
	bl_label = "Load High-Res Volume"
	bl_description = (
		"Stream multiresolution volume bricks around meshes nearest the 3D cursor"
	)
	bl_options = {'REGISTER'}

	def execute(self, context):
		depsgraph = context.evaluated_depsgraph_get()
		count = VolumeSamplerRenderEngine.retarget(depsgraph, context.scene.cursor.location)
		if count == 0:
			self.report({'WARNING'}, "No mesh geometry near the 3D cursor")
		else:
			self.report({'INFO'}, "Streaming %d bricks" % count)
		return {'FINISHED'}


def _view_menu(self, context):
	self.layout.operator(velend_OT_load_hires.bl_idname)


def register():
	bpy.utils.register_class(velend_OT_load_hires)
	bpy.types.VIEW3D_MT_view.append(_view_menu)


def unregister():
	bpy.types.VIEW3D_MT_view.remove(_view_menu)
	bpy.utils.unregister_class(velend_OT_load_hires)
