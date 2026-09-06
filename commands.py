import bpy

from .renderer import VolumeSamplerRenderEngine


class VLEND_OT_load_hires(bpy.types.Operator):
	bl_idname = "vlend.load_hires"
	bl_label = "Load High-Res Volume"
	bl_description = (
		"Stream the high-resolution bricks nearest the 3D cursor into the brick atlas"
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
	self.layout.operator(VLEND_OT_load_hires.bl_idname)


def register():
	bpy.utils.register_class(VLEND_OT_load_hires)
	bpy.types.VIEW3D_MT_view.append(_view_menu)


def unregister():
	bpy.types.VIEW3D_MT_view.remove(_view_menu)
	bpy.utils.unregister_class(VLEND_OT_load_hires)
