import bpy

from .renderer import VolumeSamplerRenderEngine


class VLEND_OT_load_hires(bpy.types.Operator):
	bl_idname = "vlend.load_hires"
	bl_label = "Load High-Res Volume"
	bl_description = "Load the high-resolution volume around the 3D cursor into the 3D texture"
	bl_options = {'REGISTER'}

	def execute(self, context):
		VolumeSamplerRenderEngine.load_hires_at(context.scene.cursor.location)
		return {'FINISHED'}


def _view_menu(self, context):
	self.layout.operator(VLEND_OT_load_hires.bl_idname)


def register():
	bpy.utils.register_class(VLEND_OT_load_hires)
	bpy.types.VIEW3D_MT_view.append(_view_menu)


def unregister():
	bpy.types.VIEW3D_MT_view.remove(_view_menu)
	bpy.utils.unregister_class(VLEND_OT_load_hires)
