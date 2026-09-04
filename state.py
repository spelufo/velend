# Unlike the other modules, this one isn't reloaded when running "Reload Scripts".

from vesuvius import Volume


volpath = "/Users/spelufo/.VC3D/remote_cache/open_data/volumes/PHerc1203/20250820131727-9.362um-1.2m-113keV-masked.zarr-adf63bbdf658dd8f"
volume = None

def get_volume():
	global volume
	if volume is None:
		volume = Volume(type="zarr", path=volpath, return_as_type='float32')
		volume.resolution = 9.362
	return volume
