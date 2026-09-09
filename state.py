# Unlike the other modules, this one isn't reloaded by Reload Scripts.
import zarr
from .mirror import MirrorStore

volume = None
volume_path = None
volume_source_url = None
online_access = True


def open_volume(path, source_url=""):
    """Open an OME-Zarr pyramid, finest first. Arrays retain their store lease."""
    store = MirrorStore(path, source_url, online=lambda: online_access) if source_url else path
    group = zarr.open_group(store, mode="r")
    datasets = group.attrs["multiscales"][0]["datasets"]
    return [group[dataset["path"]] for dataset in datasets]


def get_volume(path, source_url=""):
    global volume, volume_path, volume_source_url
    if volume is None or (path, source_url) != (volume_path, volume_source_url):
        opened = open_volume(path, source_url)
        volume, volume_path, volume_source_url = opened, path, source_url
    return volume


def close_volume():
    global volume, volume_path, volume_source_url
    volume = volume_path = volume_source_url = None
