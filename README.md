# velend

A Blender extension for exploring the X-ray scans from the [Vesuvius Challenge](https://scrollprize.org).


## Getting started

### Installation

Download the zip for your platform from the
[latest release](../../releases/latest), then in Blender:

- Edit > Preferences > Add-ons
- The dropdown at the top right > Install from Disk...
- Pick the zip, and tick the checkbox next to "velend" to enable it.


### Setup

NOTE: For the moment velend won't download volume chunks on the fly like VC3D.
It will render with whatever chunks are present in the zarr. If you've opened a volume in
VC3D before, then your omezar will at least have the L5 most downsampled data.
Use something like [vc_zarr_download_region](https://github.com/ScrollPrize/villa/pull/1707)
to download a region of an omezar at all levels of detail.

Point the extension at a scan in the Properties editor, under Scene > Velend:

- Volume: the OME-Zarr directory of the scan.
- Voxel Size: how wide a full resolution voxel is, in micrometers. It is filled
  in from the volume's directory name when that names it, as in
  `20250820131727-9.362um-1.2m-113keV-masked.zarr`.

Then hit "Setup Scene for Volume", which sets the rest up:
- Scene > Units > Unit: mm
- Scene > Units > Unit scale: 0.001
- Render > Render Engine: Volume Sampler
- Viewport Shading: Rendered
- Viewport Overlays > Grid > Scale: 0.001

It also adds a plane per axis through the middle of the volume, and puts the 3D
cursor there, which is where bricks stream in around.

Hit "space" to search for commands and search for "Frame selected".
Hit "z" and choose "Rendered" to show the scroll scan on the cutting planes.


### Blender crash course

- `shift + rmb` to place the 3d cursor
- `shift + a` to add an object, e.g. a plane
- `s` to scale
- `r` to rotate
- `g` to move
- `x/y/z` while scaling/rotating/moving to constraint to an axis
- press the same axis key twice to do it along the object's axes (e.g. `g` `z` `z` to move a plane along the normal)
- `shift + x/y/z` to constraint to the other two axis
- `shift + d` to duplicate an object
- `n` to see panel with object coordinates, etc


### Recommended preferences

- Choose "Use spacebar to search commands".
- Turn on "Use the depth under the mouse to improve orbit/rotate".
- `alt + lmb` -> view.rotate3d (default is middle mouse press which destroys mouses)
- `alt + rmb` -> pan
- `scroll wheel` -> zoom


## Development

`blender` has to be on PATH. Then:

    scripts/develop.py

which downloads the dependency wheels and links this checkout into Blender's
user extensions directory, so that "Reload Scripts" (F3 > Reload Scripts) picks
up edits without a reinstall. `scripts/develop.py --unlink` undoes the link.

The extension watches for shader changes and reloads them automatically.
