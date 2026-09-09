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

Point the extension at a scan in the Properties editor, under Scene > Velend:

- Volume: the local OME-Zarr directory of the scan, or an empty directory to
  use as a download cache.
- Source URL (optional): the public HTTP(S) root of that same OME-Zarr dataset.
  Missing metadata and chunks download into Volume. Leave it blank to retain
  the existing behavior without mirror downloads.
- Voxel Size: how wide a full resolution voxel is, in micrometers. It is filled
  in from the volume's directory name when that names it, as in
  `20250820131727-9.362um-1.2m-113keV-masked.zarr`.

The URL and local directory must identify the same dataset. "Choose from
volpkg.json" fills the three fields in from a VC3D project: pick a
`.volpkg.json`, then one of the volumes it lists. The volume's zarr path
becomes the cache directory VC3D reads it through, under
`~/.VC3D/remote_cache/open_data/volumes`, and its source URL the one VC3D
fetches it from. It then works the same way VC3D does, downloading chunks to
the same local zarr when missing, so either program's downloads count for the
other.

The file browser opens in `~/.VC3D/remote_cache/open_data/projects`, where VC3D
puts the open data projects it downloads; a project of your own works as well.

Then hit "Setup Scene for Volume", which waits for loading and sets the rest up:
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


### Segments

File > Import > Volume Cartographer Surface (tifxyz) brings a segmentation in
as a mesh. Point it at a surface directory, the one holding `x.tif`, `y.tif`,
`z.tif` and `meta.json`, or at a folder of them such as VC3D's `patches/`, and
every surface under it comes in at once.

The surface's grid becomes a quad per cell, minus the cells its `mask.tif`
takes out, with a UV map over the grid and any extra channel like
`generations.tif` as a mesh attribute. Raise "Step" to bring a large segment in
coarser, one grid point in every n. "Voxel Size" says what the surface's
coordinates are in, and starts from the scene's own, so a segment lands inside
the volume it was traced from and "Load High-Res Volume" renders the scan on it.


### UV volume view

The UV Editor renders the volume on the active mesh's active UV map, including
live Edit Mode changes. Select an imported segment and open the UV Editor to
see its flattened scan. The view shares the 3D renderer's volume textures:
high-resolution detail still follows the 3D cursor, not UV panning or zooming.

This currently uses the editable mesh, before modifiers. Overlapping UV faces
overwrite one another. UV edges and vertices are depth-tested against the
volume drawing; translucent face-selection and stretch overlays are not
preserved by this drawing pass.


## Development

`blender` has to be on PATH. Then:

    scripts/develop.py

which downloads the dependency wheels and links this checkout into Blender's
user extensions directory, so that "Reload Scripts" (F3 > Reload Scripts) picks
up edits without a reinstall. `scripts/develop.py --unlink` undoes the link.

The extension watches for shader changes and reloads them automatically.
