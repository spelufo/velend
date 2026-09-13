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
- Voxel Size: how wide a full resolution voxel of that volume is, in
  micrometers. Shown rather than set: it is read off the open data metadata, or
  the volume's directory name when that names it, as in
  `20250820131727-9.362um-1.2m-113keV-masked.zarr`. A volume neither of them
  states the size of is asked about, and the button beside the field is how to
  correct either of them.

The URL and local directory must identify the same dataset. "Choose from volpkg.json" fills the three fields in from a VC3D project: pick a `.volpkg.json`, then one of the volumes it lists. The volume's zarr path becomes the cache directory VC3D reads it through, under `~/.VC3D/remote_cache/open_data/volumes`, and its source URL the one VC3D fetches it from. It then works the same way VC3D does, downloading chunks to the same local zarr when missing, so either program's downloads count for the other.

The file browser opens in `~/.VC3D/remote_cache/open_data/projects`, where VC3D puts the open data projects it downloads; a project of your own works as well.

Downloads are kept within the storage budget VC3D's own settings put on its cache, and stop with the reason in the panel rather than filling the disk. [MANUAL.md](MANUAL.md) says how that is counted.

Then hit "Setup Scene for Volume", which waits for loading and sets the rest up:
- Scene > Units > Unit: mm
- Scene > Units > Unit scale: 0.001
- Render > Render Engine: Volume Sampler
- Viewport Shading: Rendered
- Viewport Overlays > Grid > Scale: 0.001

It also adds a plane per axis through the middle of the volume, and puts the 3D cursor there, which is where bricks stream in around.

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

- Disable the sculpt mode shortcuts that interfere with above:
  - Set pivot position
  - alt+lmb

## Development

`blender` has to be on PATH. Then:

    scripts/develop.py

which downloads the dependency wheels and links this checkout into Blender's user extensions directory, so that "Reload Scripts" (F3 > Reload Scripts) picks up edits without a reinstall. `scripts/develop.py --unlink` undoes the link.

The extension watches for shader changes and reloads them automatically.
