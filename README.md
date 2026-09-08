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

TODO: Make this easy, not my custom setup.
- hardcoded zar path goes away
- user picks .volpkg.json and there's ui to pick a volume, or sth
- autodownload? it will read whatever you have in the volume
- Blender files must have:
  - Unit: mm
  - Unit scale: 0.001
  - Renderer: Volume Sampler
- TODO: Init some axis planes when choosing a scan


### Recommended blender preferences and key mapping

Choose "Use spacebar to search commands".

Navigation changes:
- Turn on "Use the depth under the mouse to improve orbit/rotate".
- alt + left click -> orbit (view.rotate3d)
- alt + right click -> pan
- scroll wheel -> zoom


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


## Development

`blender` has to be on PATH. Then:

    scripts/develop.py

which downloads the dependency wheels and links this checkout into Blender's
user extensions directory, so that "Reload Scripts" (F3 > Reload Scripts) picks
up edits without a reinstall. `scripts/develop.py --unlink` undoes the link.

The extension watches for shader changes and reloads them automatically.
