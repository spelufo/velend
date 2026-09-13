# velend manual

What the extension does once a volume is set up. See the [README](README.md) for installing it and pointing it at a scan.


## The download budget

Downloads into VC3D's cache stay within the budget its settings put on it, under Preferences > Performance: the maximum it may grow to and the free space it must leave the disk. Reaching either stops the downloads, with the reason in the panel, rather than filling the disk. Nothing is ever deleted here: VC3D evicts the least recently read chunk when it needs room, counting ours along with its own, and a read served from the cache marks the chunk recent so that what you are working on is not the first thing it throws away. A mirror directory of your own, outside VC3D's cache, is held only to the free space floor.


## Segments

File > Import > Volume Cartographer Surface (tifxyz) brings a segmentation in as a mesh. Point it at a surface directory, the one holding `x.tif`, `y.tif`, `z.tif` and `meta.json`, or at a folder of them such as VC3D's `patches/`, and every surface under it comes in at once.

The surface's grid becomes a quad per cell, minus the cells its `mask.tif` takes out, with a UV map over the grid and any extra channel like `generations.tif` as a mesh attribute. Raise "Step" to bring a large segment in coarser, one grid point in every n. "Voxel Size" says what the surface's coordinates are in, and starts from the voxel size of the volume being rendered, that being the likeliest one they were traced against, so a segment lands inside the volume it came from and "Load High-Res Volume" renders the scan on it.


## Umbilici

File > Import > Umbilicus brings an `umbilicus.json`, or the `z, y, x` text form of it, in as a polyline running up the scroll's core. Its points are in full resolution voxels, and "Voxel Size" says whose: the volume being rendered, unless the file states one of its own, which wins.


## UV volume view

The UV Editor renders the volume on the active mesh's active UV map, including live Edit Mode changes. Select an imported segment and open the UV Editor to see its flattened scan. The view shares the 3D renderer's volume textures: high-resolution detail still follows the 3D cursor, not UV panning or zooming.

This currently uses the editable mesh, before modifiers. Overlapping UV faces overwrite one another. UV edges and vertices are depth-tested against the volume drawing; translucent face-selection and stretch overlays are not preserved by this drawing pass.


## Switching volumes

A sample is often scanned more than once, and the open data metadata registers its volumes against each other: for each ordered pair of them, the matrix that takes a point in one's voxels into the other's. There is no space they all share and no canonical one among them, so it is always a pair.

The scene's coordinates are metric, and sit in the frame of the volume it was set up against. Voxel Size states how wide a voxel of the volume being *rendered* is, which is what places that volume in those coordinates: a scan of 2 um voxels put beside one of 9 um is four and a half times the voxels across the same millimetre. Point the panel at another volume of the same sample and it renders that one through the matrix registered for the pair, so a cutting plane or an imported segment keeps showing the same place in the scroll -- at whatever resolution and orientation the new volume reconstructed it. The panel says which volume the scene is in, and how wide its voxels are, whenever that is no longer the one being rendered.

A volume the metadata cannot relate to the scene's has nothing saying how the two sit against each other, so it is placed on the only assumption there is to make about two scans of one object: same origin, same axes, each at its own voxel size. Nothing already in the scene moves or changes size, and the panel says the placement is a guess. Registering the pair later replaces the guess with the transform, without anything having been re-anchored in the meantime.

Voxel Size is read rather than set: it comes from the metadata for a volume the catalogue lists, or from the volume's directory name when that names it. A volume neither states the size of is asked about when it loads -- the scene cannot place it otherwise -- and the button beside the field asks again, which is also how to correct a size read off a name that lies.

A transform that has not reached the published metadata yet can be supplied through Preferences > Add-ons > velend > Extra Metadata, a JSON file shaped like `metadata.json` and merged over it. A direction it leaves out is taken from the opposite one inverted.


## Rendering

The Volume Sampler engine shades the meshes already in the scene with the scan. Every fragment looks the volume up where its surface sits, averaging a few samples along the face normal to thin out the noise the sheets are embedded in, so a plane through the volume comes out a slice of it and an imported segment comes out the sheet it was traced from. Where nothing is loaded the fragment is dropped rather than painted, which is what a mesh you can see through means.

"Volumetric Rendering", under Render > Volume Sampling, treats those samples as material along a ray: brighter samples absorb more of it and progressively obscure samples behind them. Turn it off for the ordinary depth-weighted average.

Only what a mesh touches is ever read. A scroll is far larger than any GPU's memory, so the triangles of every visible mesh are walked into a grid of cubic chunks, 64 voxels to a side, and those are the chunks that stream: each one is read from the zarr and uploaded into a slot of a texture atlas the shader finds through a page table. A chunk that turns out to be all zeroes is remembered as empty and never asked for again.

Each level of the pyramid streams through an atlas of its own, with a fixed number of slots. Every level sees the whole set of chunks the meshes cross, collapsed onto its own grid, and keeps as many of those nearest the 3D cursor as it has room for, so the coarsest level covers as much of the mesh as it can reach and each finer one refines a smaller neighbourhood of the cursor. A fragment is shaded from the finest level resident where it falls. The coarsest goes out first, so the mesh has something on it within a few reads, and the finer ones then sharpen it from the cursor outwards. Moving the cursor picks the working set again; "Load High-Res Volume", in the panel and in the View menu, asks for that pass without moving anything.

What falls outside the viewport is not loaded at all: the resolution the streamer can afford goes to what you are looking at, and turning away drops the rest a moment after you stop moving. Chunks are kept when they fall inside any open viewport's frustum, widened by a few chunks so that a small movement does not immediately want what was just dropped. "Frustum Culling", in the Velend panel, turns that off and keeps the whole mesh loaded wherever the view points.

"Level Colors", next to it, tints each fragment by the level its samples came from instead of shading it -- red is full resolution, through to purple for the coarsest -- which is how to see what the streamer has actually put on a mesh.
