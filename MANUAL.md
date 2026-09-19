# velend manual

What the extension does once a volume is set up. See the [README](README.md) for installing it and pointing it at a scan.


## The download budget

Downloads into VC3D's cache stay within the budget its settings put on it, under Preferences > Performance: the maximum it may grow to and the free space it must leave the disk. Reaching either stops the downloads, with the reason in the panel, rather than filling the disk. Nothing is ever deleted here: VC3D evicts the least recently read chunk when it needs room, counting ours along with its own, and a read served from the cache marks the chunk recent so that what you are working on is not the first thing it throws away. A mirror directory of your own, outside VC3D's cache, is held only to the free space floor.


## Segments

File > Import > Volume Cartographer Surface (tifxyz) brings a segmentation in as a mesh. Point it at a surface directory, the one holding `x.tif`, `y.tif`, `z.tif` and `meta.json`, or at a folder of them such as VC3D's `patches/`, and every surface under it comes in at once.

The surface's coordinate grid becomes a quad mesh with a UV map. A transparent placeholder image on its material gives Blender's UV Editor the grid's aspect ratio without loading another full-size image. For surfaces imported with an older Velend version, select them and run Object > Set Up tifxyz UV Aspect, also available from F3 search. The command infers the dimensions from the UV lattice, just as export does, and does not require import metadata. Its `mask.tif` decides which grid vertices and faces are created; it is not retained as a mesh attribute. To change the mask, delete vertices or faces from the mesh.

Velend flips both tifxyz grid axes in the UV map. It flips V because Blender's V
increases upward, while Volume Cartographer's tifxyz V follows image rows downward. It
flips U because spiral-fitting writes increasing theta from left to right, opposite the
reading direction: text is read left to right from the outside toward the inside of the
scroll, in decreasing theta. Export reverses both flips, so an unchanged surface round
trips without changing its tifxyz grid order.

To repair an accidental hole, select the mesh and run Mesh > Fill tifxyz Holes in Edit Mode, or search for the same command with F3 in either Edit or Object Mode. It restores enclosed missing UV-grid quads and smoothly interpolates missing 3D positions and float point attributes. Existing vertices stay fixed; open edges and gaps reaching the outer UV boundary are left alone. The command is undoable.

File > Export > Volume Cartographer Surface exports a quad mesh whose surviving face vertices lie on a rectangular UV lattice. It crops to their smallest UV rectangle, writes holes inside it with mask 0 and invalid coordinates, and ignores orphan vertices that have no UV face corners. Object transforms are included. Apply or remove modifiers before exporting. Imported metadata and placement are reused, while a new mesh starts with editable UUID, grid scale, and voxel-size defaults. Public FLOAT point attributes become extra TIFF channels.

Enable "Use Current Volume Coordinates" to export the geometry in the voxel coordinates of the
volume currently being rendered. This uses its registration with the scene volume when one exists;
otherwise the volumes are taken to share an origin and axes. It is off by default, preserving an
imported surface's original coordinate space.

Raise "Step" to bring a large segment in coarser, one grid point in every n. "Coordinates" says whether the tifxyz is in the scene volume's voxels or the volume currently being rendered. It starts at Scene Volume, so surfaces traced in the scan the scene was set up against stay in that frame after switching scans. Choose Rendered Volume for a surface traced in the currently rendered scan. "Voxel Size" starts from the chosen volume and can be corrected independently.


## Umbilici

File > Import > Umbilicus brings an `umbilicus.json`, or the `z, y, x` text form of it, in as a polyline running up the scroll's core. Like tifxyz import, its "Coordinates" choice defaults to the scene volume rather than the volume currently being rendered. "Voxel Size" starts from that choice; a size stated by the file itself wins.

File > Export > Scroll Umbilicus writes the active mesh as `umbilicus.json`, with `control_points` containing `x`, `y`, and `z` objects and a `score` for vertices that have that point attribute. The mesh must be one open line of at least two vertices joined by edges, with no branches, loose vertices, cycles, or faces. Its path must run upward in the volume's Z coordinate, since Volume Cartographer sorts umbilicus points by Z when reading them. Object transforms are included; apply or remove modifiers first. An imported umbilicus keeps its original coordinate placement and other metadata by default. Export stamps `timestamp`, `created`, and `modified` with the current UTC time, sets `source_volume` to the scene volume, and records Velend in `annotator_note`. For a new mesh, choose the volume coordinate frame and voxel size in the export dialog. Replacing an existing file requires enabling "Replace Existing File".

### Recommended workflow to trace umbilici (about 10 min on PHerc 0175A)

1. Setup the scene.
2. Move the "Cut Z" plane all the way down to where you first see the umbilicus.
3. `shift+d` to duplicate, `z` for the z direction and `5` mm or whatever distance between samples you need to make a good umbilicus. `enter`. `shift+r` to repeat that action until you cover all the z you need for the scroll.
4. Split the viewport and align one view to the xy plane in rendered mode, and a side view plane in wireframe mode. Hide all the cut planes except the bottom most.
5. `shift+a` > create a plane. Go into edit mode (`tab`). Delete 3 of the 4 vertices. `e` to extrude the remaining vertex, `z`, `5` to do extrude it 5 mm up z. `shift+r` to repeat until covering the z you need.
6. Now do this in a loop:
  - select the next vertex from the bottom in the side view.
  - hit `g` on the top view to move it, and drag it to where you see the umbilicus. `enter`
  - unhide the next cut plane, the one for the next vertex going up
7. Can confirm if it is good enough by hiding all the planes again except one and moving it slowly in `z` while looking in the top view.
8. File > Export > Scroll Umbilicus (umbilicus.json)


## UV volume view

The UV Editor renders the volume on the active mesh's active UV map, including live Edit Mode changes. Select an imported segment and open the UV Editor to see its flattened scan. Shift-right-click a UV face to put the 3D cursor at that point on the mesh and load high-resolution detail around it. The Velend tab in the UV Editor sidebar can turn the volume drawing off or adjust its transparency from 0 (opaque) to 1 (invisible).

This currently uses the editable mesh, before modifiers. Overlapping UV faces overwrite one another. UV edges and vertices are depth-tested against the volume drawing; translucent face-selection and stretch overlays are not preserved by this drawing pass.


## Switching volumes

A sample is often scanned more than once, and the open data metadata registers its volumes against each other: for each ordered pair of them, the matrix that takes a point in one's voxels into the other's. There is no space they all share and no canonical one among them, so it is always a pair.

The scene's coordinates are metric, and sit in the frame of the volume it was set up against. Voxel Size states how wide a voxel of the volume being *rendered* is, which is what places that volume in those coordinates: a scan of 2 um voxels put beside one of 9 um is four and a half times the voxels across the same millimetre. Point the panel at another volume of the same sample and it renders that one through the matrix registered for the pair, so a cutting plane or an imported segment keeps showing the same place in the scroll -- at whatever resolution and orientation the new volume reconstructed it. The panel says which volume the scene is in, and how wide its voxels are, whenever that is no longer the one being rendered.

A volume the metadata cannot relate to the scene's has nothing saying how the two sit against each other, so it is placed on the only assumption there is to make about two scans of one object: same origin, same axes, each at its own voxel size. Nothing already in the scene moves or changes size, and the panel says the placement is a guess. Registering the pair later replaces the guess with the transform, without anything having been re-anchored in the meantime.

Voxel Size is read rather than set: it comes from the metadata for a volume the catalogue lists, or from the volume's directory name when that names it. A volume neither states the size of is asked about when it loads -- the scene cannot place it otherwise -- and the button beside the field asks again, which is also how to correct a size read off a name that lies.

A transform that has not reached the published metadata yet can be supplied through Preferences > Add-ons > velend > Extra Metadata, a JSON file shaped like `metadata.json` and merged over it. A direction it leaves out is taken from the opposite one inverted.


## Rendering

The Volume Sampler engine shades the meshes already in the scene with the scan. Every fragment looks the volume up where its surface sits, averaging a few samples along the face normal to thin out the noise the sheets are embedded in, so a plane through the volume comes out a slice of it and an imported segment comes out the sheet it was traced from. Where nothing is loaded the fragment is dropped rather than painted, which is what a mesh you can see through means.

"Volumetric Rendering", under Render > Volume Sampling, treats those samples as material along a ray: brighter samples absorb more of it and progressively obscure samples behind them. "Transmittance Factor" sets the absorption strength from 0 (none) to 1 (strongest). Turn volumetric rendering off for the ordinary depth-weighted average.

Only what a mesh touches is ever read. A scroll is far larger than any GPU's memory, so the triangles of every visible mesh are walked into a grid of cubic chunks, 64 voxels to a side, and those are the chunks that stream: each one is read from the zarr and uploaded into a slot of a texture atlas the shader finds through a page table. A chunk that turns out to be all zeroes is remembered as empty and never asked for again.

Each level of the pyramid streams through an atlas of its own, with a fixed number of slots. Every level sees the whole set of chunks the meshes cross, collapsed onto its own grid, and keeps as many of those nearest the 3D cursor as it has room for, so the coarsest level covers as much of the mesh as it can reach and each finer one refines a smaller neighbourhood of the cursor. A fragment is shaded from the finest level resident where it falls. The coarsest goes out first, so the mesh has something on it within a few reads, and the finer ones then sharpen it from the cursor outwards. Moving the cursor picks the working set again; "Load High-Res Volume", in the panel and in the View menu, asks for that pass without moving anything.

What falls outside the viewport is not loaded at all: the resolution the streamer can afford goes to what you are looking at, and turning away drops the rest a moment after you stop moving. Chunks are kept when they fall inside any open viewport's frustum, widened by a few chunks so that a small movement does not immediately want what was just dropped. "Frustum Culling", in the Velend panel, turns that off and keeps the whole mesh loaded wherever the view points.

"Level Colors", next to it, tints each fragment by the level its samples came from instead of shading it -- red is full resolution, through to purple for the coarsest -- which is how to see what the streamer has actually put on a mesh.
