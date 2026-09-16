# v0.6.0 (main, unreleased)

# v0.5.0

- Tifxyz and umbilicus imports now default to coordinates in the scene's
  original volume, with an option to interpret them in the currently rendered
  volume instead.
- Shift-right-click in the UV Editor now moves the 3D cursor and retargets
  high-resolution volume loading in Object Mode as well as Edit Mode.
- Imported tifxyz surfaces carry a lightweight aspect image so Blender's UV
  Editor displays their rectangular grids without stretching them square.
- Set Up tifxyz UV Aspect applies that display fix to surfaces imported earlier.

# v0.4.0

- Volume sampling follows mesh normals, independent of which side is viewed.
- Imported tifxyz surfaces have correctly oriented face normals and UV orientation.
- Tifxyz masks determine which vertices and faces are imported; they are not
  retained as mesh attributes. Delete mesh vertices or faces to edit validity.
- File > Export > Volume Cartographer Surface writes rectangular UV quad grids,
  metadata, and FLOAT point-attribute channels back to tifxyz. It crops to the
  remaining UV bounds and writes holes from deleted geometry into `mask.tif`.
- Volume switching. Point the panel at another volume of the same sample and it
  renders that one through the transform the open data metadata registers for
  the pair, so cutting planes and imported segments keep showing the same place
  in the scroll. A volume the metadata cannot relate to the scene's is placed at
  its own voxel size, sharing an origin and its axes with the scene's, which
  leaves everything already placed where it is; the panel says the placement is
  a guess. Preferences > Add-ons > velend > Extra Metadata supplies a transform
  the published metadata does not have yet.
- Voxel Size states the volume being rendered rather than the scene's own frame,
  and is shown rather than set: read off the metadata or the volume's directory
  name, and asked for when neither states it.


# v0.3.0

- Volume switching. Point the panel at another volume of the same sample and it
  renders that one through the transform the open data metadata registers for
  the pair, so cutting planes and imported segments keep showing the same place
  in the scroll. A volume the metadata cannot relate to the scene's becomes the
  scene's own frame instead, and the panel says so. Preferences > Add-ons >
  velend > Extra Metadata supplies a transform the published metadata does not
  have yet.
- Frustum culling. Only the bricks inside a viewport's view frustum are
  streamed, so the resolution the streamer can afford goes to what is on
  screen. "Frustum Culling", in the panel, turns it off.
- "Choose from volpkg.json" fills the volume, source URL and voxel size in from
  a VC3D project, through the cache directory VC3D itself reads that volume
  with, so either program's downloads count for the other.
- Downloads into that cache stay within the budget VC3D's settings put on it,
  and a mirror directory of your own within the free space it must leave the
  disk. Reaching either stops the downloads, with the reason in the panel,
  rather than filling the disk. A read marks its chunk recent, so VC3D's
  eviction does not start with what you are working on.
- File > Import > Umbilicus brings an `umbilicus.json`, or the `z, y, x` text
  form of it, in as a polyline up the scroll's core.
- Rendering reworked. The coarsest level goes out as soon as a mesh is there,
  so it has something on it right away, and the finer ones sharpen it from the
  3D cursor outwards.
- "Level Colors", in the panel, tints each fragment by the resolution level its
  samples came from, to show what the streamer has loaded.
- `scripts/release.py` cuts a release: sets the version in the manifest, builds
  the zips, and tags the commit they were built from.


# v0.2.0

- Mirror downloads. Source URL names the public OME-Zarr that the local Volume
  directory mirrors, and the metadata and chunks missing from it download as
  the renderer asks for them, byte for byte as VC3D writes them. An empty
  directory is enough to start from.
- File > Import > Volume Cartographer Surface (tifxyz) brings a segmentation in
  as a quad mesh -- a whole folder of them at once -- with the cells its mask
  takes out left out, a UV map over the grid, and extra channels such as
  `generations.tif` as mesh attributes.
- The UV Editor renders the volume on the active mesh's active UV map,
  following Edit Mode changes live.
- The open data catalogue is fetched and cached at startup: which samples,
  scans, volumes and segments exist.


# v0.1.0

- The Volume Sampler render engine draws the scan on the meshes in the scene,
  each fragment sampling the volume where its surface sits.
- Chunks stream into a texture atlas around the 3D cursor at several resolution
  levels, the coarser standing in until the finer one arrives.
- The Scene > Velend panel: the volume directory and its voxel size, "Setup
  Scene for Volume", which sets the units, the render engine and the shading
  and adds a cutting plane per axis, and "Load High-Res Volume".
- Reads whatever the local OME-Zarr already holds; nothing is downloaded.
- `scripts/build.py` builds the per platform zips with their wheels, and
  `scripts/develop.py` links a checkout into Blender to develop against.
