void main() {
  vec4 worldPosition = volumeUniforms.modelMatrix * vec4(position, 1.0f);
  // `worldToVoxels` takes the scene's metric coordinates into the level 0
  // voxels of the volume being rendered, which is the common coordinate system
  // from here on: the scene's units into micrometers, the registration between
  // the frame the scene is in and this volume, and this volume's voxel size.
  // Coarser sources scale this coordinate before looking up their textures.
  voxelCoord = (volumeUniforms.worldToVoxels * vec4(worldPosition.xyz, 1.0f)).xyz;
  gl_Position = volumeUniforms.viewProjectionMatrix * worldPosition;
}
