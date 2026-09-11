void main() {
  vec4 worldPosition = volumeUniforms.modelMatrix * vec4(position, 1.0f);
  // The scene's coordinates are in the voxels of the volume it was set up
  // against; `volumeTransform` takes those into the level 0 voxels of the one
  // being rendered, which is the common coordinate system from here on.
  // Coarser sources scale this coordinate before looking up their textures.
  voxelCoord = (volumeUniforms.volumeTransform *
      vec4(worldPosition.xyz * volumeUniforms.voxelsPerUnit, 1.0f)).xyz;
  gl_Position = volumeUniforms.viewProjectionMatrix * worldPosition;
}
