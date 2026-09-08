void main() {
  vec4 worldPosition = volumeUniforms.modelMatrix * vec4(position, 1.0f);
  // Level 0 voxel space is the common coordinate system; coarser sources scale
  // this coordinate before looking up their textures.
  voxelCoord = worldPosition.xyz * volumeUniforms.voxelsPerUnit;
  gl_Position = volumeUniforms.viewProjectionMatrix * worldPosition;
}
