void main() {
  vec4 worldPosition = volumeUniforms.modelMatrix * vec4(position, 1.0f);
  // Level 0 voxel space is the one coordinate system both the brick atlas and
  // the low resolution volume are addressed in.
  voxelCoord = worldPosition.xyz * volumeUniforms.voxelsPerUnit;
  gl_Position = volumeUniforms.viewProjectionMatrix * worldPosition;
}
