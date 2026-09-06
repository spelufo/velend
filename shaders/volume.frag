void main() {
  float gamma = 0.5;

  // Which chunk of the volume this fragment falls in, and where the page table
  // says that chunk currently lives in the atlas. Slot 0 means "not resident".
  ivec3 chunk = ivec3(floor(voxelCoord / float(BRICK_CORE)));
  int slot = -1;
  if (all(greaterThanEqual(chunk, ivec3(0))) && all(lessThan(chunk, PAGE_DIMS))) {
    slot = int(texelFetch(pageTable, chunk, 0).r + 0.5f) - 1;
  }

  float raw;
  if (slot >= 0) {
    ivec3 slotCoord = ivec3(
      slot % SLOTS_PER_AXIS,
      (slot / SLOTS_PER_AXIS) % SLOTS_PER_AXIS,
      slot / (SLOTS_PER_AXIS * SLOTS_PER_AXIS));
    // Texel `BRICK_PAD` of a brick is the first voxel of its core, so a sample
    // on a core face lands exactly between the same two voxels whichever of the
    // two neighbouring bricks it is taken from. That is what hides the seams.
    vec3 local = clamp(voxelCoord - vec3(chunk) * float(BRICK_CORE), 0.0f, float(BRICK_CORE));
    vec3 texel = vec3(slotCoord * BRICK_SIZE) + float(BRICK_PAD) + local;
    raw = texture(atlas, texel / float(ATLAS_DIM)).r;
  } else {
    raw = texture(volume, voxelCoord / volumeUniforms.loresExtent).r;
  }

  float intensity = pow(raw, 1/gamma);
  // Viewport overlay composites as `render.rgb + background * (1 - render.a)`.
  FragColor = vec4(vec3(intensity), 1.0);
}
