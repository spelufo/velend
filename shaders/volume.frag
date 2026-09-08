// Samples averaged along the normal, and their spacing in level 0 voxels.
// Averaging across the face thins out the noise the surface is embedded in.
#define NORMAL_SAMPLES 5
const float SAMPLE_DELTA = 1.0f;

float sampleVolume(vec3 coord) {
  // Which chunk of the volume this sample falls in, and where the page table
  // says that chunk currently lives in the atlas. Slot 0 means "not resident".
  ivec3 chunk = ivec3(floor(coord / float(BRICK_CORE)));
  int slot = -1;
  if (all(greaterThanEqual(chunk, ivec3(0))) && all(lessThan(chunk, PAGE_DIMS))) {
    slot = int(texelFetch(pageTable, chunk, 0).r + 0.5f) - 1;
  }

  if (slot >= 0) {
    ivec3 slotCoord = ivec3(
      slot % SLOTS_PER_AXIS,
      (slot / SLOTS_PER_AXIS) % SLOTS_PER_AXIS,
      slot / (SLOTS_PER_AXIS * SLOTS_PER_AXIS));
    // Texel `BRICK_PAD` of a brick is the first voxel of its core, so a sample
    // on a core face lands exactly between the same two voxels whichever of the
    // two neighbouring bricks it is taken from. That is what hides the seams.
    vec3 local = clamp(coord - vec3(chunk) * float(BRICK_CORE), 0.0f, float(BRICK_CORE));
    vec3 texel = vec3(slotCoord * BRICK_SIZE) + float(BRICK_PAD) + local;
    return texture(atlas, texel / float(ATLAS_DIM)).r;
  }
  return texture(volume, coord / volumeUniforms.loresExtent).r;
}

void main() {
  float gamma = 0.5;

  // The face normal, in the same level 0 voxel space the samples are taken in.
  // Derivatives of the interpolated coordinate give it per triangle, which is
  // what we want here: only `position` reaches the shader as an attribute.
  vec3 normal = normalize(cross(dFdx(voxelCoord), dFdy(voxelCoord)));

  float raw = 0.0f;
  for (int i = 0; i < NORMAL_SAMPLES; i++) {
    float offset = (float(i) - float(NORMAL_SAMPLES - 1) * 0.0f) * SAMPLE_DELTA;
    raw += sampleVolume(voxelCoord + normal * offset);
  }
  raw /= float(NORMAL_SAMPLES);

  if (raw < 0.001) {
    discard;
  }
  // Masking.
  // if (raw < 0.27) {
  //   raw = 0.0;
  // } else {
  //   //raw = 1.0;
  // }

  float intensity = pow(raw, 1/gamma);
  // Viewport overlay composites as `render.rgb + background * (1 - render.a)`.
  FragColor = vec4(vec3(intensity), 1.0);
}
