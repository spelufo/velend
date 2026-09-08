// Samples averaged along the normal, and their spacing in level 0 voxels.
// Averaging across the face thins out the noise the surface is embedded in.
#define NORMAL_SAMPLES 5
const float SAMPLE_DELTA = 1.0f;

bool sampleL0(vec3 coord, out float value) {
  ivec3 chunk = ivec3(floor(coord / float(BRICK_CORE)));
  if (any(lessThan(chunk, ivec3(0))) || any(greaterThanEqual(chunk, L0_PAGE_DIMS))) {
    return false;
  }
  int slot = int(texelFetch(l0PageTable, chunk, 0).r + 0.5f) - 1;
  if (slot < 0) {
    return false;
  }
  ivec3 slotCoord = ivec3(
    slot % L0_SLOTS_PER_AXIS,
    (slot / L0_SLOTS_PER_AXIS) % L0_SLOTS_PER_AXIS,
    slot / (L0_SLOTS_PER_AXIS * L0_SLOTS_PER_AXIS));
  vec3 local = clamp(coord - vec3(chunk) * float(BRICK_CORE), 0.0f, float(BRICK_CORE));
  vec3 texel = vec3(slotCoord * BRICK_SIZE) + float(BRICK_PAD) + local;
  value = texture(l0Atlas, texel / float(L0_ATLAS_DIM)).r;
  return true;
}

#if LEVEL_CAP >= 1
bool sampleL1(vec3 coord, out float value) {
  coord *= 0.5f;
  ivec3 chunk = ivec3(floor(coord / float(BRICK_CORE)));
  if (any(lessThan(chunk, ivec3(0))) || any(greaterThanEqual(chunk, L1_PAGE_DIMS))) {
    return false;
  }
  int slot = int(texelFetch(l1PageTable, chunk, 0).r + 0.5f) - 1;
  if (slot < 0) {
    return false;
  }
  ivec3 slotCoord = ivec3(
    slot % L1_SLOTS_PER_AXIS,
    (slot / L1_SLOTS_PER_AXIS) % L1_SLOTS_PER_AXIS,
    slot / (L1_SLOTS_PER_AXIS * L1_SLOTS_PER_AXIS));
  vec3 local = clamp(coord - vec3(chunk) * float(BRICK_CORE), 0.0f, float(BRICK_CORE));
  vec3 texel = vec3(slotCoord * BRICK_SIZE) + float(BRICK_PAD) + local;
  value = texture(l1Atlas, texel / float(L1_ATLAS_DIM)).r;
  return true;
}
#endif

#if LEVEL_CAP >= 2
bool sampleL2(vec3 coord, out float value) {
  coord *= 0.25f;
  ivec3 chunk = ivec3(floor(coord / float(BRICK_CORE)));
  if (any(lessThan(chunk, ivec3(0))) || any(greaterThanEqual(chunk, L2_PAGE_DIMS))) {
    return false;
  }
  int slot = int(texelFetch(l2PageTable, chunk, 0).r + 0.5f) - 1;
  if (slot < 0) {
    return false;
  }
  ivec3 slotCoord = ivec3(
    slot % L2_SLOTS_PER_AXIS,
    (slot / L2_SLOTS_PER_AXIS) % L2_SLOTS_PER_AXIS,
    slot / (L2_SLOTS_PER_AXIS * L2_SLOTS_PER_AXIS));
  vec3 local = clamp(coord - vec3(chunk) * float(BRICK_CORE), 0.0f, float(BRICK_CORE));
  vec3 texel = vec3(slotCoord * BRICK_SIZE) + float(BRICK_PAD) + local;
  value = texture(l2Atlas, texel / float(L2_ATLAS_DIM)).r;
  return true;
}
#endif

#if LEVEL_CAP >= 3
bool sampleL3(vec3 coord, out float value) {
  coord *= 0.125f;
  ivec3 chunk = ivec3(floor(coord / float(BRICK_CORE)));
  if (any(lessThan(chunk, ivec3(0))) || any(greaterThanEqual(chunk, L3_PAGE_DIMS))) {
    return false;
  }
  int slot = int(texelFetch(l3PageTable, chunk, 0).r + 0.5f) - 1;
  if (slot < 0) {
    return false;
  }
  ivec3 slotCoord = ivec3(
    slot % L3_SLOTS_PER_AXIS,
    (slot / L3_SLOTS_PER_AXIS) % L3_SLOTS_PER_AXIS,
    slot / (L3_SLOTS_PER_AXIS * L3_SLOTS_PER_AXIS));
  vec3 local = clamp(coord - vec3(chunk) * float(BRICK_CORE), 0.0f, float(BRICK_CORE));
  vec3 texel = vec3(slotCoord * BRICK_SIZE) + float(BRICK_PAD) + local;
  value = texture(l3Atlas, texel / float(L3_ATLAS_DIM)).r;
  return true;
}
#endif

#if LEVEL_CAP >= 4
bool sampleL4(vec3 coord, out float value) {
  coord *= 0.0625f;
  ivec3 chunk = ivec3(floor(coord / float(BRICK_CORE)));
  if (any(lessThan(chunk, ivec3(0))) || any(greaterThanEqual(chunk, L4_PAGE_DIMS))) {
    return false;
  }
  int slot = int(texelFetch(l4PageTable, chunk, 0).r + 0.5f) - 1;
  if (slot < 0) {
    return false;
  }
  ivec3 slotCoord = ivec3(
    slot % L4_SLOTS_PER_AXIS,
    (slot / L4_SLOTS_PER_AXIS) % L4_SLOTS_PER_AXIS,
    slot / (L4_SLOTS_PER_AXIS * L4_SLOTS_PER_AXIS));
  vec3 local = clamp(coord - vec3(chunk) * float(BRICK_CORE), 0.0f, float(BRICK_CORE));
  vec3 texel = vec3(slotCoord * BRICK_SIZE) + float(BRICK_PAD) + local;
  value = texture(l4Atlas, texel / float(L4_ATLAS_DIM)).r;
  return true;
}
#endif

float sampleVolume(vec3 coord) {
  // Try streamed levels from finest to coarsest. Inactive levels and their
  // sampler accesses are removed by the preprocessor before compilation.
  float value;
  if (sampleL0(coord, value)) {
    return value;
  }
#if LEVEL_CAP >= 1
  if (sampleL1(coord, value)) {
    return value;
  }
#endif
#if LEVEL_CAP >= 2
  if (sampleL2(coord, value)) {
    return value;
  }
#endif
#if LEVEL_CAP >= 3
  if (sampleL3(coord, value)) {
    return value;
  }
#endif
#if LEVEL_CAP >= 4
  if (sampleL4(coord, value)) {
    return value;
  }
#endif
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
