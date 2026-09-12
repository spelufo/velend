// DEBUG_LEVEL_COLORS tints each fragment by the level its samples came from
// instead of shading it. The renderer defines it from the scene's "Level
// Colors" setting, so the toggle in the panel is what turns it on.

// Samples averaged inward from the face. The renderer defines their count,
// spacing, and starting offset in level 0 voxels from physical scene settings.

// One level's lookup: level 0 coordinate in, the atlas texel it lands on out.
// `scale` takes the coordinate into this level's voxels, the page table says
// which atlas slot its chunk lives in, and `BRICK_PAD` skips the halo of
// neighbouring voxels the brick carries so that filtering stays seamless right
// up to the core boundary.
#define DEFINE_LEVEL_SAMPLER(fn, atlas, pageTable, pageDims, slotsPerAxis, atlasDim, scale) \
  bool fn(vec3 coord, out float value) {                                                   \
    coord *= scale;                                                                        \
    ivec3 chunk = ivec3(floor(coord / float(BRICK_CORE)));                                 \
    if (any(lessThan(chunk, ivec3(0))) || any(greaterThanEqual(chunk, pageDims))) {        \
      return false;                                                                        \
    }                                                                                      \
    int slot = int(texelFetch(pageTable, chunk, 0).r + 0.5f) - 1;                          \
    if (slot < 0) {                                                                        \
      return false;                                                                        \
    }                                                                                      \
    ivec3 slotCoord = ivec3(                                                               \
      slot % slotsPerAxis,                                                                 \
      (slot / slotsPerAxis) % slotsPerAxis,                                                \
      slot / (slotsPerAxis * slotsPerAxis));                                               \
    vec3 local = clamp(coord - vec3(chunk) * float(BRICK_CORE), 0.0f, float(BRICK_CORE));  \
    vec3 texel = vec3(slotCoord * BRICK_SIZE) + float(BRICK_PAD) + local;                  \
    value = texture(atlas, texel / float(atlasDim)).r;                                     \
    return true;                                                                           \
  }

#ifdef L0_ACTIVE
DEFINE_LEVEL_SAMPLER(sampleL0, l0Atlas, l0PageTable,
                     L0_PAGE_DIMS, L0_SLOTS_PER_AXIS, L0_ATLAS_DIM, L0_SCALE)
#endif
#ifdef L1_ACTIVE
DEFINE_LEVEL_SAMPLER(sampleL1, l1Atlas, l1PageTable,
                     L1_PAGE_DIMS, L1_SLOTS_PER_AXIS, L1_ATLAS_DIM, L1_SCALE)
#endif
#ifdef L2_ACTIVE
DEFINE_LEVEL_SAMPLER(sampleL2, l2Atlas, l2PageTable,
                     L2_PAGE_DIMS, L2_SLOTS_PER_AXIS, L2_ATLAS_DIM, L2_SCALE)
#endif
#ifdef L3_ACTIVE
DEFINE_LEVEL_SAMPLER(sampleL3, l3Atlas, l3PageTable,
                     L3_PAGE_DIMS, L3_SLOTS_PER_AXIS, L3_ATLAS_DIM, L3_SCALE)
#endif
#ifdef L4_ACTIVE
DEFINE_LEVEL_SAMPLER(sampleL4, l4Atlas, l4PageTable,
                     L4_PAGE_DIMS, L4_SLOTS_PER_AXIS, L4_ATLAS_DIM, L4_SCALE)
#endif
#ifdef L5_ACTIVE
DEFINE_LEVEL_SAMPLER(sampleL5, l5Atlas, l5PageTable,
                     L5_PAGE_DIMS, L5_SLOTS_PER_AXIS, L5_ATLAS_DIM, L5_SCALE)
#endif

// The level `sampleVolume` last read from, or -1 when nothing was resident.
// An out parameter would have to be threaded through every call site for the
// sake of the debug view alone.
int sampledLevel = -1;

float sampleVolume(vec3 coord) {
  // Try the levels being rendered from finest to coarsest. The ones left out of
  // LEVELS, and their sampler accesses, are removed by the preprocessor before
  // compilation. Nothing resident anywhere leaves the fragment black, which
  // `main` discards.
  float value;
#ifdef L0_ACTIVE
  if (sampleL0(coord, value)) {
    sampledLevel = 0;
    return value;
  }
#endif
#ifdef L1_ACTIVE
  if (sampleL1(coord, value)) {
    sampledLevel = 1;
    return value;
  }
#endif
#ifdef L2_ACTIVE
  if (sampleL2(coord, value)) {
    sampledLevel = 2;
    return value;
  }
#endif
#ifdef L3_ACTIVE
  if (sampleL3(coord, value)) {
    sampledLevel = 3;
    return value;
  }
#endif
#ifdef L4_ACTIVE
  if (sampleL4(coord, value)) {
    sampledLevel = 4;
    return value;
  }
#endif
#ifdef L5_ACTIVE
  if (sampleL5(coord, value)) {
    sampledLevel = 5;
    return value;
  }
#endif
  sampledLevel = -1;
  return 0.0f;
}

// Distinguishable hues, finest to coarsest.
vec3 levelColor(int level) {
  if (level == 0) return vec3(1.0f, 0.2f, 0.2f);
  if (level == 1) return vec3(1.0f, 0.6f, 0.1f);
  if (level == 2) return vec3(0.9f, 0.9f, 0.2f);
  if (level == 3) return vec3(0.3f, 1.0f, 0.3f);
  if (level == 4) return vec3(0.3f, 0.6f, 1.0f);
  if (level == 5) return vec3(0.8f, 0.3f, 1.0f);
  return vec3(0.5f);
}

void main() {
  float gamma = 0.5;

  // The face normal, in the same level 0 voxel space the samples are taken in.
  // Derivatives of the interpolated coordinate give it per triangle, which is
  // what we want here: only `position` reaches the shader as an attribute.
  vec3 normal = normalize(cross(dFdx(voxelCoord), dFdy(voxelCoord)));

  float raw = 0.0f;
#if DEBUG_LEVEL_COLORS
  int finestLevel = -1;
#endif
  for (int i = 0; i < NORMAL_SAMPLES; i++) {
    float depth = SAMPLE_OFFSET + float(i) * SAMPLE_DELTA;
    raw += sampleVolume(voxelCoord - normal * depth);
#if DEBUG_LEVEL_COLORS
    // The samples straddle a chunk boundary near the edges of a brick, so they
    // don't all come from the same level. Report the finest of them.
    if (sampledLevel >= 0 && (finestLevel < 0 || sampledLevel < finestLevel)) {
      finestLevel = sampledLevel;
    }
#endif
  }
  raw /= float(NORMAL_SAMPLES);

#if DEBUG_LEVEL_COLORS
  if (finestLevel < 0) {
    discard;
  }
  // Keep enough of the intensity to read the surface through the tint.
  FragColor = vec4(levelColor(finestLevel) * (0.01f + 0.65f * pow(raw, 2.0f)), 1.0f);
  return;
#endif

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
