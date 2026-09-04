void main() {
  float gamma = 0.5;
  bool inHires = all(greaterThanEqual(hiresCoord, vec3(0.0)))
              && all(lessThanEqual(hiresCoord, vec3(1.0)));
  float raw = inHires ? texture(volumeHires, hiresCoord).r
                      : texture(volume, volumeCoord).r;
  float intensity = pow(raw, 1/gamma);
  float alpha = smoothstep(0.0, 0.05, intensity);
  //if (alpha <= 0.01) {
    //discard;
  //}
  // Viewport overlay composites as `render.rgb + background * (1 - render.a)`.
  FragColor = vec4(vec3(intensity), 1.0);
}
