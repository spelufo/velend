void main() {
  vec4 color = texture(volume, volumeCoord);
  float intensity = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
  float alpha = smoothstep(0.0, 0.05, intensity);
  if (alpha <= 0.01) {
    discard;
  }
  // Viewport overlay composites as `render.rgb + background * (1 - render.a)`.
  FragColor = vec4(vec3(intensity), alpha);
}
