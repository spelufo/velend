void main() {
  vec4 color = texture(surfaceImage, texCoord);
  color.a *= opacity;
  if (color.a <= 0.0f) {
    discard;
  }
  FragColor = color;
}
