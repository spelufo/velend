void main() {
  gl_Position = modelViewProjectionMatrix * vec4(position, 1.0f);
  texCoord = uv;
}
