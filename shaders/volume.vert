void main() {
  vec4 worldPosition = modelMatrix * vec4(position, 1.0f);
  volumeCoord = worldPosition.xyz / volumeScale;
  // volumeCoord = worldPosition.xyz;
  gl_Position = viewProjectionMatrix * worldPosition;
}
