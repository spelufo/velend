void main() {
  vec4 worldPosition = volumeUniforms.modelMatrix * vec4(position, 1.0f);
  volumeCoord = worldPosition.xyz / volumeUniforms.volumeScale;
  hiresCoord = (worldPosition.xyz - volumeUniforms.hiresOrigin) / volumeUniforms.hiresScale;
  // volumeCoord = worldPosition.xyz;
  gl_Position = volumeUniforms.viewProjectionMatrix * worldPosition;
}
