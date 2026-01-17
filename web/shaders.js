function compileShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "shader compile failed");
  }
  return shader;
}

function createProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "program link failed");
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

const VS_MESH = `#version 300 es
precision highp float;

layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNor;
layout(location=2) in vec4 aCol;

uniform mat4 uViewProj;

out vec3 vNor;
out vec4 vCol;
out vec3 vPos;

void main() {
  vNor = aNor;
  vCol = aCol;
  vPos = aPos;
  gl_Position = uViewProj * vec4(aPos, 1.0);
}
`;

const FS_MESH = `#version 300 es
precision highp float;

in vec3 vNor;
in vec4 vCol;
in vec3 vPos;

uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform vec3 uFogColor;
uniform float uFogDensity;

out vec4 outColor;

void main() {
  vec3 n = normalize(vNor);
  float ndl = max(dot(n, normalize(uLightDir)), 0.0);
  vec3 base = vCol.rgb;
  vec3 lit = base * (0.25 + 0.75 * ndl);

  float d = length(vPos - uCamPos);
  float fog = 1.0 - exp(-uFogDensity * d * d);
  vec3 rgb = mix(lit, uFogColor, fog);

  // Premultiply alpha.
  outColor = vec4(rgb * vCol.a, vCol.a);
}
`;

export function createMeshProgram(gl) {
  const program = createProgram(gl, VS_MESH, FS_MESH);
  return {
    program,
    uniforms: {
      uViewProj: gl.getUniformLocation(program, "uViewProj"),
      uLightDir: gl.getUniformLocation(program, "uLightDir"),
      uCamPos: gl.getUniformLocation(program, "uCamPos"),
      uFogColor: gl.getUniformLocation(program, "uFogColor"),
      uFogDensity: gl.getUniformLocation(program, "uFogDensity"),
    },
  };
}

