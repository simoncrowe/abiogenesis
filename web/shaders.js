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

uniform mat4 uView;
uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uLightIntensity;
uniform float uSssEnabled;
uniform float uSssWrap;
uniform float uSssBackStrength;
uniform float uSssBackPower;

layout(location=0) out vec4 outColor;
layout(location=1) out vec4 outNormal;

void main() {
  vec3 n = normalize(vNor);
  vec3 l = normalize(uLightDir);

  float ndlRaw = dot(n, l);
  float ndl = max(ndlRaw, 0.0);

  float sssEnabled = clamp(uSssEnabled, 0.0, 1.0);

  // "Wrap" diffuse lighting (cheap subsurface feel).
  float wrap = clamp(uSssWrap, 0.0, 1.0) * sssEnabled;
  float ndlWrap = clamp((ndlRaw + wrap) / (1.0 + wrap), 0.0, 1.0);

  vec3 base = vCol.rgb;
  vec3 lit = base * (0.25 + 0.75 * ndlWrap) * uLightIntensity;

  // Backscatter (view-dependent), for translucent blob feel.
  vec3 v = normalize(uCamPos - vPos);
  float back = pow(clamp(dot(-l, v), 0.0, 1.0), max(uSssBackPower, 0.1));
  back *= clamp(uSssBackStrength, 0.0, 2.0) * sssEnabled;
  back *= (1.0 - ndl);

  vec3 backTint = mix(base, vec3(1.0, 0.65, 0.45), 0.25);
  lit += backTint * back * uLightIntensity;

  float d = length(vPos - uCamPos);
  float fog = 1.0 - exp(-uFogDensity * d * d);
  vec3 rgb = mix(lit, uFogColor, fog);

  // Premultiply alpha.
  outColor = vec4(rgb * vCol.a, vCol.a);

  // View-space normal encoded in 0..1.
  vec3 nView = normalize(mat3(uView) * n);
  outNormal = vec4(nView * 0.5 + 0.5, 1.0);
}
`;

export function createMeshProgram(gl) {
  const program = createProgram(gl, VS_MESH, FS_MESH);
  return {
    program,
    uniforms: {
      uViewProj: gl.getUniformLocation(program, "uViewProj"),
      uView: gl.getUniformLocation(program, "uView"),
      uLightDir: gl.getUniformLocation(program, "uLightDir"),
      uCamPos: gl.getUniformLocation(program, "uCamPos"),
      uFogColor: gl.getUniformLocation(program, "uFogColor"),
      uFogDensity: gl.getUniformLocation(program, "uFogDensity"),
      uLightIntensity: gl.getUniformLocation(program, "uLightIntensity"),
      uSssEnabled: gl.getUniformLocation(program, "uSssEnabled"),
      uSssWrap: gl.getUniformLocation(program, "uSssWrap"),
      uSssBackStrength: gl.getUniformLocation(program, "uSssBackStrength"),
      uSssBackPower: gl.getUniformLocation(program, "uSssBackPower"),
    },
  };
}

const VS_POST = `#version 300 es
precision highp float;

out vec2 vUv;

void main() {
  vec2 p = vec2(
    (gl_VertexID == 2) ? 3.0 : -1.0,
    (gl_VertexID == 1) ? 3.0 : -1.0
  );
  vUv = 0.5 * (p + 1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}
`;

const FS_SSAO = `#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uDepth;
uniform sampler2D uNormal;
uniform vec2 uInvResolution;
uniform mat4 uProj;
uniform mat4 uInvProj;
uniform float uTanHalfFovy;
uniform float uFogDensity;
uniform float uNear;
uniform float uFar;
uniform float uAoEnabled;
uniform float uAoIntensity;
uniform float uAoRadiusPx;
uniform int uAoSampleCount;
uniform float uAoBias;

out vec4 outColor;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float linearizeDepth(float depth) {
  float zNdc = depth * 2.0 - 1.0;
  return (2.0 * uNear * uFar) / (uFar + uNear - zNdc * (uFar - uNear));
}

vec3 viewPosFromDepth(float depth, vec2 uv) {
  float zNdc = depth * 2.0 - 1.0;
  vec4 ndc = vec4(uv * 2.0 - 1.0, zNdc, 1.0);
  vec4 v = uInvProj * ndc;
  return v.xyz / v.w;
}

void main() {
  float enabled = step(0.5, uAoEnabled);
  if (enabled < 0.5) {
    outColor = vec4(1.0);
    return;
  }

  float d = texture(uDepth, vUv).r;
  if (d >= 0.999999) {
    outColor = vec4(1.0);
    return;
  }

  vec3 n = texture(uNormal, vUv).rgb * 2.0 - 1.0;
  n = normalize(n);

  vec3 p = viewPosFromDepth(d, vUv);
  float zDist = linearizeDepth(d);

  // Convert screen-radius in px to view-space radius at this depth.
  float viewportH = 1.0 / max(uInvResolution.y, 1e-6);
  float radius = (uAoRadiusPx * (2.0 * zDist * uTanHalfFovy)) / viewportH;

  // Depth-based fade to avoid AO in the fog.
  float fog = 1.0 - exp(-uFogDensity * zDist * zDist);
  float aoFade = 1.0 - fog;

  float a = hash12(gl_FragCoord.xy) * 6.28318530718;
  vec3 rand = normalize(vec3(cos(a), sin(a), hash12(gl_FragCoord.yx) * 2.0 - 1.0));

  vec3 t = normalize(rand - n * dot(rand, n));
  vec3 b = cross(n, t);
  mat3 tbn = mat3(t, b, n);

  const int MAX_SAMPLES = 16;
  int sc = clamp(uAoSampleCount, 1, MAX_SAMPLES);

  float occ = 0.0;
  for (int i = 0; i < MAX_SAMPLES; i++) {
    if (i >= sc) continue;

    float fi = float(i) + 0.5;
    float t01 = fi / float(MAX_SAMPLES);
    float phi = fi * 2.399963229728653; // golden angle

    // Hemisphere sample (z in 0..1).
    float z01 = mix(0.15, 1.0, t01);
    float r = sqrt(max(1.0 - z01 * z01, 0.0));
    vec3 k = vec3(cos(phi) * r, sin(phi) * r, z01);

    // Concentrate samples closer to the surface.
    float scale = mix(0.1, 1.0, t01 * t01);

    vec3 sampDir = normalize(tbn * k);
    vec3 sampPos = p + sampDir * (radius * scale);

    vec4 clip = uProj * vec4(sampPos, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;

    float sd = texture(uDepth, uv).r;
    float sz = linearizeDepth(sd);
    float sampZ = -sampPos.z;

    float rangeCheck = smoothstep(0.0, 1.0, radius / max(abs(sz - zDist), 1e-3));
    float occluded = step(sz, sampZ - uAoBias);
    occ += occluded * rangeCheck;
  }
  occ *= 1.0 / float(sc);

  float ao = 1.0 - occ * uAoIntensity * aoFade;
  ao = clamp(ao, 0.0, 1.0);

  outColor = vec4(ao);
}
`;

const FS_AO_BLUR = `#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uAo;
uniform sampler2D uDepth;
uniform vec2 uInvResolution;
uniform vec2 uDirection;
uniform int uBlurRadius;
uniform float uNear;
uniform float uFar;

out vec4 outColor;

float linearizeDepth(float depth) {
  float zNdc = depth * 2.0 - 1.0;
  return (2.0 * uNear * uFar) / (uFar + uNear - zNdc * (uFar - uNear));
}

void main() {
  float centerD = texture(uDepth, vUv).r;
  float centerZ = linearizeDepth(centerD);

  float centerAo = texture(uAo, vUv).r;

  int r = clamp(uBlurRadius, 0, 4);
  if (r == 0) {
    outColor = vec4(centerAo);
    return;
  }

  float sigma = max(0.8, float(r) * 0.75);

  float sum = centerAo;
  float wsum = 1.0;

  // Separable depth-aware blur (one axis per pass).
  for (int i = -4; i <= 4; i++) {
    if (i == 0) continue;
    if (abs(i) > r) continue;

    vec2 uv = vUv + uDirection * (float(i) * uInvResolution);

    float ao = texture(uAo, uv).r;
    float dz = abs(linearizeDepth(texture(uDepth, uv).r) - centerZ);

    float dist2 = float(i * i);
    float spatialW = exp(-dist2 / (2.0 * sigma * sigma));
    float depthW = exp(-(dz * dz) / max(0.0005 + 0.02 * centerZ, 1e-5));
    float w = spatialW * depthW;

    sum += ao * w;
    wsum += w;
  }

  outColor = vec4(sum / wsum);
}
`;

const FS_AO_COMPOSITE = `#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uColor;
uniform sampler2D uAo;
uniform vec3 uFogColor;

out vec4 outColor;

void main() {
  vec4 col = texture(uColor, vUv);
  float ao = texture(uAo, vUv).r;

  vec3 rgb = mix(uFogColor, col.rgb, ao);
  outColor = vec4(rgb, col.a);
}
`;

function createPostVao(gl) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.bindVertexArray(null);
  return vao;
}

export function createAoProgram(gl) {
  const program = createProgram(gl, VS_POST, FS_SSAO);
  return {
    program,
    vao: createPostVao(gl),
    uniforms: {
      uDepth: gl.getUniformLocation(program, "uDepth"),
      uNormal: gl.getUniformLocation(program, "uNormal"),
      uInvResolution: gl.getUniformLocation(program, "uInvResolution"),
      uProj: gl.getUniformLocation(program, "uProj"),
      uInvProj: gl.getUniformLocation(program, "uInvProj"),
      uTanHalfFovy: gl.getUniformLocation(program, "uTanHalfFovy"),
      uFogDensity: gl.getUniformLocation(program, "uFogDensity"),
      uNear: gl.getUniformLocation(program, "uNear"),
      uFar: gl.getUniformLocation(program, "uFar"),
      uAoEnabled: gl.getUniformLocation(program, "uAoEnabled"),
      uAoIntensity: gl.getUniformLocation(program, "uAoIntensity"),
      uAoRadiusPx: gl.getUniformLocation(program, "uAoRadiusPx"),
      uAoSampleCount: gl.getUniformLocation(program, "uAoSampleCount"),
      uAoBias: gl.getUniformLocation(program, "uAoBias"),
    },
  };
}

export function createAoBlurProgram(gl) {
  const program = createProgram(gl, VS_POST, FS_AO_BLUR);
  return {
    program,
    vao: createPostVao(gl),
    uniforms: {
      uAo: gl.getUniformLocation(program, "uAo"),
      uDepth: gl.getUniformLocation(program, "uDepth"),
      uInvResolution: gl.getUniformLocation(program, "uInvResolution"),
      uDirection: gl.getUniformLocation(program, "uDirection"),
      uBlurRadius: gl.getUniformLocation(program, "uBlurRadius"),
      uNear: gl.getUniformLocation(program, "uNear"),
      uFar: gl.getUniformLocation(program, "uFar"),
    },
  };
}

export function createAoCompositeProgram(gl) {
  const program = createProgram(gl, VS_POST, FS_AO_COMPOSITE);
  return {
    program,
    vao: createPostVao(gl),
    uniforms: {
      uColor: gl.getUniformLocation(program, "uColor"),
      uAo: gl.getUniformLocation(program, "uAo"),
      uFogColor: gl.getUniformLocation(program, "uFogColor"),
    },
  };
}
