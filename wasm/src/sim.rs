use wasm_bindgen::prelude::*;

use crate::math::{clamp01, lerp};
use crate::meshing::{
    generate_isosurface_mesh, mesh_region_append, mesh_region_append_lerp, MeshBuffers,
};

#[wasm_bindgen]
#[derive(Clone)]
pub struct GrayScottParams {
    du: f32,
    dv: f32,
    feed: f32,
    kill: f32,
}

#[wasm_bindgen]
impl GrayScottParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            du: 0.16,
            dv: 0.08,
            feed: 0.035,
            kill: 0.065,
        }
    }

    pub fn du(&self) -> f32 {
        self.du
    }
    pub fn dv(&self) -> f32 {
        self.dv
    }
    pub fn feed(&self) -> f32 {
        self.feed
    }
    pub fn kill(&self) -> f32 {
        self.kill
    }

    pub fn set_du(&mut self, v: f32) {
        self.du = v;
    }
    pub fn set_dv(&mut self, v: f32) {
        self.dv = v;
    }
    pub fn set_feed(&mut self, v: f32) {
        self.feed = v;
    }
    pub fn set_kill(&mut self, v: f32) {
        self.kill = v;
    }
}

#[derive(Clone, Copy)]
struct LcgRng {
    state: u32,
}

impl LcgRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes LCG.
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        let x = self.next_u32();
        // Take 24 bits.
        let mant = (x >> 8) as u32;
        (mant as f32) / ((1u32 << 24) as f32)
    }
}

fn idx(nx: usize, ny: usize, x: usize, y: usize, z: usize) -> usize {
    x + nx * (y + ny * z)
}

fn wrap(v: isize, n: usize) -> usize {
    let n = n as isize;
    let mut x = v % n;
    if x < 0 {
        x += n;
    }
    x as usize
}

fn fade(t: f32) -> f32 {
    // Quintic smoothstep.
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn hash3(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    // Cheap 3D integer hash (deterministic across platforms).
    let mut h = seed;
    h ^= (x as u32).wrapping_mul(0x9E3779B1);
    h ^= (y as u32).wrapping_mul(0x85EBCA77);
    h ^= (z as u32).wrapping_mul(0xC2B2AE3D);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846CA68B);
    h ^ (h >> 16)
}

fn grad_from_hash(h: u32) -> [f32; 3] {
    // 12 classic Perlin gradient directions.
    const G: [[f32; 3]; 12] = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ];
    G[(h % (G.len() as u32)) as usize]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn perlin3(seed: u32, x: f32, y: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;

    let xf = x - x0 as f32;
    let yf = y - y0 as f32;
    let zf = z - z0 as f32;

    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let z1 = z0 + 1;

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let g000 = grad_from_hash(hash3(seed, x0, y0, z0));
    let g100 = grad_from_hash(hash3(seed, x1, y0, z0));
    let g010 = grad_from_hash(hash3(seed, x0, y1, z0));
    let g110 = grad_from_hash(hash3(seed, x1, y1, z0));
    let g001 = grad_from_hash(hash3(seed, x0, y0, z1));
    let g101 = grad_from_hash(hash3(seed, x1, y0, z1));
    let g011 = grad_from_hash(hash3(seed, x0, y1, z1));
    let g111 = grad_from_hash(hash3(seed, x1, y1, z1));

    let n000 = dot3(g000, [xf, yf, zf]);
    let n100 = dot3(g100, [xf - 1.0, yf, zf]);
    let n010 = dot3(g010, [xf, yf - 1.0, zf]);
    let n110 = dot3(g110, [xf - 1.0, yf - 1.0, zf]);
    let n001 = dot3(g001, [xf, yf, zf - 1.0]);
    let n101 = dot3(g101, [xf - 1.0, yf, zf - 1.0]);
    let n011 = dot3(g011, [xf, yf - 1.0, zf - 1.0]);
    let n111 = dot3(g111, [xf - 1.0, yf - 1.0, zf - 1.0]);

    let x00 = lerp(n000, n100, u);
    let x10 = lerp(n010, n110, u);
    let x01 = lerp(n001, n101, u);
    let x11 = lerp(n011, n111, u);

    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);

    lerp(y0, y1, w)
}

const CHUNK: usize = 16;

#[wasm_bindgen]
pub struct Simulation {
    nx: usize,
    ny: usize,
    nz: usize,
    nxy: usize,

    // Precomputed wrapped neighbor indices, to avoid per-voxel `wrap()` in hot loops.
    x_minus: Vec<usize>,
    x_plus: Vec<usize>,
    y_minus: Vec<usize>,
    y_plus: Vec<usize>,
    z_minus: Vec<usize>,
    z_plus: Vec<usize>,

    params: GrayScottParams,
    base_seed: u32,
    dt: f32,

    chunk_nx: usize,
    chunk_ny: usize,
    chunk_nz: usize,
    chunk_v_min: Vec<f32>,
    chunk_v_max: Vec<f32>,

    // Chunk min/max are used for iso culling in meshing.
    // For performance, we recompute them on-demand (e.g. when publishing a snapshot),
    // not every simulation step.
    u: Vec<f32>,
    v: Vec<f32>,
    u2: Vec<f32>,
    v2: Vec<f32>,

    rng: LcgRng,

    mesh: MeshBuffers,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: usize, ny: usize, nz: usize, seed: u32, params: GrayScottParams) -> Self {
        let n = nx * ny * nz;
        let nxy = nx * ny;

        let x_minus: Vec<usize> = (0..nx).map(|x| wrap(x as isize - 1, nx)).collect();
        let x_plus: Vec<usize> = (0..nx).map(|x| wrap(x as isize + 1, nx)).collect();
        let y_minus: Vec<usize> = (0..ny).map(|y| wrap(y as isize - 1, ny)).collect();
        let y_plus: Vec<usize> = (0..ny).map(|y| wrap(y as isize + 1, ny)).collect();
        let z_minus: Vec<usize> = (0..nz).map(|z| wrap(z as isize - 1, nz)).collect();
        let z_plus: Vec<usize> = (0..nz).map(|z| wrap(z as isize + 1, nz)).collect();

        let cubes_x = nx.saturating_sub(1);
        let cubes_y = ny.saturating_sub(1);
        let cubes_z = nz.saturating_sub(1);

        let chunk_nx = (cubes_x + CHUNK - 1) / CHUNK;
        let chunk_ny = (cubes_y + CHUNK - 1) / CHUNK;
        let chunk_nz = (cubes_z + CHUNK - 1) / CHUNK;
        let chunk_total = chunk_nx * chunk_ny * chunk_nz;

        Self {
            nx,
            ny,
            nz,
            nxy,

            x_minus,
            x_plus,
            y_minus,
            y_plus,
            z_minus,
            z_plus,

            params,
            base_seed: seed,
            dt: 0.25,

            chunk_nx,
            chunk_ny,
            chunk_nz,
            chunk_v_min: vec![f32::INFINITY; chunk_total],
            chunk_v_max: vec![f32::NEG_INFINITY; chunk_total],

            u: vec![1.0; n],
            v: vec![0.0; n],
            u2: vec![1.0; n],
            v2: vec![0.0; n],
            rng: LcgRng::new(seed),
            mesh: MeshBuffers::new(),
        }
    }

    pub fn dims_x(&self) -> usize {
        self.nx
    }
    pub fn dims_y(&self) -> usize {
        self.ny
    }
    pub fn dims_z(&self) -> usize {
        self.nz
    }

    pub fn dt(&self) -> f32 {
        self.dt
    }

    pub fn set_dt(&mut self, dt: f32) {
        // Keep it sane for explicit Euler.
        self.dt = dt.max(0.001).min(1.0);
    }

    fn chunk_index(&self, cx: usize, cy: usize, cz: usize) -> usize {
        cx + self.chunk_nx * (cy + self.chunk_ny * cz)
    }

    fn reset_chunk_ranges(&mut self) {
        self.chunk_v_min.fill(f32::INFINITY);
        self.chunk_v_max.fill(f32::NEG_INFINITY);
    }

    fn update_chunk_ranges_for_point(&mut self, x: usize, y: usize, z: usize, v: f32) {
        let cubes_x = self.nx.saturating_sub(1);
        let cubes_y = self.ny.saturating_sub(1);
        let cubes_z = self.nz.saturating_sub(1);

        if cubes_x == 0 || cubes_y == 0 || cubes_z == 0 {
            return;
        }

        let x_left = if x > 0 { x - 1 } else { 0 };
        let x_right = if x < cubes_x { x } else { cubes_x - 1 };
        let y_left = if y > 0 { y - 1 } else { 0 };
        let y_right = if y < cubes_y { y } else { cubes_y - 1 };
        let z_left = if z > 0 { z - 1 } else { 0 };
        let z_right = if z < cubes_z { z } else { cubes_z - 1 };

        let cx0 = x_left / CHUNK;
        let cx1 = x_right / CHUNK;
        let cy0 = y_left / CHUNK;
        let cy1 = y_right / CHUNK;
        let cz0 = z_left / CHUNK;
        let cz1 = z_right / CHUNK;

        let cxs = if cx0 == cx1 { [cx0, cx0] } else { [cx0, cx1] };
        let cys = if cy0 == cy1 { [cy0, cy0] } else { [cy0, cy1] };
        let czs = if cz0 == cz1 { [cz0, cz0] } else { [cz0, cz1] };

        for &cz in &czs {
            for &cy in &cys {
                for &cx in &cxs {
                    let i = self.chunk_index(cx, cy, cz);
                    let minv = self.chunk_v_min[i];
                    let maxv = self.chunk_v_max[i];
                    self.chunk_v_min[i] = if v < minv { v } else { minv };
                    self.chunk_v_max[i] = if v > maxv { v } else { maxv };
                }
            }
        }
    }

    pub fn seed_sphere(&mut self, radius01: f32) {
        let rx = (self.nx as f32 * radius01).max(1.0);
        let ry = (self.ny as f32 * radius01).max(1.0);
        let rz = (self.nz as f32 * radius01).max(1.0);

        let cx = (self.nx as f32 - 1.0) * 0.5;
        let cy = (self.ny as f32 - 1.0) * 0.5;
        let cz = (self.nz as f32 - 1.0) * 0.5;

        self.reset_chunk_ranges();

        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let dx = (x as f32 - cx) / rx;
                    let dy = (y as f32 - cy) / ry;
                    let dz = (z as f32 - cz) / rz;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    let i = idx(self.nx, self.ny, x, y, z);

                    if d2 <= 1.0 {
                        self.v[i] = 1.0;
                        self.u[i] = 0.0;
                    } else {
                        // Add a tiny amount of deterministic noise outside.
                        let n = (self.rng.next_f32() - 0.5) * 0.02;
                        self.v[i] = clamp01(self.v[i] + n);
                    }

                    self.update_chunk_ranges_for_point(x, y, z, self.v[i]);
                }
            }
        }
    }

    /// Seeds the lattice with deterministic 3D Perlin-style fractal noise.
    ///
    /// - `frequency`: roughly the number of noise cells across the volume.
    /// - `octaves`: number of fBm octaves (1..=8 recommended).
    /// - `v_bias`: constant offset added to V.
    /// - `v_amp`: multiplier applied to noise in [0,1].
    ///
    /// Sets `U = 1 - V` for a common Gray–Scott starting condition.
    pub fn seed_perlin(&mut self, frequency: f32, octaves: u32, v_bias: f32, v_amp: f32) {
        let n = self.nx * self.ny * self.nz;
        if n == 0 {
            return;
        }

        let freq = frequency.max(0.0001);
        let oct = octaves.max(1).min(12);

        let fx = freq / (self.nx.max(1) as f32);
        let fy = freq / (self.ny.max(1) as f32);
        let fz = freq / (self.nz.max(1) as f32);

        self.reset_chunk_ranges();

        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let mut f = 1.0;
                    let mut a = 0.5;
                    let mut sum = 0.0;
                    let mut norm = 0.0;

                    for _ in 0..oct {
                        let nx = x as f32 * fx * f;
                        let ny = y as f32 * fy * f;
                        let nz = z as f32 * fz * f;
                        sum += a * perlin3(self.base_seed, nx, ny, nz);
                        norm += a;
                        f *= 2.0;
                        a *= 0.5;
                    }

                    // Perlin is approximately in [-1, 1]. Map to [0, 1].
                    let noise01 = (sum / norm) * 0.5 + 0.5;

                    let i = idx(self.nx, self.ny, x, y, z);
                    let vv = clamp01(v_bias + v_amp * noise01);
                    self.v[i] = vv;
                    self.u[i] = 1.0 - vv;
                    self.update_chunk_ranges_for_point(x, y, z, vv);
                }
            }
        }
    }

    pub fn step(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step_once();
        }
    }

    pub fn recompute_chunk_ranges_from_v(&mut self) {
        let cubes_x = self.nx.saturating_sub(1);
        let cubes_y = self.ny.saturating_sub(1);
        let cubes_z = self.nz.saturating_sub(1);

        if cubes_x == 0 || cubes_y == 0 || cubes_z == 0 {
            return;
        }

        self.chunk_v_min.fill(f32::INFINITY);
        self.chunk_v_max.fill(f32::NEG_INFINITY);

        for cz in 0..self.chunk_nz {
            let z0 = cz * CHUNK;
            let z1 = ((cz + 1) * CHUNK).min(cubes_z);
            for cy in 0..self.chunk_ny {
                let y0 = cy * CHUNK;
                let y1 = ((cy + 1) * CHUNK).min(cubes_y);
                for cx in 0..self.chunk_nx {
                    let x0 = cx * CHUNK;
                    let x1 = ((cx + 1) * CHUNK).min(cubes_x);

                    let mut minv = f32::INFINITY;
                    let mut maxv = f32::NEG_INFINITY;

                    // Chunk ranges are defined over cubes, but cube corner values live on points.
                    // For cube range [x0, x1), we need point range [x0, x1] (inclusive).
                    for z in z0..=z1 {
                        for y in y0..=y1 {
                            for x in x0..=x1 {
                                let v = self.v[idx(self.nx, self.ny, x, y, z)];
                                if v < minv {
                                    minv = v;
                                }
                                if v > maxv {
                                    maxv = v;
                                }
                            }
                        }
                    }

                    let ci = self.chunk_index(cx, cy, cz);
                    self.chunk_v_min[ci] = minv;
                    self.chunk_v_max[ci] = maxv;
                }
            }
        }
    }

    fn step_once(&mut self) {
        let du = self.params.du;
        let dv = self.params.dv;
        let f = self.params.feed;
        let k = self.params.kill;

        let nx = self.nx;
        let nxy = self.nxy;

        let x_minus = &self.x_minus;
        let x_plus = &self.x_plus;
        let y_minus = &self.y_minus;
        let y_plus = &self.y_plus;
        let z_minus = &self.z_minus;
        let z_plus = &self.z_plus;

        let u = &self.u;
        let v = &self.v;
        let u2 = &mut self.u2;
        let v2 = &mut self.v2;

        for z in 0..self.nz {
            let z_off = z * nxy;
            let z_off_m = z_minus[z] * nxy;
            let z_off_p = z_plus[z] * nxy;

            for y in 0..self.ny {
                let y_off = z_off + y * nx;
                let y_off_m = z_off + y_minus[y] * nx;
                let y_off_p = z_off + y_plus[y] * nx;
                let y_off_zm = z_off_m + y * nx;
                let y_off_zp = z_off_p + y * nx;

                for x in 0..nx {
                    let c = y_off + x;
                    let uu = u[c];
                    let vv = v[c];

                    let u_lap = u[y_off + x_minus[x]]
                        + u[y_off + x_plus[x]]
                        + u[y_off_m + x]
                        + u[y_off_p + x]
                        + u[y_off_zm + x]
                        + u[y_off_zp + x]
                        - 6.0 * uu;

                    let v_lap = v[y_off + x_minus[x]]
                        + v[y_off + x_plus[x]]
                        + v[y_off_m + x]
                        + v[y_off_p + x]
                        + v[y_off_zm + x]
                        + v[y_off_zp + x]
                        - 6.0 * vv;

                    let uvv = uu * vv * vv;
                    let du_dt = du * u_lap - uvv + f * (1.0 - uu);
                    let dv_dt = dv * v_lap + uvv - (f + k) * vv;

                    u2[c] = clamp01(uu + self.dt * du_dt);
                    v2[c] = clamp01(vv + self.dt * dv_dt);
                }
            }
        }

        std::mem::swap(&mut self.u, &mut self.u2);
        std::mem::swap(&mut self.v, &mut self.v2);
    }

    pub fn v_min(&self) -> f32 {
        self.v
            .iter()
            .copied()
            .fold(f32::INFINITY, |a, b| if b < a { b } else { a })
    }

    pub fn v_max(&self) -> f32 {
        self.v
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a })
    }

    pub fn v_count_ge(&self, threshold: f32) -> usize {
        self.v.iter().filter(|&&x| x >= threshold).count()
    }

    // Raw access for fast snapshotting / JS-side seeding (e.g. into SharedArrayBuffer).
    pub fn u_ptr(&self) -> u32 {
        self.u.as_ptr() as u32
    }

    pub fn u_len(&self) -> usize {
        self.u.len()
    }

    pub fn v_ptr(&self) -> u32 {
        self.v.as_ptr() as u32
    }

    pub fn v_len(&self) -> usize {
        self.v.len()
    }

    // Chunk min/max are used for iso culling in meshing.
    pub fn chunk_v_min_ptr(&self) -> u32 {
        self.chunk_v_min.as_ptr() as u32
    }

    pub fn chunk_v_max_ptr(&self) -> u32 {
        self.chunk_v_max.as_ptr() as u32
    }

    pub fn chunk_v_len(&self) -> usize {
        self.chunk_v_min.len()
    }

    pub fn generate_isosurface_mesh_visible(
        &mut self,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius: f32,
        iso: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) {
        self.mesh.clear();

        let cubes_x = self.nx.saturating_sub(1);
        let cubes_y = self.ny.saturating_sub(1);
        let cubes_z = self.nz.saturating_sub(1);

        if cubes_x == 0 || cubes_y == 0 || cubes_z == 0 {
            return;
        }

        let fx = cubes_x as f32;
        let fy = cubes_y as f32;
        let fz = cubes_z as f32;

        let rad = radius.max(0.0);
        let rad2 = rad * rad;

        // Camera in cube coordinates.
        let cam_ix = (cam_x + 0.5) * fx;
        let cam_iy = (cam_y + 0.5) * fy;
        let cam_iz = (cam_z + 0.5) * fz;

        let rx = rad * fx;
        let ry = rad * fy;
        let rz = rad * fz;

        let min_x = (cam_ix - rx).floor().max(0.0) as usize;
        let min_y = (cam_iy - ry).floor().max(0.0) as usize;
        let min_z = (cam_iz - rz).floor().max(0.0) as usize;

        let max_x = (cam_ix + rx).ceil().min((cubes_x - 1) as f32) as usize;
        let max_y = (cam_iy + ry).ceil().min((cubes_y - 1) as f32) as usize;
        let max_z = (cam_iz + rz).ceil().min((cubes_z - 1) as f32) as usize;

        let min_cx = min_x / CHUNK;
        let min_cy = min_y / CHUNK;
        let min_cz = min_z / CHUNK;
        let max_cx = max_x / CHUNK;
        let max_cy = max_y / CHUNK;
        let max_cz = max_z / CHUNK;

        let color = [r, g, b, a];

        let near_r = rad * 0.33;
        let mid_r = rad * 0.66;

        fn axis_dist2(p: f32, a0: f32, a1: f32) -> f32 {
            if p < a0 {
                let d = a0 - p;
                d * d
            } else if p > a1 {
                let d = p - a1;
                d * d
            } else {
                0.0
            }
        }

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    let ci = self.chunk_index(cx, cy, cz);
                    let cmin = self.chunk_v_min[ci];
                    let cmax = self.chunk_v_max[ci];
                    if !cmin.is_finite() {
                        continue;
                    }
                    if iso < cmin || iso > cmax {
                        continue;
                    }

                    let x0 = cx * CHUNK;
                    let y0 = cy * CHUNK;
                    let z0 = cz * CHUNK;

                    let x1 = ((cx + 1) * CHUNK).min(cubes_x);
                    let y1 = ((cy + 1) * CHUNK).min(cubes_y);
                    let z1 = ((cz + 1) * CHUNK).min(cubes_z);

                    let wx0 = x0 as f32 / fx - 0.5;
                    let wy0 = y0 as f32 / fy - 0.5;
                    let wz0 = z0 as f32 / fz - 0.5;
                    let wx1 = x1 as f32 / fx - 0.5;
                    let wy1 = y1 as f32 / fy - 0.5;
                    let wz1 = z1 as f32 / fz - 0.5;

                    // Sphere-AABB cull.
                    let d2 = axis_dist2(cam_x, wx0, wx1)
                        + axis_dist2(cam_y, wy0, wy1)
                        + axis_dist2(cam_z, wz0, wz1);
                    if d2 > rad2 {
                        continue;
                    }

                    // LOD stride by chunk-center distance.
                    let cxw = (wx0 + wx1) * 0.5;
                    let cyw = (wy0 + wy1) * 0.5;
                    let czw = (wz0 + wz1) * 0.5;
                    let dist = ((cxw - cam_x) * (cxw - cam_x)
                        + (cyw - cam_y) * (cyw - cam_y)
                        + (czw - cam_z) * (czw - cam_z))
                        .sqrt();

                    let stride = if dist < near_r {
                        1
                    } else if dist < mid_r {
                        2
                    } else {
                        4
                    };

                    mesh_region_append(
                        &self.v,
                        self.nx,
                        self.ny,
                        self.nz,
                        iso,
                        color,
                        &mut self.mesh,
                        x0 as isize,
                        x1 as isize,
                        y0 as isize,
                        y1 as isize,
                        z0 as isize,
                        z1 as isize,
                        stride,
                    );
                }
            }
        }
    }

    pub fn generate_isosurface_mesh(&mut self, iso: f32, r: f32, g: f32, b: f32, a: f32) {
        generate_isosurface_mesh(
            &self.v,
            self.nx,
            self.ny,
            self.nz,
            iso,
            [r, g, b, a],
            &mut self.mesh,
        );
    }

    pub fn mesh_vertex_count(&self) -> usize {
        self.mesh.positions.len() / 3
    }

    pub fn mesh_positions_ptr(&self) -> u32 {
        self.mesh.positions.as_ptr() as u32
    }
    pub fn mesh_positions_len(&self) -> usize {
        self.mesh.positions.len()
    }

    pub fn mesh_normals_ptr(&self) -> u32 {
        self.mesh.normals.as_ptr() as u32
    }
    pub fn mesh_normals_len(&self) -> usize {
        self.mesh.normals.len()
    }

    pub fn mesh_colors_ptr(&self) -> u32 {
        self.mesh.colors.as_ptr() as u32
    }
    pub fn mesh_colors_len(&self) -> usize {
        self.mesh.colors.len()
    }

    pub fn mesh_indices_ptr(&self) -> u32 {
        self.mesh.indices.as_ptr() as u32
    }
    pub fn mesh_indices_len(&self) -> usize {
        self.mesh.indices.len()
    }
}

#[wasm_bindgen]
pub struct ScalarFieldMesher {
    nx: usize,
    ny: usize,
    nz: usize,

    chunk_nx: usize,
    chunk_ny: usize,
    chunk_nz: usize,

    // Two keyframes (prev/current) so we can interpolate meshes without
    // re-uploading the scalar field every frame.
    chunk_min_prev: Vec<f32>,
    chunk_max_prev: Vec<f32>,
    chunk_min_next: Vec<f32>,
    chunk_max_next: Vec<f32>,

    scalars_prev: Vec<f32>,
    scalars_next: Vec<f32>,

    has_initialized_keyframes: bool,

    mesh: MeshBuffers,
}

#[wasm_bindgen]
impl ScalarFieldMesher {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let n = nx * ny * nz;

        let cubes_x = nx.saturating_sub(1);
        let cubes_y = ny.saturating_sub(1);
        let cubes_z = nz.saturating_sub(1);

        let chunk_nx = (cubes_x + CHUNK - 1) / CHUNK;
        let chunk_ny = (cubes_y + CHUNK - 1) / CHUNK;
        let chunk_nz = (cubes_z + CHUNK - 1) / CHUNK;
        let chunk_total = chunk_nx * chunk_ny * chunk_nz;

        Self {
            nx,
            ny,
            nz,
            chunk_nx,
            chunk_ny,
            chunk_nz,
            chunk_min_prev: vec![f32::INFINITY; chunk_total],
            chunk_max_prev: vec![f32::NEG_INFINITY; chunk_total],
            chunk_min_next: vec![f32::INFINITY; chunk_total],
            chunk_max_next: vec![f32::NEG_INFINITY; chunk_total],
            scalars_prev: vec![0.0; n],
            scalars_next: vec![0.0; n],
            has_initialized_keyframes: false,
            mesh: MeshBuffers::new(),
        }
    }

    fn chunk_index(&self, cx: usize, cy: usize, cz: usize) -> usize {
        cx + self.chunk_nx * (cy + self.chunk_ny * cz)
    }

    fn reset_chunk_ranges(chunk_v_min: &mut [f32], chunk_v_max: &mut [f32]) {
        chunk_v_min.fill(f32::INFINITY);
        chunk_v_max.fill(f32::NEG_INFINITY);
    }

    fn update_chunk_ranges_for_point(
        chunk_count_x: usize,
        chunk_count_y: usize,
        grid_nx: usize,
        grid_ny: usize,
        grid_nz: usize,
        chunk_min_values: &mut [f32],
        chunk_max_values: &mut [f32],
        x_idx: usize,
        y_idx: usize,
        z_idx: usize,
        scalar: f32,
    ) {
        let cube_count_x = grid_nx.saturating_sub(1);
        let cube_count_y = grid_ny.saturating_sub(1);
        let cube_count_z = grid_nz.saturating_sub(1);

        if cube_count_x == 0 || cube_count_y == 0 || cube_count_z == 0 {
            return;
        }

        let x_left = x_idx.saturating_sub(1);
        let x_right = if x_idx < cube_count_x {
            x_idx
        } else {
            cube_count_x - 1
        };
        let y_left = y_idx.saturating_sub(1);
        let y_right = if y_idx < cube_count_y {
            y_idx
        } else {
            cube_count_y - 1
        };
        let z_left = z_idx.saturating_sub(1);
        let z_right = if z_idx < cube_count_z {
            z_idx
        } else {
            cube_count_z - 1
        };

        let cx0 = x_left / CHUNK;
        let cx1 = x_right / CHUNK;
        let cy0 = y_left / CHUNK;
        let cy1 = y_right / CHUNK;
        let cz0 = z_left / CHUNK;
        let cz1 = z_right / CHUNK;

        let cxs = if cx0 == cx1 { [cx0, cx0] } else { [cx0, cx1] };
        let cys = if cy0 == cy1 { [cy0, cy0] } else { [cy0, cy1] };
        let czs = if cz0 == cz1 { [cz0, cz0] } else { [cz0, cz1] };

        for &cz in &czs {
            for &cy in &cys {
                for &cx in &cxs {
                    let chunk_index = cx + chunk_count_x * (cy + chunk_count_y * cz);
                    let minv = chunk_min_values[chunk_index];
                    let maxv = chunk_max_values[chunk_index];
                    chunk_min_values[chunk_index] = if scalar < minv { scalar } else { minv };
                    chunk_max_values[chunk_index] = if scalar > maxv { scalar } else { maxv };
                }
            }
        }
    }

    pub fn set_v(&mut self, data: &[f32]) {
        if data.len() != self.scalars_next.len() {
            return;
        }

        self.scalars_next.copy_from_slice(data);

        Self::reset_chunk_ranges(&mut self.chunk_min_next, &mut self.chunk_max_next);

        let chunk_count_x = self.chunk_nx;
        let chunk_count_y = self.chunk_ny;
        let grid_nx = self.nx;
        let grid_ny = self.ny;
        let grid_nz = self.nz;

        for z_idx in 0..grid_nz {
            for y_idx in 0..grid_ny {
                for x_idx in 0..grid_nx {
                    let i = idx(grid_nx, grid_ny, x_idx, y_idx, z_idx);
                    Self::update_chunk_ranges_for_point(
                        chunk_count_x,
                        chunk_count_y,
                        grid_nx,
                        grid_ny,
                        grid_nz,
                        &mut self.chunk_min_next,
                        &mut self.chunk_max_next,
                        x_idx,
                        y_idx,
                        z_idx,
                        self.scalars_next[i],
                    );
                }
            }
        }

        if !self.has_initialized_keyframes {
            self.scalars_prev.copy_from_slice(&self.scalars_next);
            self.chunk_min_prev.copy_from_slice(&self.chunk_min_next);
            self.chunk_max_prev.copy_from_slice(&self.chunk_max_next);
            self.has_initialized_keyframes = true;
        }
    }

    pub fn set_v_with_chunk_ranges(
        &mut self,
        data: &[f32],
        chunk_v_min: &[f32],
        chunk_v_max: &[f32],
    ) {
        if data.len() != self.scalars_next.len() {
            return;
        }
        if chunk_v_min.len() != self.chunk_min_next.len() {
            return;
        }
        if chunk_v_max.len() != self.chunk_max_next.len() {
            return;
        }

        self.scalars_next.copy_from_slice(data);
        self.chunk_min_next.copy_from_slice(chunk_v_min);
        self.chunk_max_next.copy_from_slice(chunk_v_max);

        if !self.has_initialized_keyframes {
            self.scalars_prev.copy_from_slice(&self.scalars_next);
            self.chunk_min_prev.copy_from_slice(&self.chunk_min_next);
            self.chunk_max_prev.copy_from_slice(&self.chunk_max_next);
            self.has_initialized_keyframes = true;
        }
    }

    pub fn push_keyframe_with_chunk_ranges(
        &mut self,
        data: &[f32],
        chunk_v_min: &[f32],
        chunk_v_max: &[f32],
    ) {
        if !self.has_initialized_keyframes {
            self.set_v_with_chunk_ranges(data, chunk_v_min, chunk_v_max);
            return;
        }
        if data.len() != self.scalars_next.len() {
            return;
        }
        if chunk_v_min.len() != self.chunk_min_next.len() {
            return;
        }
        if chunk_v_max.len() != self.chunk_max_next.len() {
            return;
        }

        std::mem::swap(&mut self.scalars_prev, &mut self.scalars_next);
        std::mem::swap(&mut self.chunk_min_prev, &mut self.chunk_min_next);
        std::mem::swap(&mut self.chunk_max_prev, &mut self.chunk_max_next);

        self.scalars_next.copy_from_slice(data);
        self.chunk_min_next.copy_from_slice(chunk_v_min);
        self.chunk_max_next.copy_from_slice(chunk_v_max);
    }

    pub fn generate_mesh_visible(
        &mut self,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius: f32,
        iso: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) {
        self.mesh.clear();

        if !self.has_initialized_keyframes {
            return;
        }

        fn floor_div(a: isize, b: isize) -> isize {
            debug_assert!(b > 0);
            if a >= 0 {
                a / b
            } else {
                -((-a + b - 1) / b)
            }
        }

        fn imod(a: isize, m: usize) -> usize {
            let m = m as isize;
            let mut r = a % m;
            if r < 0 {
                r += m;
            }
            r as usize
        }

        let cubes_x = self.nx.saturating_sub(1);
        let cubes_y = self.ny.saturating_sub(1);
        let cubes_z = self.nz.saturating_sub(1);

        if cubes_x == 0 || cubes_y == 0 || cubes_z == 0 {
            return;
        }

        let fx = cubes_x as f32;
        let fy = cubes_y as f32;
        let fz = cubes_z as f32;

        let rad = radius.max(0.0);
        let rad2 = rad * rad;

        // Camera in cube coordinates (can be unbounded in global space).
        let cam_ix = (cam_x + 0.5) * fx;
        let cam_iy = (cam_y + 0.5) * fy;
        let cam_iz = (cam_z + 0.5) * fz;

        let rx = rad * fx;
        let ry = rad * fy;
        let rz = rad * fz;

        let min_x = (cam_ix - rx).floor() as isize;
        let min_y = (cam_iy - ry).floor() as isize;
        let min_z = (cam_iz - rz).floor() as isize;

        let max_x = (cam_ix + rx).ceil() as isize;
        let max_y = (cam_iy + ry).ceil() as isize;
        let max_z = (cam_iz + rz).ceil() as isize;

        let min_cx = floor_div(min_x, CHUNK as isize);
        let min_cy = floor_div(min_y, CHUNK as isize);
        let min_cz = floor_div(min_z, CHUNK as isize);
        let max_cx = floor_div(max_x, CHUNK as isize);
        let max_cy = floor_div(max_y, CHUNK as isize);
        let max_cz = floor_div(max_z, CHUNK as isize);

        let color = [r, g, b, a];

        let near_r = rad * 0.33;
        let mid_r = rad * 0.66;

        fn axis_dist2(p: f32, a0: f32, a1: f32) -> f32 {
            if p < a0 {
                let d = a0 - p;
                d * d
            } else if p > a1 {
                let d = p - a1;
                d * d
            } else {
                0.0
            }
        }

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    let cxw = imod(cx, self.chunk_nx);
                    let cyw = imod(cy, self.chunk_ny);
                    let czw = imod(cz, self.chunk_nz);
                    let ci = cxw + self.chunk_nx * (cyw + self.chunk_ny * czw);

                    let cmin = self.chunk_min_next[ci];
                    let cmax = self.chunk_max_next[ci];
                    if !cmin.is_finite() {
                        continue;
                    }
                    if iso < cmin || iso > cmax {
                        continue;
                    }

                    let x0 = cx * CHUNK as isize;
                    let y0 = cy * CHUNK as isize;
                    let z0 = cz * CHUNK as isize;

                    let x1 = (cx + 1) * CHUNK as isize;
                    let y1 = (cy + 1) * CHUNK as isize;
                    let z1 = (cz + 1) * CHUNK as isize;

                    let wx0 = x0 as f32 / fx - 0.5;
                    let wy0 = y0 as f32 / fy - 0.5;
                    let wz0 = z0 as f32 / fz - 0.5;
                    let wx1 = x1 as f32 / fx - 0.5;
                    let wy1 = y1 as f32 / fy - 0.5;
                    let wz1 = z1 as f32 / fz - 0.5;

                    // Sphere-AABB cull.
                    let d2 = axis_dist2(cam_x, wx0, wx1)
                        + axis_dist2(cam_y, wy0, wy1)
                        + axis_dist2(cam_z, wz0, wz1);
                    if d2 > rad2 {
                        continue;
                    }

                    // LOD stride by chunk-center distance.
                    let cxw = (wx0 + wx1) * 0.5;
                    let cyw = (wy0 + wy1) * 0.5;
                    let czw = (wz0 + wz1) * 0.5;
                    let dist = ((cxw - cam_x) * (cxw - cam_x)
                        + (cyw - cam_y) * (cyw - cam_y)
                        + (czw - cam_z) * (czw - cam_z))
                        .sqrt();

                    let stride = if dist < near_r {
                        1
                    } else if dist < mid_r {
                        2
                    } else {
                        4
                    };

                    mesh_region_append(
                        &self.scalars_next,
                        self.nx,
                        self.ny,
                        self.nz,
                        iso,
                        color,
                        &mut self.mesh,
                        x0,
                        x1,
                        y0,
                        y1,
                        z0,
                        z1,
                        stride,
                    );
                }
            }
        }
    }

    pub fn generate_mesh_visible_lerp(
        &mut self,
        lerp_t: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius: f32,
        iso: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) {
        self.mesh.clear();

        if !self.has_initialized_keyframes {
            return;
        }

        fn floor_div(a: isize, b: isize) -> isize {
            debug_assert!(b > 0);
            if a >= 0 {
                a / b
            } else {
                -((-a + b - 1) / b)
            }
        }

        fn imod(a: isize, m: usize) -> usize {
            let m = m as isize;
            let mut r = a % m;
            if r < 0 {
                r += m;
            }
            r as usize
        }

        let lerp_t = lerp_t.max(0.0).min(1.0);

        let cubes_x = self.nx.saturating_sub(1);
        let cubes_y = self.ny.saturating_sub(1);
        let cubes_z = self.nz.saturating_sub(1);

        if cubes_x == 0 || cubes_y == 0 || cubes_z == 0 {
            return;
        }

        let fx = cubes_x as f32;
        let fy = cubes_y as f32;
        let fz = cubes_z as f32;

        let rad = radius.max(0.0);
        let rad2 = rad * rad;

        // Camera in cube coordinates (can be unbounded in global space).
        let cam_ix = (cam_x + 0.5) * fx;
        let cam_iy = (cam_y + 0.5) * fy;
        let cam_iz = (cam_z + 0.5) * fz;

        let rx = rad * fx;
        let ry = rad * fy;
        let rz = rad * fz;

        let min_x = (cam_ix - rx).floor() as isize;
        let min_y = (cam_iy - ry).floor() as isize;
        let min_z = (cam_iz - rz).floor() as isize;

        let max_x = (cam_ix + rx).ceil() as isize;
        let max_y = (cam_iy + ry).ceil() as isize;
        let max_z = (cam_iz + rz).ceil() as isize;

        let min_cx = floor_div(min_x, CHUNK as isize);
        let min_cy = floor_div(min_y, CHUNK as isize);
        let min_cz = floor_div(min_z, CHUNK as isize);
        let max_cx = floor_div(max_x, CHUNK as isize);
        let max_cy = floor_div(max_y, CHUNK as isize);
        let max_cz = floor_div(max_z, CHUNK as isize);

        let color = [r, g, b, a];

        let near_r = rad * 0.33;
        let mid_r = rad * 0.66;

        fn axis_dist2(p: f32, a0: f32, a1: f32) -> f32 {
            if p < a0 {
                let d = a0 - p;
                d * d
            } else if p > a1 {
                let d = p - a1;
                d * d
            } else {
                0.0
            }
        }

        for cz in min_cz..=max_cz {
            for cy in min_cy..=max_cy {
                for cx in min_cx..=max_cx {
                    let cxw = imod(cx, self.chunk_nx);
                    let cyw = imod(cy, self.chunk_ny);
                    let czw = imod(cz, self.chunk_nz);
                    let ci = cxw + self.chunk_nx * (cyw + self.chunk_ny * czw);

                    // Conservative bounds for the interpolated field.
                    let cmin = self.chunk_min_prev[ci].min(self.chunk_min_next[ci]);
                    let cmax = self.chunk_max_prev[ci].max(self.chunk_max_next[ci]);
                    if !cmin.is_finite() {
                        continue;
                    }
                    if iso < cmin || iso > cmax {
                        continue;
                    }

                    let x0 = cx * CHUNK as isize;
                    let y0 = cy * CHUNK as isize;
                    let z0 = cz * CHUNK as isize;

                    let x1 = (cx + 1) * CHUNK as isize;
                    let y1 = (cy + 1) * CHUNK as isize;
                    let z1 = (cz + 1) * CHUNK as isize;

                    let wx0 = x0 as f32 / fx - 0.5;
                    let wy0 = y0 as f32 / fy - 0.5;
                    let wz0 = z0 as f32 / fz - 0.5;
                    let wx1 = x1 as f32 / fx - 0.5;
                    let wy1 = y1 as f32 / fy - 0.5;
                    let wz1 = z1 as f32 / fz - 0.5;

                    // Sphere-AABB cull.
                    let d2 = axis_dist2(cam_x, wx0, wx1)
                        + axis_dist2(cam_y, wy0, wy1)
                        + axis_dist2(cam_z, wz0, wz1);
                    if d2 > rad2 {
                        continue;
                    }

                    // LOD stride by chunk-center distance.
                    let cxw = (wx0 + wx1) * 0.5;
                    let cyw = (wy0 + wy1) * 0.5;
                    let czw = (wz0 + wz1) * 0.5;
                    let dist = ((cxw - cam_x) * (cxw - cam_x)
                        + (cyw - cam_y) * (cyw - cam_y)
                        + (czw - cam_z) * (czw - cam_z))
                        .sqrt();

                    let stride = if dist < near_r {
                        1
                    } else if dist < mid_r {
                        2
                    } else {
                        4
                    };

                    mesh_region_append_lerp(
                        &self.scalars_prev,
                        &self.scalars_next,
                        lerp_t,
                        self.nx,
                        self.ny,
                        self.nz,
                        iso,
                        color,
                        &mut self.mesh,
                        x0,
                        x1,
                        y0,
                        y1,
                        z0,
                        z1,
                        stride,
                    );
                }
            }
        }
    }

    pub fn mesh_vertex_count(&self) -> usize {
        self.mesh.positions.len() / 3
    }

    pub fn mesh_positions_ptr(&self) -> u32 {
        self.mesh.positions.as_ptr() as u32
    }
    pub fn mesh_positions_len(&self) -> usize {
        self.mesh.positions.len()
    }

    pub fn mesh_normals_ptr(&self) -> u32 {
        self.mesh.normals.as_ptr() as u32
    }
    pub fn mesh_normals_len(&self) -> usize {
        self.mesh.normals.len()
    }

    pub fn mesh_colors_ptr(&self) -> u32 {
        self.mesh.colors.as_ptr() as u32
    }
    pub fn mesh_colors_len(&self) -> usize {
        self.mesh.colors.len()
    }

    pub fn mesh_indices_ptr(&self) -> u32 {
        self.mesh.indices.as_ptr() as u32
    }
    pub fn mesh_indices_len(&self) -> usize {
        self.mesh.indices.len()
    }
}
