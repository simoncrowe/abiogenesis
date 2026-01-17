use wasm_bindgen::prelude::*;

use crate::math::clamp01;
use rayon::prelude::*;

const CHUNK: usize = 16;

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

#[derive(Clone, Copy)]
struct LcgRng {
    state: u32,
}

impl LcgRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes LCG.
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }

    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        // Take 24 high bits -> [0,1).
        let mant = (self.next_u32() >> 8) as u32;
        (mant as f32) / ((1u32 << 24) as f32)
    }

    #[inline(always)]
    fn next_two_f32_16(&mut self) -> (f32, f32) {
        // Two uniforms from one LCG step (16-bit resolution each).
        let x = self.next_u32();
        let a = (x >> 16) as u32;
        let b = (x & 0xFFFF) as u32;
        const INV_65536: f32 = 1.0 / 65536.0;
        (a as f32 * INV_65536, b as f32 * INV_65536)
    }
}

fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846CA68B);
    x ^ (x >> 16)
}

fn seed_for_voxel(base_seed: u32, voxel_index: u32, tick: u32, stage: u32) -> u32 {
    mix32(
        base_seed
            ^ voxel_index.wrapping_mul(0x9E3779B1)
            ^ tick.wrapping_mul(0x85EBCA77)
            ^ stage.wrapping_mul(0xC2B2AE3D),
    )
}

#[inline(always)]
fn normal01_fast(rng: &mut LcgRng) -> f32 {
    // Very cheap normal-ish deviate for visual noise.
    // If u1,u2 ~ U(0,1), then (u1+u2-1) is triangular with var = 1/6.
    // Scale by sqrt(6) to get unit variance.
    //
    // Uses two 16-bit uniforms from a single LCG step to minimize RNG cost.
    const SQRT6: f32 = 2.449_489_8;
    let (u1, u2) = rng.next_two_f32_16();
    (u1 + u2 - 1.0) * SQRT6
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct StochasticRdmeParams {
    // Diffusion coefficients.
    df: f32,
    da: f32,
    di: f32,

    // Reaction rates.
    k1: f32,
    k2: f32,
    k3: f32,

    // Feed field.
    feed_base: f32,
    feed_noise_amp: f32,
    feed_noise_scale: u32,

    // Decay.
    d_a: f32,
    d_i: f32,
    d_f: f32,

    // Intrinsic noise scale.
    eta_scale: f32,

    // Output mapping.
    aliveness_alpha: f32,
    aliveness_gain: f32,

    // Optional internal substeps (stability knob).
    substeps: u32,
}

#[wasm_bindgen]
impl StochasticRdmeParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            // Defaults match the earlier RDME MVP-style tuning (now using f32 fields).
            df: 0.2,
            da: 0.05,
            di: 0.02,

            k1: 0.002,
            k2: 0.02,
            k3: 0.001,

            feed_base: 2.0,
            feed_noise_amp: 0.35,
            feed_noise_scale: 8,

            d_a: 0.01,
            d_i: 0.005,
            d_f: 0.0,

            eta_scale: 0.25,

            aliveness_alpha: 0.25,
            aliveness_gain: 0.05,

            substeps: 1,
        }
    }

    pub fn df(&self) -> f32 {
        self.df
    }
    pub fn da(&self) -> f32 {
        self.da
    }
    pub fn di(&self) -> f32 {
        self.di
    }

    pub fn k1(&self) -> f32 {
        self.k1
    }
    pub fn k2(&self) -> f32 {
        self.k2
    }
    pub fn k3(&self) -> f32 {
        self.k3
    }

    pub fn feed_base(&self) -> f32 {
        self.feed_base
    }
    pub fn feed_noise_amp(&self) -> f32 {
        self.feed_noise_amp
    }
    pub fn feed_noise_scale(&self) -> u32 {
        self.feed_noise_scale
    }

    pub fn d_a(&self) -> f32 {
        self.d_a
    }
    pub fn d_i(&self) -> f32 {
        self.d_i
    }
    pub fn d_f(&self) -> f32 {
        self.d_f
    }

    pub fn eta_scale(&self) -> f32 {
        self.eta_scale
    }

    pub fn aliveness_alpha(&self) -> f32 {
        self.aliveness_alpha
    }
    pub fn aliveness_gain(&self) -> f32 {
        self.aliveness_gain
    }

    pub fn substeps(&self) -> u32 {
        self.substeps
    }

    pub fn set_df(&mut self, v: f32) {
        self.df = v.max(0.0);
    }
    pub fn set_da(&mut self, v: f32) {
        self.da = v.max(0.0);
    }
    pub fn set_di(&mut self, v: f32) {
        self.di = v.max(0.0);
    }

    pub fn set_k1(&mut self, v: f32) {
        self.k1 = v.max(0.0);
    }
    pub fn set_k2(&mut self, v: f32) {
        self.k2 = v.max(0.0);
    }
    pub fn set_k3(&mut self, v: f32) {
        self.k3 = v.max(0.0);
    }

    pub fn set_feed_base(&mut self, v: f32) {
        self.feed_base = v.max(0.0);
    }
    pub fn set_feed_noise_amp(&mut self, v: f32) {
        self.feed_noise_amp = v.max(0.0);
    }
    pub fn set_feed_noise_scale(&mut self, v: u32) {
        self.feed_noise_scale = v.max(1);
    }

    pub fn set_d_a(&mut self, v: f32) {
        self.d_a = v.max(0.0);
    }
    pub fn set_d_i(&mut self, v: f32) {
        self.d_i = v.max(0.0);
    }
    pub fn set_d_f(&mut self, v: f32) {
        self.d_f = v.max(0.0);
    }

    pub fn set_eta_scale(&mut self, v: f32) {
        self.eta_scale = v.max(0.0);
    }

    pub fn set_aliveness_alpha(&mut self, v: f32) {
        self.aliveness_alpha = v.max(0.0);
    }
    pub fn set_aliveness_gain(&mut self, v: f32) {
        self.aliveness_gain = v.max(0.00001);
    }

    pub fn set_substeps(&mut self, v: u32) {
        self.substeps = v.max(1).min(16);
    }
}

#[wasm_bindgen]
pub struct StochasticRdmeSimulation {
    nx: usize,
    ny: usize,
    nz: usize,
    nxy: usize,

    // Precomputed wrapped neighbor indices.
    x_minus: Vec<usize>,
    x_plus: Vec<usize>,
    y_minus: Vec<usize>,
    y_plus: Vec<usize>,
    z_minus: Vec<usize>,
    z_plus: Vec<usize>,

    params: StochasticRdmeParams,
    base_seed: u32,
    dt: f32,
    tick: u32,

    chunk_nx: usize,
    chunk_ny: usize,
    chunk_nz: usize,
    chunk_v_min: Vec<f32>,
    chunk_v_max: Vec<f32>,

    // Species state (concentration-like, f32).
    f: Vec<f32>,
    a: Vec<f32>,
    i: Vec<f32>,
    f2: Vec<f32>,
    a2: Vec<f32>,
    i2: Vec<f32>,

    // Precomputed feed field per voxel.
    feed: Vec<f32>,

    // Exported scalar field (same contract as Gray–Scott V).
    v: Vec<f32>,
    v_dirty: bool,
}

#[wasm_bindgen]
impl StochasticRdmeSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: usize, ny: usize, nz: usize, seed: u32, params: StochasticRdmeParams) -> Self {
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

        let mut sim = Self {
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
            dt: 0.05,
            tick: 0,
            chunk_nx,
            chunk_ny,
            chunk_nz,
            chunk_v_min: vec![f32::INFINITY; chunk_total],
            chunk_v_max: vec![f32::NEG_INFINITY; chunk_total],
            f: vec![0.0; n],
            a: vec![0.0; n],
            i: vec![0.0; n],
            f2: vec![0.0; n],
            a2: vec![0.0; n],
            i2: vec![0.0; n],
            feed: vec![0.0; n],
            v: vec![0.0; n],
            v_dirty: true,
        };

        sim.recompute_feed_cache();

        // A reasonable default seed matching prior behavior.
        sim.seed_spheres(0.05, 20, 50, 0, 0, 25, 20, 0, 0.02);
        sim
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
        self.dt = dt.max(0.0001).min(1.0);
    }

    pub fn step(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step_once();
        }
    }

    fn chunk_index(&self, cx: usize, cy: usize, cz: usize) -> usize {
        cx + self.chunk_nx * (cy + self.chunk_ny * cz)
    }

    pub fn recompute_chunk_ranges_from_v(&mut self) {
        let cubes_x = self.nx.saturating_sub(1);
        let cubes_y = self.ny.saturating_sub(1);
        let cubes_z = self.nz.saturating_sub(1);

        if cubes_x == 0 || cubes_y == 0 || cubes_z == 0 {
            return;
        }

        if self.v_dirty {
            self.update_aliveness_field();
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

    pub fn v_ptr(&self) -> u32 {
        self.v.as_ptr() as u32
    }

    pub fn v_len(&self) -> usize {
        self.v.len()
    }

    pub(crate) fn v_slice(&self) -> &[f32] {
        &self.v
    }

    pub(crate) fn chunk_v_min_slice(&self) -> &[f32] {
        &self.chunk_v_min
    }

    pub(crate) fn chunk_v_max_slice(&self) -> &[f32] {
        &self.chunk_v_max
    }

    pub fn chunk_v_min_ptr(&self) -> u32 {
        self.chunk_v_min.as_ptr() as u32
    }

    pub fn chunk_v_max_ptr(&self) -> u32 {
        self.chunk_v_max.as_ptr() as u32
    }

    pub fn chunk_v_len(&self) -> usize {
        self.chunk_v_min.len()
    }

    /// Seed the lattice with a background concentration and random catalyst spheres.
    ///
    /// This keeps the same JS contract as the previous RDME MVP (u32 "counts"),
    /// but stores the state as f32 fields.
    pub fn seed_spheres(
        &mut self,
        radius01: f32,
        sphere_count: u32,
        base_f: u32,
        base_a: u32,
        base_i: u32,
        sphere_f: u32,
        sphere_a: u32,
        sphere_i: u32,
        a_noise_prob: f32,
    ) {
        let n = self.nx * self.ny * self.nz;
        if n == 0 {
            return;
        }

        self.f.fill(base_f as f32);
        self.a.fill(base_a as f32);
        self.i.fill(base_i as f32);

        let mut rng = LcgRng::new(self.base_seed ^ 0xA11CE117);
        let noise_p = a_noise_prob.max(0.0).min(1.0);
        if noise_p > 0.0 {
            for a in &mut self.a {
                if rng.next_f32() < noise_p {
                    *a += 1.0;
                }
            }
        }

        let min_dim = self.nx.min(self.ny).min(self.nz).max(1);
        let r = (min_dim as f32 * radius01).max(1.0) as isize;
        let r2 = (r * r) as i64;

        for _ in 0..sphere_count {
            let cx = (rng.next_f32() * (self.nx.max(1) as f32)) as isize;
            let cy = (rng.next_f32() * (self.ny.max(1) as f32)) as isize;
            let cz = (rng.next_f32() * (self.nz.max(1) as f32)) as isize;

            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        let d2 = (dx * dx + dy * dy + dz * dz) as i64;
                        if d2 > r2 {
                            continue;
                        }
                        let x = wrap(cx + dx, self.nx);
                        let y = wrap(cy + dy, self.ny);
                        let z = wrap(cz + dz, self.nz);
                        let p = idx(self.nx, self.ny, x, y, z);
                        self.f[p] = sphere_f as f32;
                        self.a[p] = sphere_a as f32;
                        self.i[p] = sphere_i as f32;
                    }
                }
            }
        }

        self.update_aliveness_field();
    }

    fn recompute_feed_cache(&mut self) {
        let n = self.nx * self.ny * self.nz;
        if n == 0 {
            return;
        }

        let scale = self.params.feed_noise_scale.max(1) as usize;
        let bx = (self.nx + scale - 1) / scale;
        let by = (self.ny + scale - 1) / scale;
        let bz = (self.nz + scale - 1) / scale;

        let mut blocks = vec![0.0f32; bx * by * bz];
        for bz_i in 0..bz {
            for by_i in 0..by {
                for bx_i in 0..bx {
                    let h = mix32(
                        (self.base_seed ^ 0xFEEDBEEFu32)
                            ^ (bx_i as u32).wrapping_mul(0x9E3779B1)
                            ^ (by_i as u32).wrapping_mul(0x85EBCA77)
                            ^ (bz_i as u32).wrapping_mul(0xC2B2AE3D),
                    );
                    let noise01 = ((h >> 8) as f32) / ((1u32 << 24) as f32);
                    let signed = noise01 * 2.0 - 1.0;
                    let feed = (self.params.feed_base
                        * (1.0 + self.params.feed_noise_amp * signed))
                        .max(0.0);
                    blocks[bx_i + bx * (by_i + by * bz_i)] = feed;
                }
            }
        }

        for z in 0..self.nz {
            let bz_i = z / scale;
            let z_off = z * self.nxy;
            for y in 0..self.ny {
                let by_i = y / scale;
                let y_off = z_off + y * self.nx;
                let base = bx * (by_i + by * bz_i);
                for x in 0..self.nx {
                    let bx_i = x / scale;
                    self.feed[y_off + x] = blocks[bx_i + base];
                }
            }
        }
    }

    fn update_aliveness_field(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nxy = self.nxy;

        let alpha = self.params.aliveness_alpha;
        let gain = self.params.aliveness_gain;

        let a = &self.a;
        let i = &self.i;

        // Slice-parallel mapping to avoid per-element enumerate/index overhead.
        self.v
            .par_chunks_mut(nxy)
            .enumerate()
            .for_each(|(z, v_out_z)| {
                if z >= nz {
                    return;
                }
                let z_off = z * nxy;
                let a_z = &a[z_off..z_off + nxy];
                let i_z = &i[z_off..z_off + nxy];

                for y in 0..ny {
                    let row_off = y * nx;
                    for x in 0..nx {
                        let p = row_off + x;
                        let raw = (a_z[p] - alpha * i_z[p]).max(0.0);
                        let v = 1.0 - (-gain * raw).exp();
                        v_out_z[p] = clamp01(v);
                    }
                }
            });

        self.v_dirty = false;
    }

    fn step_kernel(&mut self, dt: f32, stage: u32) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nxy = self.nxy;

        let x_minus = &self.x_minus;
        let x_plus = &self.x_plus;
        let y_minus = &self.y_minus;
        let y_plus = &self.y_plus;
        let z_minus = &self.z_minus;
        let z_plus = &self.z_plus;

        let df = self.params.df;
        let da = self.params.da;
        let di = self.params.di;

        let k1 = self.params.k1;
        let k2 = self.params.k2;
        let k3 = self.params.k3;

        let d_a = self.params.d_a;
        let d_i = self.params.d_i;
        let d_f = self.params.d_f;

        let eta = self.params.eta_scale;
        let sqrt_dt = dt.sqrt();

        let base_seed = self.base_seed;
        let tick = self.tick;

        let feed = &self.feed;

        let f0 = &self.f;
        let a0 = &self.a;
        let i0 = &self.i;

        let f1 = &mut self.f2;
        let a1 = &mut self.a2;
        let i1 = &mut self.i2;

        // Parallelize over z-slices to avoid per-voxel div/mod for (x,y,z).
        f1.par_chunks_mut(nxy)
            .zip(a1.par_chunks_mut(nxy))
            .zip(i1.par_chunks_mut(nxy))
            .enumerate()
            .for_each(|(z, ((f_out_z, a_out_z), i_out_z))| {
                if z >= nz {
                    return;
                }

                let z_off = z * nxy;
                let zm_off = z_minus[z] * nxy;
                let zp_off = z_plus[z] * nxy;

                let f0z = &f0[z_off..z_off + nxy];
                let a0z = &a0[z_off..z_off + nxy];
                let i0z = &i0[z_off..z_off + nxy];

                let f0zm = &f0[zm_off..zm_off + nxy];
                let a0zm = &a0[zm_off..zm_off + nxy];
                let i0zm = &i0[zm_off..zm_off + nxy];

                let f0zp = &f0[zp_off..zp_off + nxy];
                let a0zp = &a0[zp_off..zp_off + nxy];
                let i0zp = &i0[zp_off..zp_off + nxy];

                let feedz = &feed[z_off..z_off + nxy];

                for y in 0..ny {
                    let row_off = y * nx;
                    let ym_row_off = y_minus[y] * nx;
                    let yp_row_off = y_plus[y] * nx;

                    for x in 0..nx {
                        let p = row_off + x;

                        let f_c = f0z[p];
                        let a_c = a0z[p];
                        let i_c = i0z[p];

                        let xm = row_off + x_minus[x];
                        let xp = row_off + x_plus[x];
                        let ym = ym_row_off + x;
                        let yp = yp_row_off + x;

                        // Laplacians (h=1).
                        let lap_f =
                            f0z[xm] + f0z[xp] + f0z[ym] + f0z[yp] + f0zm[p] + f0zp[p] - 6.0 * f_c;
                        let lap_a =
                            a0z[xm] + a0z[xp] + a0z[ym] + a0z[yp] + a0zm[p] + a0zp[p] - 6.0 * a_c;
                        let lap_i =
                            i0z[xm] + i0z[xp] + i0z[ym] + i0z[yp] + i0zm[p] + i0zp[p] - 6.0 * i_c;

                        let diff_f = df * lap_f;
                        let diff_a = da * lap_a;
                        let diff_i = di * lap_i;

                        // Reaction drift.
                        let s = feedz[p];

                        let r_autocat = k1 * a_c * f_c;
                        let r_inhib_prod = k2 * a_c;
                        let r_inhibit = k3 * a_c * i_c;

                        let r_decay_a = d_a * a_c;
                        let r_decay_i = d_i * i_c;
                        let r_decay_f = d_f * f_c;

                        let drift_f = s - r_autocat - r_decay_f;
                        let drift_a = r_autocat - r_inhibit - r_decay_a;
                        let drift_i = r_inhib_prod - r_decay_i;

                        // Intrinsic noise (CLE-style).
                        let mut n_f = 0.0f32;
                        let mut n_a = 0.0f32;
                        let mut n_i = 0.0f32;

                        if eta > 0.0 {
                            let voxel_index = (z_off + p) as u32;
                            let mut rng =
                                LcgRng::new(seed_for_voxel(base_seed, voxel_index, tick, stage));

                            // Reuse a small number of deviates across channels.
                            // This trades a bit of independence for speed, but preserves the
                            // overall "intrinsic noise" character.
                            let z0 = normal01_fast(&mut rng);
                            let z1 = normal01_fast(&mut rng);
                            let z2 = normal01_fast(&mut rng);

                            // Feed: +F
                            let sigma = eta * s.max(0.0).sqrt() * sqrt_dt;
                            n_f += sigma * z0;

                            // Autocatalysis: A+F -> 2A  (F -1, A +1)
                            let sigma = eta * r_autocat.max(0.0).sqrt() * sqrt_dt;
                            n_f -= sigma * z1;
                            n_a += sigma * z1;

                            // Inhibitor production: A -> A + I (I +1)
                            let sigma = eta * r_inhib_prod.max(0.0).sqrt() * sqrt_dt;
                            n_i += sigma * z2;

                            // Inhibition: A + I -> I (A -1)
                            let sigma = eta * r_inhibit.max(0.0).sqrt() * sqrt_dt;
                            n_a -= sigma * z0;

                            // Decay noise.
                            let sigma = eta * r_decay_a.max(0.0).sqrt() * sqrt_dt;
                            n_a -= sigma * z1;

                            let sigma = eta * r_decay_i.max(0.0).sqrt() * sqrt_dt;
                            n_i -= sigma * z2;

                            if d_f > 0.0 {
                                let sigma = eta * r_decay_f.max(0.0).sqrt() * sqrt_dt;
                                n_f -= sigma * z0;
                            }
                        }

                        let mut f_new = f_c + dt * (drift_f + diff_f) + n_f;
                        let mut a_new = a_c + dt * (drift_a + diff_a) + n_a;
                        let mut i_new = i_c + dt * (drift_i + diff_i) + n_i;

                        if !f_new.is_finite() {
                            f_new = 0.0;
                        }
                        if !a_new.is_finite() {
                            a_new = 0.0;
                        }
                        if !i_new.is_finite() {
                            i_new = 0.0;
                        }

                        // Positivity clamp.
                        if f_new < 0.0 {
                            f_new = 0.0;
                        }
                        if a_new < 0.0 {
                            a_new = 0.0;
                        }
                        if i_new < 0.0 {
                            i_new = 0.0;
                        }

                        f_out_z[p] = f_new;
                        a_out_z[p] = a_new;
                        i_out_z[p] = i_new;
                    }
                }
            });

        std::mem::swap(&mut self.f, &mut self.f2);
        std::mem::swap(&mut self.a, &mut self.a2);
        std::mem::swap(&mut self.i, &mut self.i2);
    }

    fn step_once(&mut self) {
        let dt = self.dt;
        let substeps = self.params.substeps.max(1) as usize;
        let h = dt / (substeps as f32);

        for s in 0..substeps {
            self.step_kernel(h, 0xC1E0_0000u32 ^ (s as u32));
            self.tick = self.tick.wrapping_add(1);
        }

        self.v_dirty = true;
    }
}
