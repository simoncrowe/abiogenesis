use wasm_bindgen::prelude::*;

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
}

fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846CA68B);
    x ^ (x >> 16)
}

fn hash3(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    mix32(
        seed ^ (x as u32).wrapping_mul(0x9E3779B1)
            ^ (y as u32).wrapping_mul(0x85EBCA77)
            ^ (z as u32).wrapping_mul(0xC2B2AE3D),
    )
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct ReplicatorMutatorParams {
    // Model size.
    types: u32,

    // Replicator kinetics.
    g_base: f32,
    g_spread: f32,
    d_r: f32,

    // Resource kinetics.
    feed_rate: f32,
    d_f: f32,

    // Mutation (linear chain).
    mu: f32,

    // Diffusion.
    d_r_diff: f32,
    d_f_diff: f32,

    // Optional stability knob.
    substeps: u32,
}

#[wasm_bindgen]
impl ReplicatorMutatorParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            types: 4,

            // Defaults aim for long-lived dynamics without runaway biomass.
            g_base: 0.06,
            g_spread: 0.20,
            d_r: 0.03,

            feed_rate: 0.04,
            d_f: 0.01,

            mu: 0.003,

            d_r_diff: 0.01,
            d_f_diff: 0.20,

            substeps: 2,
        }
    }

    pub fn types(&self) -> u32 {
        self.types
    }

    pub fn set_types(&mut self, v: u32) {
        // Practical bounds for this PoC.
        self.types = v.max(2).min(8);
    }

    pub fn g_base(&self) -> f32 {
        self.g_base
    }
    pub fn g_spread(&self) -> f32 {
        self.g_spread
    }
    pub fn d_r(&self) -> f32 {
        self.d_r
    }

    pub fn feed_rate(&self) -> f32 {
        self.feed_rate
    }
    pub fn d_f(&self) -> f32 {
        self.d_f
    }

    pub fn mu(&self) -> f32 {
        self.mu
    }

    pub fn d_r_diff(&self) -> f32 {
        self.d_r_diff
    }
    pub fn d_f_diff(&self) -> f32 {
        self.d_f_diff
    }

    pub fn substeps(&self) -> u32 {
        self.substeps
    }

    pub fn set_g_base(&mut self, v: f32) {
        self.g_base = v.max(0.0).min(10.0);
    }

    pub fn set_g_spread(&mut self, v: f32) {
        self.g_spread = v.max(0.0).min(10.0);
    }

    pub fn set_d_r(&mut self, v: f32) {
        self.d_r = v.max(0.0).min(10.0);
    }

    pub fn set_feed_rate(&mut self, v: f32) {
        self.feed_rate = v.max(0.0).min(10.0);
    }

    pub fn set_d_f(&mut self, v: f32) {
        self.d_f = v.max(0.0).min(10.0);
    }

    pub fn set_mu(&mut self, v: f32) {
        self.mu = v.max(0.0).min(1.0);
    }

    pub fn set_d_r_diff(&mut self, v: f32) {
        self.d_r_diff = v.max(0.0).min(10.0);
    }

    pub fn set_d_f_diff(&mut self, v: f32) {
        self.d_f_diff = v.max(0.0).min(10.0);
    }

    pub fn set_substeps(&mut self, v: u32) {
        self.substeps = v.max(1).min(16);
    }
}

#[wasm_bindgen]
pub struct ReplicatorMutatorSimulation {
    nx: usize,
    ny: usize,
    nz: usize,
    nxy: usize,

    x_minus: Vec<usize>,
    x_plus: Vec<usize>,
    y_minus: Vec<usize>,
    y_plus: Vec<usize>,
    z_minus: Vec<usize>,
    z_plus: Vec<usize>,

    params: ReplicatorMutatorParams,
    base_seed: u32,
    dt: f32,

    types: usize,

    // Per-type growth coefficients derived from (g_base, g_spread).
    gk: Vec<f32>,

    // Resource source term s(x) per voxel.
    feed_s: Vec<f32>,

    // Chunk min/max for iso culling.
    chunk_nx: usize,
    chunk_ny: usize,
    chunk_nz: usize,
    chunk_v_min: Vec<f32>,
    chunk_v_max: Vec<f32>,

    // Fields.
    r0: Vec<f32>,
    r1: Vec<f32>,

    f0: Vec<f32>,
    f1: Vec<f32>,

    // Exported scalar field (biomass).
    v: Vec<f32>,
    v_dirty: bool,

    rng: LcgRng,
}

#[wasm_bindgen]
impl ReplicatorMutatorSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        seed: u32,
        params: ReplicatorMutatorParams,
    ) -> Self {
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

        let types = params.types.max(2).min(8) as usize;

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
            dt: 0.02,

            types,
            gk: vec![0.0; types],

            feed_s: vec![0.0; n],

            chunk_nx,
            chunk_ny,
            chunk_nz,
            chunk_v_min: vec![f32::INFINITY; chunk_total],
            chunk_v_max: vec![f32::NEG_INFINITY; chunk_total],

            r0: vec![0.0; n * types],
            r1: vec![0.0; n * types],

            f0: vec![0.0; n],
            f1: vec![0.0; n],

            v: vec![0.0; n],
            v_dirty: true,

            rng: LcgRng::new(seed),
        };

        sim.recompute_growth_coeffs();
        sim.set_feed_uniform(sim.params.feed_rate);

        // Default init: low-density noise + resource reservoir.
        sim.seed_uniform(0.02, 0.012, 0.8);
        sim
    }

    pub fn dt(&self) -> f32 {
        self.dt
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt.max(0.00001).min(1.0);
    }

    pub fn types(&self) -> usize {
        self.types
    }

    pub fn seed_uniform(&mut self, noise_amp: f32, r_base: f32, f_init: f32) {
        let noise_amp = noise_amp.max(0.0).min(1.0);
        let r_base = r_base.max(0.0);
        let f_init = f_init.max(0.0);

        self.f0.fill(f_init);

        let types = self.types;
        for i in 0..(self.nx * self.ny * self.nz) {
            let base = i * types;
            for k in 0..types {
                let n = (self.rng.next_f32() - 0.5) * 2.0 * noise_amp;
                let v = (r_base + n).max(0.0);
                self.r0[base + k] = v;
            }
        }

        self.r1.copy_from_slice(&self.r0);
        self.f1.copy_from_slice(&self.f0);
        self.v_dirty = true;
    }

    pub fn seed_regions(&mut self, noise_amp: f32, r_peak: f32, f_init: f32) {
        let noise_amp = noise_amp.max(0.0).min(1.0);
        let r_peak = r_peak.max(0.0);
        let f_init = f_init.max(0.0);

        self.f0.fill(f_init);
        self.r0.fill(0.0);

        let types = self.types;
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;

        // Split along x into K bands.
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = idx(nx, ny, x, y, z);
                    let base = i * types;

                    let k = ((x * types) / nx).min(types - 1);
                    let h = hash3(self.base_seed ^ 0xA11C_EC01, x as i32, y as i32, z as i32);
                    let noise01 = ((h >> 8) as f32) / ((1u32 << 24) as f32);
                    let n = (noise01 - 0.5) * 2.0 * noise_amp;
                    self.r0[base + k] = (r_peak + n).max(0.0);
                }
            }
        }

        self.r1.copy_from_slice(&self.r0);
        self.f1.copy_from_slice(&self.f0);
        self.v_dirty = true;
    }

    pub fn seed_gradient_niches(
        &mut self,
        noise_amp: f32,
        r_base: f32,
        f_init: f32,
        feed_base: f32,
        feed_amp: f32,
        axis: u32,
    ) {
        // Gradient-driven niches: vary the feed source term spatially.
        self.seed_uniform(noise_amp, r_base, f_init);
        self.set_feed_gradient(feed_base, feed_amp, axis);
    }

    pub fn set_feed_uniform(&mut self, rate: f32) {
        let rate = rate.max(0.0);
        self.feed_s.fill(rate);
    }

    pub fn set_feed_gradient(&mut self, base: f32, amp: f32, axis: u32) {
        let base = base.max(0.0);
        let amp = amp.max(0.0);

        let nx = self.nx.max(1) as f32;
        let ny = self.ny.max(1) as f32;
        let nz = self.nz.max(1) as f32;

        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let t = match axis {
                        1 => (y as f32 + 0.5) / ny,
                        2 => (z as f32 + 0.5) / nz,
                        _ => (x as f32 + 0.5) / nx,
                    };
                    // Map t in [0,1] to [-1,1] and apply amplitude.
                    let g = (t * 2.0 - 1.0) * amp;
                    let s = (base * (1.0 + g)).max(0.0);
                    self.feed_s[idx(self.nx, self.ny, x, y, z)] = s;
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

        if self.v_dirty {
            self.update_v_field();
        }

        let nx = self.nx;
        let nxy = self.nxy;
        let v = &self.v;

        let chunk_nx = self.chunk_nx;
        let chunk_ny = self.chunk_ny;
        let chunk_nz = self.chunk_nz;

        self.chunk_v_min
            .par_iter_mut()
            .zip(self.chunk_v_max.par_iter_mut())
            .enumerate()
            .for_each(|(ci, (min_out, max_out))| {
                let cx = ci % chunk_nx;
                let cy = (ci / chunk_nx) % chunk_ny;
                let cz = ci / (chunk_nx * chunk_ny);
                if cz >= chunk_nz {
                    return;
                }

                let x0 = cx * CHUNK;
                let y0 = cy * CHUNK;
                let z0 = cz * CHUNK;

                let x1 = ((cx + 1) * CHUNK).min(cubes_x);
                let y1 = ((cy + 1) * CHUNK).min(cubes_y);
                let z1 = ((cz + 1) * CHUNK).min(cubes_z);

                let mut minv = f32::INFINITY;
                let mut maxv = f32::NEG_INFINITY;

                for z in z0..=z1 {
                    let z_off = z * nxy;
                    for y in y0..=y1 {
                        let row = z_off + y * nx;
                        for x in x0..=x1 {
                            let vv = v[row + x];
                            if vv < minv {
                                minv = vv;
                            }
                            if vv > maxv {
                                maxv = vv;
                            }
                        }
                    }
                }

                *min_out = minv;
                *max_out = maxv;
            });
    }

    pub fn v_ptr(&self) -> u32 {
        self.v.as_ptr() as u32
    }

    pub fn v_len(&self) -> usize {
        self.v.len()
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
}

impl ReplicatorMutatorSimulation {
    pub(crate) fn v_slice(&self) -> &[f32] {
        &self.v
    }

    pub(crate) fn chunk_v_min_slice(&self) -> &[f32] {
        &self.chunk_v_min
    }

    pub(crate) fn chunk_v_max_slice(&self) -> &[f32] {
        &self.chunk_v_max
    }

    fn recompute_growth_coeffs(&mut self) {
        // K is tiny (<=8). Keep this simple and branch-free.
        let k = self.types.max(1) as f32;
        let mid = (k - 1.0) * 0.5;
        let inv_mid = 1.0 / mid.max(1.0);
        let base = self.params.g_base;
        let spread = self.params.g_spread;

        for i in 0..self.types {
            let t = (i as f32 - mid) * inv_mid;
            self.gk[i] = (base * (1.0 + spread * t)).max(0.0);
        }
    }

    fn step_once(&mut self) {
        // If params were mutated from JS between steps, refresh derived coeffs.
        // This is cheap (K<=8), but avoid the call if values didn't change.
        if self.types != (self.params.types as usize) {
            // Types count requires restart; ignore silently here.
        }
        self.recompute_growth_coeffs();

        let sub = self.params.substeps.max(1) as usize;
        let dt = self.dt / (sub as f32);
        for _ in 0..sub {
            self.step_kernel(dt);
        }
    }

    fn update_v_field(&mut self) {
        let nxy = self.nxy;
        let types = self.types;
        let r = &self.r0;

        self.v.par_chunks_mut(nxy).enumerate().for_each(|(z, vz)| {
            let z_off = z * nxy;
            for i in 0..nxy {
                let vi = z_off + i;
                let base = vi * types;
                let mut sum = 0.0;
                for k in 0..types {
                    sum += r[base + k];
                }
                vz[i] = sum;
            }
        });

        self.v_dirty = false;
    }

    fn step_kernel(&mut self, dt: f32) {
        let nx = self.nx;
        let ny = self.ny;
        let nxy = self.nxy;

        let types = self.types;

        let gk = &self.gk;

        let dr = self.params.d_r;
        let df = self.params.d_f;
        let mu = self.params.mu;
        let d_r_diff = self.params.d_r_diff;
        let d_f_diff = self.params.d_f_diff;

        // Avoid Laplacian neighbor reads when diffusion is disabled.
        let do_r_diff = d_r_diff != 0.0;
        let do_f_diff = d_f_diff != 0.0;
        let do_mut = mu != 0.0;

        let x_minus = &self.x_minus;
        let x_plus = &self.x_plus;
        let y_minus = &self.y_minus;
        let y_plus = &self.y_plus;
        let z_minus = &self.z_minus;
        let z_plus = &self.z_plus;

        let feed_s = &self.feed_s;

        let r0 = &self.r0;
        let f0 = &self.f0;
        let r1 = &mut self.r1;
        let f1 = &mut self.f1;

        // Update along z-slices in parallel.
        f1.par_chunks_mut(nxy)
            .zip(r1.par_chunks_mut(nxy * types))
            .enumerate()
            .for_each(|(z, (f1z, r1z))| {
                let z_off = z * nxy;
                let z_off_m = z_minus[z] * nxy;
                let z_off_p = z_plus[z] * nxy;

                for y in 0..ny {
                    let y_off = z_off + y * nx;
                    let y_off_m = z_off + y_minus[y] * nx;
                    let y_off_p = z_off + y_plus[y] * nx;

                    let y_off_zm = z_off_m + y * nx;
                    let y_off_zp = z_off_p + y * nx;

                    let row = y * nx;
                    for x in 0..nx {
                        let c = y_off + x;
                        let base = c * types;

                        let f = f0[c];

                        let c_xm = y_off + x_minus[x];
                        let c_xp = y_off + x_plus[x];
                        let c_ym = y_off_m + x;
                        let c_yp = y_off_p + x;
                        let c_zm = y_off_zm + x;
                        let c_zp = y_off_zp + x;

                        let f_lap = if do_f_diff {
                            f0[c_xm] + f0[c_xp] + f0[c_ym] + f0[c_yp] + f0[c_zm] + f0[c_zp]
                                - 6.0 * f
                        } else {
                            0.0
                        };

                        // Load local replicator values once (K <= 8).
                        let mut r_loc = [0.0f32; 8];
                        for k in 0..types {
                            r_loc[k] = r0[base + k];
                        }

                        // Total consumption term: sum_k g_k * R_k * F.
                        let mut sum_g_r = 0.0;
                        for k in 0..types {
                            sum_g_r += gk[k] * r_loc[k];
                        }
                        let consume = f * sum_g_r;

                        let mut fnxt = f + dt * (feed_s[c] - df * f - consume + d_f_diff * f_lap);

                        if !fnxt.is_finite() {
                            fnxt = 0.0;
                        }
                        fnxt = fnxt.max(0.0);

                        f1z[row + x] = fnxt;

                        // Neighbor bases for r arrays.
                        let b_xm = c_xm * types;
                        let b_xp = c_xp * types;
                        let b_ym = c_ym * types;
                        let b_yp = c_yp * types;
                        let b_zm = c_zm * types;
                        let b_zp = c_zp * types;

                        // Compute Laplacian for all types with contiguous loads from each neighbor.
                        // This is noticeably faster than per-type scattered neighbor access.
                        let mut lap = [0.0f32; 8];
                        if do_r_diff {
                            for k in 0..types {
                                lap[k] = r0[b_xm + k]
                                    + r0[b_xp + k]
                                    + r0[b_ym + k]
                                    + r0[b_yp + k]
                                    + r0[b_zm + k]
                                    + r0[b_zp + k]
                                    - 6.0 * r_loc[k];
                            }
                        }

                        // Replicator updates.
                        let out_base = (row + x) * types;
                        for k in 0..types {
                            let r = r_loc[k];

                            // Linear-chain mutation term (computed from local values).
                            let mut mut_term = 0.0;
                            if do_mut {
                                if k == 0 {
                                    mut_term = mu * (r_loc[1] - r);
                                } else if k + 1 == types {
                                    mut_term = mu * (r_loc[types - 2] - r);
                                } else {
                                    mut_term = mu * (r_loc[k - 1] + r_loc[k + 1] - 2.0 * r);
                                }
                            }

                            let r_lap = if do_r_diff { lap[k] } else { 0.0 };

                            let grow = gk[k] * r * f;
                            let mut rnxt = r + dt * (grow - dr * r + mut_term + d_r_diff * r_lap);

                            if !rnxt.is_finite() {
                                rnxt = 0.0;
                            }
                            rnxt = rnxt.max(0.0);

                            r1z[out_base + k] = rnxt;
                        }
                    }
                }
            });

        std::mem::swap(&mut self.r0, &mut self.r1);
        std::mem::swap(&mut self.f0, &mut self.f1);
        self.v_dirty = true;
    }
}
