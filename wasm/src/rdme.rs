use wasm_bindgen::prelude::*;

use crate::math::clamp01;

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

fn mix32(mut x: u32) -> u32 {
    // Same style of integer mixing as used elsewhere in the repo.
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

fn normal01(rng: &mut LcgRng) -> f32 {
    // Cheap normal-ish generator for tau-leaping approximations.
    // CLT: sum of 12 U(0,1) - 6 ~ N(0,1).
    // Avoids wasm `ln`/`cos` costs from Box–Muller.
    let mut s = 0.0;
    for _ in 0..12 {
        s += rng.next_f32();
    }
    s - 6.0
}

fn poisson(rng: &mut LcgRng, lambda: f32) -> u32 {
    if !(lambda > 0.0) {
        return 0;
    }

    // Knuth for small lambda.
    if lambda < 10.0 {
        let l = (-lambda).exp();
        let mut k: u32 = 0;
        let mut p: f32 = 1.0;
        loop {
            k = k.wrapping_add(1);
            p *= rng.next_f32();
            if p <= l {
                return k - 1;
            }
        }
    }

    // Normal approximation for moderate/large lambda.
    let z = normal01(rng);
    let k = (lambda + lambda.sqrt() * z).round();
    if k <= 0.0 {
        0
    } else if k >= (u32::MAX as f32) {
        u32::MAX
    } else {
        k as u32
    }
}

fn binomial(rng: &mut LcgRng, n: u32, p: f32) -> u32 {
    if n == 0 {
        return 0;
    }
    let p = p.max(0.0).min(1.0);
    if p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }

    // Exact Bernoulli sum for small n.
    if n < 25 {
        let mut k = 0;
        for _ in 0..n {
            if rng.next_f32() < p {
                k += 1;
            }
        }
        return k;
    }

    // Normal approximation.
    let nf = n as f32;
    let mean = nf * p;
    let var = nf * p * (1.0 - p);
    let z = normal01(rng);
    let k = (mean + var.sqrt() * z).round();
    if k <= 0.0 {
        0
    } else if k >= nf {
        n
    } else {
        k as u32
    }
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

    // Output mapping.
    aliveness_alpha: f32,
    aliveness_gain: f32,
}

#[wasm_bindgen]
impl StochasticRdmeParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
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
            aliveness_alpha: 0.25,
            aliveness_gain: 0.05,
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

    pub fn aliveness_alpha(&self) -> f32 {
        self.aliveness_alpha
    }
    pub fn aliveness_gain(&self) -> f32 {
        self.aliveness_gain
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

    pub fn set_aliveness_alpha(&mut self, v: f32) {
        self.aliveness_alpha = v.max(0.0);
    }
    pub fn set_aliveness_gain(&mut self, v: f32) {
        self.aliveness_gain = v.max(0.00001);
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

    // Species counts.
    f: Vec<u32>,
    a: Vec<u32>,
    i: Vec<u32>,
    f2: Vec<u32>,
    a2: Vec<u32>,
    i2: Vec<u32>,

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
            f: vec![0; n],
            a: vec![0; n],
            i: vec![0; n],
            f2: vec![0; n],
            a2: vec![0; n],
            i2: vec![0; n],
            feed: vec![0.0; n],
            v: vec![0.0; n],
            v_dirty: true,
        };

        sim.recompute_feed_cache();

        // A reasonable default seed.
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
        self.dt = dt.max(0.001).min(1.0);
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

    /// Seed the lattice with a background concentration and random catalyst spheres.
    ///
    /// Background:
    /// - `base_f`, `base_a`, `base_i` counts everywhere.
    ///
    /// Spheres:
    /// - `sphere_count` spheres of radius `radius01` (fraction of dims)
    /// - inside each sphere: set counts to (`sphere_f`, `sphere_a`, `sphere_i`)
    ///
    /// Noise:
    /// - `a_noise_prob`: per-voxel probability of adding +1 to A.
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

        self.f.fill(base_f);
        self.a.fill(base_a);
        self.i.fill(base_i);

        let mut rng = LcgRng::new(self.base_seed ^ 0xA11CE117);
        let noise_p = a_noise_prob.max(0.0).min(1.0);
        if noise_p > 0.0 {
            for a in &mut self.a {
                if rng.next_f32() < noise_p {
                    *a = a.saturating_add(1);
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
                        self.f[p] = sphere_f;
                        self.a[p] = sphere_a;
                        self.i[p] = sphere_i;
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

    fn seed_for_stage(&self, tick: u32, stage: u32) -> u32 {
        mix32(self.base_seed ^ tick.wrapping_mul(0x85EBCA77) ^ stage.wrapping_mul(0xC2B2AE3D))
    }

    fn reaction_step(&mut self, dt: f32, stage: u32) {
        let mut rng = LcgRng::new(self.seed_for_stage(self.tick, stage));

        let k1 = self.params.k1;
        let k2 = self.params.k2;
        let k3 = self.params.k3;

        let decay_a_p = if self.params.d_a > 0.0 {
            1.0 - (-self.params.d_a * dt).exp()
        } else {
            0.0
        };
        let decay_i_p = if self.params.d_i > 0.0 {
            1.0 - (-self.params.d_i * dt).exp()
        } else {
            0.0
        };
        let decay_f_p = if self.params.d_f > 0.0 {
            1.0 - (-self.params.d_f * dt).exp()
        } else {
            0.0
        };

        for c in 0..self.f.len() {
            let mut f = self.f[c];
            let mut a = self.a[c];
            let mut i = self.i[c];

            // 1) Feed: Ø -> F
            let add_f = poisson(&mut rng, self.feed[c] * dt);
            f = f.saturating_add(add_f);

            // 2) Autocatalysis: A + F -> 2A
            if a > 0 && f > 0 && k1 > 0.0 {
                let lam = k1 * (a as f32) * (f as f32) * dt;
                let k = poisson(&mut rng, lam).min(a.min(f));
                f -= k;
                a = a.saturating_add(k);
            }

            // 3) Inhibitor production: A -> A + I
            if a > 0 && k2 > 0.0 {
                let lam = k2 * (a as f32) * dt;
                let k = poisson(&mut rng, lam);
                i = i.saturating_add(k);
            }

            // 4) Inhibition: A + I -> I (consumes A)
            if a > 0 && i > 0 && k3 > 0.0 {
                let lam = k3 * (a as f32) * (i as f32) * dt;
                let k = poisson(&mut rng, lam).min(a.min(i));
                a -= k;
            }

            // 5) Decay.
            if a > 0 && decay_a_p > 0.0 {
                a -= binomial(&mut rng, a, decay_a_p);
            }
            if i > 0 && decay_i_p > 0.0 {
                i -= binomial(&mut rng, i, decay_i_p);
            }
            if f > 0 && decay_f_p > 0.0 {
                f -= binomial(&mut rng, f, decay_f_p);
            }

            self.f[c] = f;
            self.a[c] = a;
            self.i[c] = i;
        }
    }

    fn diffuse_species(
        nx: usize,
        ny: usize,
        nz: usize,
        nxy: usize,
        dt: f32,
        base_seed: u32,
        tick: u32,
        stage: u32,
        x_minus: &[usize],
        x_plus: &[usize],
        y_minus: &[usize],
        y_plus: &[usize],
        z_minus: &[usize],
        z_plus: &[usize],
        cur: &[u32],
        next: &mut [u32],
        d: f32,
    ) {
        next.fill(0);

        if !(d > 0.0) {
            next.copy_from_slice(cur);
            return;
        }

        // For MVP we use h=1.
        let rate = 6.0 * d;
        let p_leave = 1.0 - (-rate * dt).exp();
        if !(p_leave > 0.0) {
            next.copy_from_slice(cur);
            return;
        }

        let mut rng = LcgRng::new(seed_for_voxel(base_seed, 0, tick, stage));

        for z in 0..nz {
            let z_off = z * nxy;
            let z_off_m = z_minus[z] * nxy;
            let z_off_p = z_plus[z] * nxy;

            for y in 0..ny {
                let y_off = z_off + y * nx;
                let y_off_m = z_off + y_minus[y] * nx;
                let y_off_p = z_off + y_plus[y] * nx;
                let y_off_zm = z_off_m + y * nx;
                let y_off_zp = z_off_p + y * nx;

                for x in 0..nx {
                    let c = y_off + x;
                    let n0 = cur[c];
                    if n0 == 0 {
                        continue;
                    }

                    let l = binomial(&mut rng, n0, p_leave);
                    let stay = n0 - l;
                    next[c] = next[c].saturating_add(stay);

                    // Sequential multinomial splitting into 6 equal directions.
                    let mut rem = l;
                    let out0 = binomial(&mut rng, rem, 1.0 / 6.0);
                    rem -= out0;
                    let out1 = binomial(&mut rng, rem, 1.0 / 5.0);
                    rem -= out1;
                    let out2 = binomial(&mut rng, rem, 1.0 / 4.0);
                    rem -= out2;
                    let out3 = binomial(&mut rng, rem, 1.0 / 3.0);
                    rem -= out3;
                    let out4 = binomial(&mut rng, rem, 1.0 / 2.0);
                    let out5 = rem - out4;

                    let xm = y_off + x_minus[x];
                    let xp = y_off + x_plus[x];
                    let ym = y_off_m + x;
                    let yp = y_off_p + x;
                    let zm = y_off_zm + x;
                    let zp = y_off_zp + x;

                    next[xm] = next[xm].saturating_add(out0);
                    next[xp] = next[xp].saturating_add(out1);
                    next[ym] = next[ym].saturating_add(out2);
                    next[yp] = next[yp].saturating_add(out3);
                    next[zm] = next[zm].saturating_add(out4);
                    next[zp] = next[zp].saturating_add(out5);
                }
            }
        }
    }

    fn diffusion_step(&mut self) {
        let df = self.params.df;
        let da = self.params.da;
        let di = self.params.di;

        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nxy = self.nxy;
        let dt = self.dt;
        let base_seed = self.base_seed;
        let tick = self.tick;

        let x_minus = &self.x_minus;
        let x_plus = &self.x_plus;
        let y_minus = &self.y_minus;
        let y_plus = &self.y_plus;
        let z_minus = &self.z_minus;
        let z_plus = &self.z_plus;

        Self::diffuse_species(
            nx,
            ny,
            nz,
            nxy,
            dt,
            base_seed,
            tick,
            0xD1FFu32,
            x_minus,
            x_plus,
            y_minus,
            y_plus,
            z_minus,
            z_plus,
            &self.f,
            &mut self.f2,
            df,
        );
        Self::diffuse_species(
            nx,
            ny,
            nz,
            nxy,
            dt,
            base_seed,
            tick,
            0xD1FFu32 ^ 0xA,
            x_minus,
            x_plus,
            y_minus,
            y_plus,
            z_minus,
            z_plus,
            &self.a,
            &mut self.a2,
            da,
        );
        Self::diffuse_species(
            nx,
            ny,
            nz,
            nxy,
            dt,
            base_seed,
            tick,
            0xD1FFu32 ^ 0xB,
            x_minus,
            x_plus,
            y_minus,
            y_plus,
            z_minus,
            z_plus,
            &self.i,
            &mut self.i2,
            di,
        );

        std::mem::swap(&mut self.f, &mut self.f2);
        std::mem::swap(&mut self.a, &mut self.a2);
        std::mem::swap(&mut self.i, &mut self.i2);
    }

    fn update_aliveness_field(&mut self) {
        let alpha = self.params.aliveness_alpha;
        let gain = self.params.aliveness_gain;
        for idx in 0..self.v.len() {
            let raw = ((self.a[idx] as f32) - alpha * (self.i[idx] as f32)).max(0.0);
            let v = 1.0 - (-gain * raw).exp();
            self.v[idx] = clamp01(v);
        }
        self.v_dirty = false;
    }

    fn step_once(&mut self) {
        let dt = self.dt;
        let half = 0.5 * dt;

        // Strang splitting.
        self.reaction_step(half, 0xFACE_1234u32);
        self.diffusion_step();
        self.reaction_step(half, 0xFACE_1234u32 ^ 0x1);

        self.v_dirty = true;
        self.tick = self.tick.wrapping_add(1);
    }
}
