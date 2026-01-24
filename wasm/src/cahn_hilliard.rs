use wasm_bindgen::prelude::*;

use rayon::prelude::*;

const CHUNK: usize = 16;

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

#[wasm_bindgen]
#[derive(Clone)]
pub struct CahnHilliardParams {
    a: f32,
    kappa: f32,
    m: f32,
    substeps: u32,

    // 2: two-pass update (default)
    // 1: one-pass approximate update (faster, less faithful; not strictly mass-conserving)
    pass_mode: u32,

    // Only used when pass_mode == 2.
    // 0: full two-pass CH
    // 1: approximate two-pass update
    approx_mode: u32,

    // Simple spinodal seeding controls.
    phi_mean: f32,
    noise_amp: f32,
}

#[wasm_bindgen]
impl CahnHilliardParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            a: 1.0,
            kappa: 1.0,
            m: 0.2,
            substeps: 4,
            pass_mode: 2,
            approx_mode: 0,
            phi_mean: 0.0,
            noise_amp: 0.01,
        }
    }

    pub fn a(&self) -> f32 {
        self.a
    }
    pub fn kappa(&self) -> f32 {
        self.kappa
    }
    pub fn m(&self) -> f32 {
        self.m
    }
    pub fn substeps(&self) -> u32 {
        self.substeps
    }
    pub fn pass_mode(&self) -> u32 {
        self.pass_mode
    }
    pub fn approx_mode(&self) -> u32 {
        self.approx_mode
    }
    pub fn phi_mean(&self) -> f32 {
        self.phi_mean
    }
    pub fn noise_amp(&self) -> f32 {
        self.noise_amp
    }

    pub fn set_a(&mut self, v: f32) {
        self.a = v.max(0.0);
    }
    pub fn set_kappa(&mut self, v: f32) {
        self.kappa = v.max(0.0);
    }
    pub fn set_m(&mut self, v: f32) {
        self.m = v.max(0.0);
    }

    pub fn set_substeps(&mut self, v: u32) {
        self.substeps = v.max(1).min(16);
    }

    pub fn set_pass_mode(&mut self, mode: u32) {
        self.pass_mode = mode.clamp(1, 2);
    }

    pub fn set_approx_mode(&mut self, mode: u32) {
        self.approx_mode = mode.min(1);
    }

    pub fn set_phi_mean(&mut self, v: f32) {
        self.phi_mean = v.max(-1.0).min(1.0);
    }

    pub fn set_noise_amp(&mut self, v: f32) {
        self.noise_amp = v.max(0.0).min(1.0);
    }
}

#[wasm_bindgen]
pub struct CahnHilliardSimulation {
    nx: usize,
    ny: usize,
    nz: usize,
    nxy: usize,

    x_minus: Vec<usize>,
    x_plus: Vec<usize>,
    x_minus2: Vec<usize>,
    x_plus2: Vec<usize>,
    y_minus: Vec<usize>,
    y_plus: Vec<usize>,
    y_minus2: Vec<usize>,
    y_plus2: Vec<usize>,
    z_minus: Vec<usize>,
    z_plus: Vec<usize>,
    z_minus2: Vec<usize>,
    z_plus2: Vec<usize>,

    params: CahnHilliardParams,
    base_seed: u32,
    dt: f32,

    chunk_nx: usize,
    chunk_ny: usize,
    chunk_nz: usize,
    chunk_v_min: Vec<f32>,
    chunk_v_max: Vec<f32>,

    phi0: Vec<f32>,
    phi1: Vec<f32>,
    mu: Vec<f32>,
    mu_dirty: bool,

    v: Vec<f32>,
    v_dirty: bool,

    // 0: (phi+1)/2
    // 1: 1-exp(-gain*|mu|)
    // 2: 0.5 + 0.5*tanh(gain*phi)
    // 3: 1-exp(-gain*energy_density)
    export_mode: u32,

    // Meaning depends on export_mode.
    export_gain: f32,
}

#[wasm_bindgen]
impl CahnHilliardSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: usize, ny: usize, nz: usize, seed: u32, params: CahnHilliardParams) -> Self {
        let n = nx * ny * nz;
        let nxy = nx * ny;

        let x_minus: Vec<usize> = (0..nx).map(|x| wrap(x as isize - 1, nx)).collect();
        let x_plus: Vec<usize> = (0..nx).map(|x| wrap(x as isize + 1, nx)).collect();
        let x_minus2: Vec<usize> = (0..nx).map(|x| wrap(x as isize - 2, nx)).collect();
        let x_plus2: Vec<usize> = (0..nx).map(|x| wrap(x as isize + 2, nx)).collect();

        let y_minus: Vec<usize> = (0..ny).map(|y| wrap(y as isize - 1, ny)).collect();
        let y_plus: Vec<usize> = (0..ny).map(|y| wrap(y as isize + 1, ny)).collect();
        let y_minus2: Vec<usize> = (0..ny).map(|y| wrap(y as isize - 2, ny)).collect();
        let y_plus2: Vec<usize> = (0..ny).map(|y| wrap(y as isize + 2, ny)).collect();

        let z_minus: Vec<usize> = (0..nz).map(|z| wrap(z as isize - 1, nz)).collect();
        let z_plus: Vec<usize> = (0..nz).map(|z| wrap(z as isize + 1, nz)).collect();
        let z_minus2: Vec<usize> = (0..nz).map(|z| wrap(z as isize - 2, nz)).collect();
        let z_plus2: Vec<usize> = (0..nz).map(|z| wrap(z as isize + 2, nz)).collect();

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
            x_minus2,
            x_plus2,
            y_minus,
            y_plus,
            y_minus2,
            y_plus2,
            z_minus,
            z_plus,
            z_minus2,
            z_plus2,

            params,
            base_seed: seed,
            dt: 0.002,

            chunk_nx,
            chunk_ny,
            chunk_nz,
            chunk_v_min: vec![f32::INFINITY; chunk_total],
            chunk_v_max: vec![f32::NEG_INFINITY; chunk_total],

            phi0: vec![0.0; n],
            phi1: vec![0.0; n],
            mu: vec![0.0; n],
            mu_dirty: true,

            v: vec![0.0; n],
            v_dirty: true,

            export_mode: 0,
            export_gain: 6.0,
        };

        let mean = sim.params.phi_mean;
        let amp = sim.params.noise_amp;
        sim.seed_spinodal(mean, amp);

        sim
    }

    pub fn dt(&self) -> f32 {
        self.dt
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt.max(0.00001).min(1.0);
    }

    pub fn set_export_mode(&mut self, mode: u32) {
        self.export_mode = mode.min(3);
        self.v_dirty = true;
    }

    pub fn set_export_gain(&mut self, gain: f32) {
        self.export_gain = gain.max(0.00001);
        self.v_dirty = true;
    }

    pub fn step(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step_once();
        }
    }

    pub fn seed_spinodal(&mut self, phi_mean: f32, noise_amp: f32) {
        let n = self.phi0.len();
        if n == 0 {
            return;
        }

        let mean_target = phi_mean.max(-1.0).min(1.0);
        let amp = noise_amp.max(0.0);

        let mut rng = LcgRng::new(self.base_seed ^ 0xC411_DEAD);
        let mut sum = 0.0f64;

        for v in &mut self.phi0 {
            let u = rng.next_f32() * 2.0 - 1.0;
            let p = mean_target + amp * u;
            *v = p;
            sum += p as f64;
        }

        let mean_cur = (sum / (n as f64)) as f32;
        let corr = mean_target - mean_cur;
        for v in &mut self.phi0 {
            *v += corr;
        }

        self.phi1.copy_from_slice(&self.phi0);
        self.mu_dirty = true;
        self.v_dirty = true;
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

        self.chunk_v_min.fill(f32::INFINITY);
        self.chunk_v_max.fill(f32::NEG_INFINITY);

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

    pub(crate) fn v_slice(&self) -> &[f32] {
        &self.v
    }

    pub(crate) fn chunk_v_min_slice(&self) -> &[f32] {
        &self.chunk_v_min
    }

    pub(crate) fn chunk_v_max_slice(&self) -> &[f32] {
        &self.chunk_v_max
    }

    fn step_once(&mut self) {
        let sub = self.params.substeps.max(1) as usize;
        let dt = self.dt / (sub as f32);
        for _ in 0..sub {
            self.step_kernel(dt);
        }
    }

    fn update_v_field(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nxy = self.nxy;
        let phi = &self.phi0;

        match self.export_mode {
            1 => {
                if self.mu_dirty {
                    self.recompute_mu();
                }
                let mu = &self.mu;
                let gain = self.export_gain.max(0.00001);

                self.v
                    .par_chunks_mut(nxy)
                    .enumerate()
                    .for_each(|(z, v_out_z)| {
                        let z_off = z * nxy;
                        let mu_z = &mu[z_off..z_off + nxy];
                        for i in 0..nxy {
                            let x = mu_z[i].abs();
                            v_out_z[i] = 1.0 - (-gain * x).exp();
                        }
                    });
            }
            2 => {
                let gain = self.export_gain.max(0.00001).min(32.0);

                self.v
                    .par_chunks_mut(nxy)
                    .enumerate()
                    .for_each(|(z, v_out_z)| {
                        let z_off = z * nxy;
                        let phi_z = &phi[z_off..z_off + nxy];
                        for i in 0..nxy {
                            let t = (gain * phi_z[i]).tanh();
                            v_out_z[i] = 0.5 + 0.5 * t;
                        }
                    });
            }
            3 => {
                // Energy density proxy:
                //   e = (phi^2 - 1)^2 + kappa * |grad(phi)|^2
                // mapped to v = 1 - exp(-gain * e)

                let gain = self.export_gain.max(0.00001);
                let kappa = self.params.kappa;

                let x_minus = &self.x_minus;
                let x_plus = &self.x_plus;
                let y_minus = &self.y_minus;
                let y_plus = &self.y_plus;
                let z_minus = &self.z_minus;
                let z_plus = &self.z_plus;

                self.v
                    .par_chunks_mut(nxy)
                    .enumerate()
                    .for_each(|(z, v_out_z)| {
                        let z_off = z * nxy;
                        let zm_off = z_minus[z] * nxy;
                        let zp_off = z_plus[z] * nxy;

                        let phi_z = &phi[z_off..z_off + nxy];
                        let phi_zm = &phi[zm_off..zm_off + nxy];
                        let phi_zp = &phi[zp_off..zp_off + nxy];

                        for y in 0..ny {
                            let row_off = y * nx;
                            let ym_row_off = unsafe { *y_minus.get_unchecked(y) } * nx;
                            let yp_row_off = unsafe { *y_plus.get_unchecked(y) } * nx;

                            for x in 0..nx {
                                let p = row_off + x;
                                let xm = row_off + unsafe { *x_minus.get_unchecked(x) };
                                let xp = row_off + unsafe { *x_plus.get_unchecked(x) };

                                let phi_c = unsafe { *phi_z.get_unchecked(p) };

                                let phi2 = phi_c * phi_c;
                                let w = phi2 - 1.0;
                                let e_bulk = w * w;

                                let dx = (unsafe { *phi_z.get_unchecked(xp) }
                                    - unsafe { *phi_z.get_unchecked(xm) })
                                    * 0.5;
                                let dy = (unsafe { *phi_z.get_unchecked(yp_row_off + x) }
                                    - unsafe { *phi_z.get_unchecked(ym_row_off + x) })
                                    * 0.5;
                                let dz = (unsafe { *phi_zp.get_unchecked(p) }
                                    - unsafe { *phi_zm.get_unchecked(p) })
                                    * 0.5;

                                let e_grad = kappa * (dx * dx + dy * dy + dz * dz);
                                let e = e_bulk + e_grad;

                                v_out_z[p] = 1.0 - (-gain * e).exp();
                            }
                        }
                    });
            }
            _ => {
                self.v
                    .par_chunks_mut(nxy)
                    .enumerate()
                    .for_each(|(z, v_out_z)| {
                        let z_off = z * nxy;
                        let phi_z = &phi[z_off..z_off + nxy];
                        for i in 0..nxy {
                            v_out_z[i] = (phi_z[i] + 1.0) * 0.5;
                        }
                    });
            }
        }

        self.v_dirty = false;
    }

    fn recompute_mu(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nxy = self.nxy;

        let a = self.params.a;
        let kappa = self.params.kappa;

        let x_minus = &self.x_minus;
        let x_plus = &self.x_plus;
        let y_minus = &self.y_minus;
        let y_plus = &self.y_plus;
        let z_minus = &self.z_minus;
        let z_plus = &self.z_plus;

        let phi0 = &self.phi0;
        let mu = &mut self.mu;

        mu.par_chunks_mut(nxy)
            .enumerate()
            .for_each(|(z, mu_out_z)| {
                let z_off = z * nxy;
                let zm_off = z_minus[z] * nxy;
                let zp_off = z_plus[z] * nxy;

                let phi_z = &phi0[z_off..z_off + nxy];
                let phi_zm = &phi0[zm_off..zm_off + nxy];
                let phi_zp = &phi0[zp_off..zp_off + nxy];

                for y in 0..ny {
                    let row_off = y * nx;
                    let ym_row_off = unsafe { *y_minus.get_unchecked(y) } * nx;
                    let yp_row_off = unsafe { *y_plus.get_unchecked(y) } * nx;

                    for x in 0..nx {
                        let p = row_off + x;
                        let phi_c = unsafe { *phi_z.get_unchecked(p) };

                        let xm = row_off + unsafe { *x_minus.get_unchecked(x) };
                        let xp = row_off + unsafe { *x_plus.get_unchecked(x) };

                        let lap = unsafe { *phi_z.get_unchecked(xm) }
                            + unsafe { *phi_z.get_unchecked(xp) }
                            + unsafe { *phi_z.get_unchecked(ym_row_off + x) }
                            + unsafe { *phi_z.get_unchecked(yp_row_off + x) }
                            + unsafe { *phi_zm.get_unchecked(p) }
                            + unsafe { *phi_zp.get_unchecked(p) }
                            - 6.0 * phi_c;

                        let phi2 = phi_c * phi_c;
                        let dw = phi_c * (phi2 - 1.0);
                        unsafe {
                            *mu_out_z.get_unchecked_mut(p) = a * dw - kappa * lap;
                        }
                    }
                }
            });

        self.mu_dirty = false;
    }

    fn step_kernel(&mut self, dt: f32) {
        let nx = self.nx;
        let ny = self.ny;
        let nxy = self.nxy;

        let a = self.params.a;
        let kappa = self.params.kappa;
        let m = self.params.m;

        let x_minus = &self.x_minus;
        let x_plus = &self.x_plus;
        let y_minus = &self.y_minus;
        let y_plus = &self.y_plus;
        let z_minus = &self.z_minus;
        let z_plus = &self.z_plus;

        if self.params.pass_mode == 1 {
            let x_minus2 = &self.x_minus2;
            let x_plus2 = &self.x_plus2;
            let y_minus2 = &self.y_minus2;
            let y_plus2 = &self.y_plus2;
            let z_minus2 = &self.z_minus2;
            let z_plus2 = &self.z_plus2;

            // Pass 1: mu from phi.
            {
                let phi0 = &self.phi0;
                let mu = &mut self.mu;

                mu.par_chunks_mut(nxy)
                    .enumerate()
                    .for_each(|(z, mu_out_z)| {
                        let z_off = z * nxy;
                        let zm_off = z_minus[z] * nxy;
                        let zp_off = z_plus[z] * nxy;

                        let phi_z = &phi0[z_off..z_off + nxy];
                        let phi_zm = &phi0[zm_off..zm_off + nxy];
                        let phi_zp = &phi0[zp_off..zp_off + nxy];

                        for y in 0..ny {
                            let row_off = y * nx;
                            let ym_row_off = unsafe { *y_minus.get_unchecked(y) } * nx;
                            let yp_row_off = unsafe { *y_plus.get_unchecked(y) } * nx;

                            for x in 0..nx {
                                let p = row_off + x;
                                let phi_c = unsafe { *phi_z.get_unchecked(p) };

                                let xm = row_off + unsafe { *x_minus.get_unchecked(x) };
                                let xp = row_off + unsafe { *x_plus.get_unchecked(x) };

                                let lap = unsafe { *phi_z.get_unchecked(xm) }
                                    + unsafe { *phi_z.get_unchecked(xp) }
                                    + unsafe { *phi_z.get_unchecked(ym_row_off + x) }
                                    + unsafe { *phi_z.get_unchecked(yp_row_off + x) }
                                    + unsafe { *phi_zm.get_unchecked(p) }
                                    + unsafe { *phi_zp.get_unchecked(p) }
                                    - 6.0 * phi_c;

                                let phi2 = phi_c * phi_c;
                                let dw = phi_c * (phi2 - 1.0);
                                unsafe {
                                    *mu_out_z.get_unchecked_mut(p) = a * dw - kappa * lap;
                                }
                            }
                        }
                    });
            }

            // Pass 2: approximate Lap(mu) using radius-2 stencil.
            {
                let mu = &self.mu;
                let phi0 = &self.phi0;
                let phi1 = &mut self.phi1;

                phi1.par_chunks_mut(nxy)
                    .enumerate()
                    .for_each(|(z, phi_out_z)| {
                        let z_off = z * nxy;
                        let zm2_off = z_minus2[z] * nxy;
                        let zp2_off = z_plus2[z] * nxy;

                        let phi_z = &phi0[z_off..z_off + nxy];
                        let mu_z = &mu[z_off..z_off + nxy];
                        let mu_zm2 = &mu[zm2_off..zm2_off + nxy];
                        let mu_zp2 = &mu[zp2_off..zp2_off + nxy];

                        for y in 0..ny {
                            let row_off = y * nx;
                            let ym2_row_off = unsafe { *y_minus2.get_unchecked(y) } * nx;
                            let yp2_row_off = unsafe { *y_plus2.get_unchecked(y) } * nx;

                            for x in 0..nx {
                                let p = row_off + x;
                                let xm2 = row_off + unsafe { *x_minus2.get_unchecked(x) };
                                let xp2 = row_off + unsafe { *x_plus2.get_unchecked(x) };

                                let mu_c = unsafe { *mu_z.get_unchecked(p) };
                                let lap_mu2 = unsafe { *mu_z.get_unchecked(xm2) }
                                    + unsafe { *mu_z.get_unchecked(xp2) }
                                    + unsafe { *mu_z.get_unchecked(ym2_row_off + x) }
                                    + unsafe { *mu_z.get_unchecked(yp2_row_off + x) }
                                    + unsafe { *mu_zm2.get_unchecked(p) }
                                    + unsafe { *mu_zp2.get_unchecked(p) }
                                    - 6.0 * mu_c;

                                let phi_c = unsafe { *phi_z.get_unchecked(p) };
                                unsafe {
                                    *phi_out_z.get_unchecked_mut(p) = phi_c + dt * m * lap_mu2;
                                }
                            }
                        }
                    });
            }

            std::mem::swap(&mut self.phi0, &mut self.phi1);
            self.mu_dirty = false;
            self.v_dirty = true;
            return;
        }

        // Two-pass CH update.

        // Pass 1: mu from phi.
        {
            let phi0 = &self.phi0;
            let mu = &mut self.mu;

            mu.par_chunks_mut(nxy)
                .enumerate()
                .for_each(|(z, mu_out_z)| {
                    let z_off = z * nxy;
                    let zm_off = z_minus[z] * nxy;
                    let zp_off = z_plus[z] * nxy;

                    let phi_z = &phi0[z_off..z_off + nxy];
                    let phi_zm = &phi0[zm_off..zm_off + nxy];
                    let phi_zp = &phi0[zp_off..zp_off + nxy];

                    for y in 0..ny {
                        let row_off = y * nx;
                        let ym_row_off = unsafe { *y_minus.get_unchecked(y) } * nx;
                        let yp_row_off = unsafe { *y_plus.get_unchecked(y) } * nx;

                        for x in 0..nx {
                            let p = row_off + x;
                            let phi_c = unsafe { *phi_z.get_unchecked(p) };

                            let xm = row_off + unsafe { *x_minus.get_unchecked(x) };
                            let xp = row_off + unsafe { *x_plus.get_unchecked(x) };

                            let lap = unsafe { *phi_z.get_unchecked(xm) }
                                + unsafe { *phi_z.get_unchecked(xp) }
                                + unsafe { *phi_z.get_unchecked(ym_row_off + x) }
                                + unsafe { *phi_z.get_unchecked(yp_row_off + x) }
                                + unsafe { *phi_zm.get_unchecked(p) }
                                + unsafe { *phi_zp.get_unchecked(p) }
                                - 6.0 * phi_c;

                            let phi2 = phi_c * phi_c;
                            let dw = phi_c * (phi2 - 1.0);
                            unsafe {
                                *mu_out_z.get_unchecked_mut(p) = a * dw - kappa * lap;
                            }
                        }
                    }
                });
        }

        // Pass 2: phi update.
        {
            let approx = self.params.approx_mode == 1;

            let x_minus2 = &self.x_minus2;
            let x_plus2 = &self.x_plus2;
            let y_minus2 = &self.y_minus2;
            let y_plus2 = &self.y_plus2;
            let z_minus2 = &self.z_minus2;
            let z_plus2 = &self.z_plus2;

            let mu = &self.mu;
            let phi0 = &self.phi0;
            let phi1 = &mut self.phi1;

            phi1.par_chunks_mut(nxy)
                .enumerate()
                .for_each(|(z, phi_out_z)| {
                    let z_off = z * nxy;

                    let phi_z = &phi0[z_off..z_off + nxy];
                    let mu_z = &mu[z_off..z_off + nxy];

                    if approx {
                        let zm2_off = z_minus2[z] * nxy;
                        let zp2_off = z_plus2[z] * nxy;
                        let mu_zm2 = &mu[zm2_off..zm2_off + nxy];
                        let mu_zp2 = &mu[zp2_off..zp2_off + nxy];

                        for y in 0..ny {
                            let row_off = y * nx;
                            let ym2_row_off = unsafe { *y_minus2.get_unchecked(y) } * nx;
                            let yp2_row_off = unsafe { *y_plus2.get_unchecked(y) } * nx;

                            for x in 0..nx {
                                let p = row_off + x;
                                let xm2 = row_off + unsafe { *x_minus2.get_unchecked(x) };
                                let xp2 = row_off + unsafe { *x_plus2.get_unchecked(x) };

                                let mu_c = unsafe { *mu_z.get_unchecked(p) };
                                let lap_mu2 = unsafe { *mu_z.get_unchecked(xm2) }
                                    + unsafe { *mu_z.get_unchecked(xp2) }
                                    + unsafe { *mu_z.get_unchecked(ym2_row_off + x) }
                                    + unsafe { *mu_z.get_unchecked(yp2_row_off + x) }
                                    + unsafe { *mu_zm2.get_unchecked(p) }
                                    + unsafe { *mu_zp2.get_unchecked(p) }
                                    - 6.0 * mu_c;

                                let phi_c = unsafe { *phi_z.get_unchecked(p) };
                                unsafe {
                                    *phi_out_z.get_unchecked_mut(p) = phi_c + dt * m * lap_mu2;
                                }
                            }
                        }
                    } else {
                        let zm_off = z_minus[z] * nxy;
                        let zp_off = z_plus[z] * nxy;
                        let mu_zm = &mu[zm_off..zm_off + nxy];
                        let mu_zp = &mu[zp_off..zp_off + nxy];

                        for y in 0..ny {
                            let row_off = y * nx;
                            let ym_row_off = unsafe { *y_minus.get_unchecked(y) } * nx;
                            let yp_row_off = unsafe { *y_plus.get_unchecked(y) } * nx;

                            for x in 0..nx {
                                let p = row_off + x;
                                let xm = row_off + unsafe { *x_minus.get_unchecked(x) };
                                let xp = row_off + unsafe { *x_plus.get_unchecked(x) };

                                let mu_c = unsafe { *mu_z.get_unchecked(p) };
                                let lap_mu = unsafe { *mu_z.get_unchecked(xm) }
                                    + unsafe { *mu_z.get_unchecked(xp) }
                                    + unsafe { *mu_z.get_unchecked(ym_row_off + x) }
                                    + unsafe { *mu_z.get_unchecked(yp_row_off + x) }
                                    + unsafe { *mu_zm.get_unchecked(p) }
                                    + unsafe { *mu_zp.get_unchecked(p) }
                                    - 6.0 * mu_c;

                                let phi_c = unsafe { *phi_z.get_unchecked(p) };
                                unsafe {
                                    *phi_out_z.get_unchecked_mut(p) = phi_c + dt * m * lap_mu;
                                }
                            }
                        }
                    }
                });
        }

        std::mem::swap(&mut self.phi0, &mut self.phi1);
        self.mu_dirty = false;
        self.v_dirty = true;
    }
}
