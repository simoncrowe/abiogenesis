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

#[wasm_bindgen]
#[derive(Clone)]
pub struct ExcitableMediaParams {
    // Barkley kinetics.
    epsilon: f32,
    a: f32,
    b: f32,

    // Diffusion.
    du: f32,
    dv: f32,

    // Optional stability knob.
    substeps: u32,
}

#[wasm_bindgen]
impl ExcitableMediaParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            epsilon: 0.03,
            a: 0.75,
            b: 0.02,
            du: 1.0,
            dv: 0.0,
            substeps: 1,
        }
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
    pub fn a(&self) -> f32 {
        self.a
    }
    pub fn b(&self) -> f32 {
        self.b
    }
    pub fn du(&self) -> f32 {
        self.du
    }
    pub fn dv(&self) -> f32 {
        self.dv
    }
    pub fn substeps(&self) -> u32 {
        self.substeps
    }

    pub fn set_epsilon(&mut self, v: f32) {
        self.epsilon = v.max(0.00001).min(1.0);
    }

    pub fn set_a(&mut self, v: f32) {
        self.a = v.max(0.00001).min(4.0);
    }

    pub fn set_b(&mut self, v: f32) {
        self.b = v.max(0.0).min(1.0);
    }

    pub fn set_du(&mut self, v: f32) {
        self.du = v.max(0.0).min(10.0);
    }

    pub fn set_dv(&mut self, v: f32) {
        self.dv = v.max(0.0).min(10.0);
    }

    pub fn set_substeps(&mut self, v: u32) {
        self.substeps = v.max(1).min(16);
    }
}

#[wasm_bindgen]
pub struct ExcitableMediaSimulation {
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

    params: ExcitableMediaParams,
    dt: f32,

    chunk_nx: usize,
    chunk_ny: usize,
    chunk_nz: usize,
    chunk_v_min: Vec<f32>,
    chunk_v_max: Vec<f32>,

    // Barkley fields:
    // u: activator
    // v: recovery / inhibitor
    u0: Vec<f32>,
    v0: Vec<f32>,
    u1: Vec<f32>,
    v1: Vec<f32>,

    rng: LcgRng,
}

#[wasm_bindgen]
impl ExcitableMediaSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: usize, ny: usize, nz: usize, seed: u32, params: ExcitableMediaParams) -> Self {
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
            dt: 0.01,

            chunk_nx,
            chunk_ny,
            chunk_nz,
            chunk_v_min: vec![f32::INFINITY; chunk_total],
            chunk_v_max: vec![f32::NEG_INFINITY; chunk_total],

            u0: vec![0.0; n],
            v0: vec![0.0; n],
            u1: vec![0.0; n],
            v1: vec![0.0; n],

            rng: LcgRng::new(seed),
        };

        // Default: gentle random excitation.
        sim.seed_random(0.02, 0.002);
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
        // Explicit Euler + diffusion. Keep it sane.
        self.dt = dt.max(0.00001).min(0.2);
    }

    pub fn seed_random(&mut self, noise_amp: f32, excited_prob: f32) {
        let noise_amp = noise_amp.max(0.0).min(1.0);
        let excited_prob = excited_prob.max(0.0).min(1.0);

        for i in 0..self.u0.len() {
            let mut u = 0.0;
            if self.rng.next_f32() < excited_prob {
                u = 1.0;
            }
            u += (self.rng.next_f32() - 0.5) * 2.0 * noise_amp;
            self.u0[i] = u.max(-2.0).min(2.0);
            self.v0[i] = 0.0;
        }

        self.u1.copy_from_slice(&self.u0);
        self.v1.copy_from_slice(&self.v0);
    }

    pub fn seed_sources(&mut self, source_count: u32, radius01: f32, u_peak: f32) {
        let count = source_count.min(4096) as usize;
        let r01 = radius01.max(0.0).min(0.5);
        let u_peak = u_peak.max(0.0).min(2.0);

        self.u0.fill(0.0);
        self.v0.fill(0.0);

        if self.u0.is_empty() || count == 0 {
            return;
        }

        let rx = (self.nx as f32 * r01).max(1.0);
        let ry = (self.ny as f32 * r01).max(1.0);
        let rz = (self.nz as f32 * r01).max(1.0);

        for _ in 0..count {
            let cx = self.rng.next_f32() * ((self.nx.max(1) - 1) as f32);
            let cy = self.rng.next_f32() * ((self.ny.max(1) - 1) as f32);
            let cz = self.rng.next_f32() * ((self.nz.max(1) - 1) as f32);

            for z in 0..self.nz {
                let dz = (z as f32 - cz) / rz;
                let dz2 = dz * dz;
                for y in 0..self.ny {
                    let dy = (y as f32 - cy) / ry;
                    let dy2 = dy * dy;
                    for x in 0..self.nx {
                        let dx = (x as f32 - cx) / rx;
                        let d2 = dx * dx + dy2 + dz2;
                        if d2 <= 1.0 {
                            let i = idx(self.nx, self.ny, x, y, z);
                            // Soft profile to reduce numerical ringing.
                            let w = (1.0 - d2).max(0.0);
                            self.u0[i] = self.u0[i].max(u_peak * w);
                        }
                    }
                }
            }
        }

        self.u1.copy_from_slice(&self.u0);
        self.v1.copy_from_slice(&self.v0);
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

        let nx = self.nx;
        let nxy = self.nxy;
        let u = &self.u0;

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
                            let uu = u[row + x];
                            if uu < minv {
                                minv = uu;
                            }
                            if uu > maxv {
                                maxv = uu;
                            }
                        }
                    }
                }

                *min_out = minv;
                *max_out = maxv;
            });
    }

    pub fn v_ptr(&self) -> u32 {
        self.u0.as_ptr() as u32
    }

    pub fn v_len(&self) -> usize {
        self.u0.len()
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

impl ExcitableMediaSimulation {
    pub(crate) fn v_slice(&self) -> &[f32] {
        &self.u0
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

    fn step_kernel(&mut self, dt: f32) {
        let nx = self.nx;
        let ny = self.ny;
        let nxy = self.nxy;

        let eps_inv = 1.0 / self.params.epsilon.max(0.00001);
        let a_inv = 1.0 / self.params.a.max(0.00001);
        let b = self.params.b;
        let du = self.params.du;
        let dv = self.params.dv;

        // Avoid Laplacian neighbor reads when diffusion is disabled.
        // This is a big win because dv==0 is a very common Barkley setting.

        let x_minus = &self.x_minus;
        let x_plus = &self.x_plus;
        let y_minus = &self.y_minus;
        let y_plus = &self.y_plus;
        let z_minus = &self.z_minus;
        let z_plus = &self.z_plus;

        let u0 = &self.u0;
        let v0 = &self.v0;
        let u1 = &mut self.u1;
        let v1 = &mut self.v1;

        u1.par_chunks_mut(nxy)
            .zip(v1.par_chunks_mut(nxy))
            .enumerate()
            .for_each(|(z, (u1z, v1z))| {
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

                        let u = u0[c];
                        let v = v0[c];

                        let u_lap = if du != 0.0 {
                            u0[y_off + x_minus[x]]
                                + u0[y_off + x_plus[x]]
                                + u0[y_off_m + x]
                                + u0[y_off_p + x]
                                + u0[y_off_zm + x]
                                + u0[y_off_zp + x]
                                - 6.0 * u
                        } else {
                            0.0
                        };

                        let v_lap = if dv != 0.0 {
                            v0[y_off + x_minus[x]]
                                + v0[y_off + x_plus[x]]
                                + v0[y_off_m + x]
                                + v0[y_off_p + x]
                                + v0[y_off_zm + x]
                                + v0[y_off_zp + x]
                                - 6.0 * v
                        } else {
                            0.0
                        };

                        // Barkley reaction terms.
                        let ru = eps_inv * u * (1.0 - u) * (u - (v + b) * a_inv);
                        let rv = u - v;

                        let mut un = u + dt * (ru + du * u_lap);
                        let mut vn = v + dt * (rv + dv * v_lap);

                        // Keep the field in a numerically sane range for meshing.
                        // Barkley is typically in ~[0,1], but explicit Euler + tuning can overshoot.
                        if !un.is_finite() {
                            un = 0.0;
                        }
                        if !vn.is_finite() {
                            vn = 0.0;
                        }
                        un = un.max(-2.0).min(2.0);
                        vn = vn.max(-2.0).min(2.0);

                        u1z[row + x] = un;
                        v1z[row + x] = vn;
                    }
                }
            });

        std::mem::swap(&mut self.u0, &mut self.u1);
        std::mem::swap(&mut self.v0, &mut self.v1);
    }
}
