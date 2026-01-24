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
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }

    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        let mant = (self.next_u32() >> 8) as u32;
        (mant as f32) / ((1u32 << 24) as f32)
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct LeniaParams {
    radius: u32,
    mu: f32,
    sigma: f32,
    dt: f32,

    // Optional: bias the kernel shape. v1 uses a simple gaussian in radius.
    kernel_sharpness: f32,
}

#[wasm_bindgen]
impl LeniaParams {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            radius: 5,
            mu: 0.15,
            sigma: 0.03,
            dt: 0.10,
            kernel_sharpness: 1.0,
        }
    }

    pub fn radius(&self) -> u32 {
        self.radius
    }
    pub fn mu(&self) -> f32 {
        self.mu
    }
    pub fn sigma(&self) -> f32 {
        self.sigma
    }
    pub fn dt(&self) -> f32 {
        self.dt
    }
    pub fn kernel_sharpness(&self) -> f32 {
        self.kernel_sharpness
    }

    pub fn set_radius(&mut self, v: u32) {
        self.radius = v.max(1).min(12);
    }
    pub fn set_mu(&mut self, v: f32) {
        self.mu = v.max(0.0).min(1.0);
    }
    pub fn set_sigma(&mut self, v: f32) {
        self.sigma = v.max(0.00001).min(1.0);
    }
    pub fn set_dt(&mut self, v: f32) {
        self.dt = v.max(0.00001).min(1.0);
    }
    pub fn set_kernel_sharpness(&mut self, v: f32) {
        self.kernel_sharpness = v.max(0.1).min(8.0);
    }
}

#[derive(Clone, Copy)]
struct KernelTap {
    // Flattened neighbor delta index into the lattice (periodic wrap applied at runtime).
    // This is only valid for interior arithmetic; we still need per-axis wrapping.
    dx: i16,
    dy: i16,
    dz: i16,
    w: f32,
}

fn build_kernel(radius: usize, sharpness: f32) -> Vec<KernelTap> {
    let r = radius as i32;
    let r2 = (radius as f32) * (radius as f32);

    let mut taps: Vec<KernelTap> = Vec::new();

    // Simple radial gaussian-ish bump with configurable sharpness.
    // w = exp(-sharpness * (d^2 / r^2))
    let mut sum = 0.0f64;
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let d2 = (dx * dx + dy * dy + dz * dz) as f32;
                if d2 > r2 {
                    continue;
                }
                let t = d2 / r2.max(1e-6);
                let w = (-sharpness * t).exp();
                taps.push(KernelTap {
                    dx: dx as i16,
                    dy: dy as i16,
                    dz: dz as i16,
                    w,
                });
                sum += w as f64;
            }
        }
    }

    // Normalize.
    let inv = if sum > 0.0 { (1.0 / sum) as f32 } else { 1.0 };
    for t in &mut taps {
        t.w *= inv;
    }

    taps
}

#[wasm_bindgen]
pub struct LeniaSimulation {
    nx: usize,
    ny: usize,
    nz: usize,
    nxy: usize,

    // For small |d| <= radius, precompute the wrapped coordinate mapping per axis:
    // xwrap[(d+R)*nx + x] = wrap(x + d)
    radius: usize,
    xwrap: Vec<u16>,
    ywrap: Vec<u16>,
    zwrap: Vec<u16>,

    params: LeniaParams,

    dt: f32,

    chunk_nx: usize,
    chunk_ny: usize,
    chunk_nz: usize,
    chunk_v_min: Vec<f32>,
    chunk_v_max: Vec<f32>,

    kernel: Vec<KernelTap>,

    a0: Vec<f32>,
    a1: Vec<f32>,

    v: Vec<f32>,
    v_dirty: bool,

    rng: LcgRng,
}

#[wasm_bindgen]
impl LeniaSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: usize, ny: usize, nz: usize, seed: u32, params: LeniaParams) -> Self {
        let n = nx * ny * nz;
        let nxy = nx * ny;

        let cubes_x = nx.saturating_sub(1);
        let cubes_y = ny.saturating_sub(1);
        let cubes_z = nz.saturating_sub(1);

        let chunk_nx = (cubes_x + CHUNK - 1) / CHUNK;
        let chunk_ny = (cubes_y + CHUNK - 1) / CHUNK;
        let chunk_nz = (cubes_z + CHUNK - 1) / CHUNK;
        let chunk_total = chunk_nx * chunk_ny * chunk_nz;

        let r = params.radius as usize;
        let kernel = build_kernel(r, params.kernel_sharpness);

        // Precompute wrapped coordinate maps for dx/dy/dz in [-R, R].
        // Using u16 is safe for dims<=256 and cuts memory bandwidth.
        let rspan = 2 * r + 1;
        let mut xwrap: Vec<u16> = vec![0; rspan * nx];
        let mut ywrap: Vec<u16> = vec![0; rspan * ny];
        let mut zwrap: Vec<u16> = vec![0; rspan * nz];

        for di in 0..rspan {
            let d = di as isize - (r as isize);
            for x in 0..nx {
                xwrap[di * nx + x] = wrap(x as isize + d, nx) as u16;
            }
            for y in 0..ny {
                ywrap[di * ny + y] = wrap(y as isize + d, ny) as u16;
            }
            for z in 0..nz {
                zwrap[di * nz + z] = wrap(z as isize + d, nz) as u16;
            }
        }

        let mut sim = Self {
            nx,
            ny,
            nz,
            nxy,

            radius: r,
            xwrap,
            ywrap,
            zwrap,

            dt: params.dt,
            params,

            chunk_nx,
            chunk_ny,
            chunk_nz,
            chunk_v_min: vec![f32::INFINITY; chunk_total],
            chunk_v_max: vec![f32::NEG_INFINITY; chunk_total],

            kernel,

            a0: vec![0.0; n],
            a1: vec![0.0; n],

            v: vec![0.0; n],
            v_dirty: true,

            rng: LcgRng::new(seed),
        };

        sim.seed_noise(0.02);
        sim
    }

    pub fn dt(&self) -> f32 {
        self.dt
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt.max(0.00001).min(1.0);
    }

    pub fn set_radius(&mut self, radius: u32) {
        let r = radius.max(1).min(12) as usize;
        self.params.radius = r as u32;
        self.radius = r;
        self.kernel = build_kernel(r, self.params.kernel_sharpness);

        let rspan = 2 * r + 1;
        self.xwrap.resize(rspan * self.nx, 0);
        self.ywrap.resize(rspan * self.ny, 0);
        self.zwrap.resize(rspan * self.nz, 0);
        for di in 0..rspan {
            let d = di as isize - (r as isize);
            for x in 0..self.nx {
                self.xwrap[di * self.nx + x] = wrap(x as isize + d, self.nx) as u16;
            }
            for y in 0..self.ny {
                self.ywrap[di * self.ny + y] = wrap(y as isize + d, self.ny) as u16;
            }
            for z in 0..self.nz {
                self.zwrap[di * self.nz + z] = wrap(z as isize + d, self.nz) as u16;
            }
        }

        self.v_dirty = true;
    }

    pub fn set_kernel_sharpness(&mut self, sharpness: f32) {
        self.params.kernel_sharpness = sharpness.max(0.1).min(8.0);
        self.kernel = build_kernel(self.params.radius as usize, self.params.kernel_sharpness);
        self.v_dirty = true;
    }

    pub fn set_mu(&mut self, mu: f32) {
        self.params.mu = mu.max(0.0).min(1.0);
    }

    pub fn set_sigma(&mut self, sigma: f32) {
        self.params.sigma = sigma.max(0.00001).min(1.0);
    }

    pub fn seed_noise(&mut self, amp: f32) {
        let amp = amp.max(0.0).min(1.0);
        for v in &mut self.a0 {
            *v = (self.rng.next_f32() * amp).max(0.0).min(1.0);
        }
        self.a1.copy_from_slice(&self.a0);
        self.v_dirty = true;
    }

    pub fn seed_blobs(&mut self, blob_count: u32, radius01: f32, peak: f32) {
        let count = blob_count.min(4096) as usize;
        let r01 = radius01.max(0.0).min(0.5);
        let peak = peak.max(0.0).min(1.0);

        self.a0.fill(0.0);

        if self.a0.is_empty() || count == 0 {
            return;
        }

        let rx = (self.nx as f32 * r01).max(1.0);
        let ry = (self.ny as f32 * r01).max(1.0);
        let rz = (self.nz as f32 * r01).max(1.0);

        for _ in 0..count {
            let cx = self.rng.next_f32() * ((self.nx.max(1) - 1) as f32);
            let cy = self.rng.next_f32() * ((self.ny.max(1) - 1) as f32);
            let cz = self.rng.next_f32() * ((self.nz.max(1) - 1) as f32);

            // Only touch a bounding box around the blob center (big win vs scanning whole volume).
            let x0 = ((cx - rx).floor() as isize).max(0) as usize;
            let x1 = ((cx + rx).ceil() as isize).min((self.nx - 1) as isize) as usize;
            let y0 = ((cy - ry).floor() as isize).max(0) as usize;
            let y1 = ((cy + ry).ceil() as isize).min((self.ny - 1) as isize) as usize;
            let z0 = ((cz - rz).floor() as isize).max(0) as usize;
            let z1 = ((cz + rz).ceil() as isize).min((self.nz - 1) as isize) as usize;

            for z in z0..=z1 {
                let dz = (z as f32 - cz) / rz;
                let dz2 = dz * dz;
                for y in y0..=y1 {
                    let dy = (y as f32 - cy) / ry;
                    let dy2 = dy * dy;
                    for x in x0..=x1 {
                        let dx = (x as f32 - cx) / rx;
                        let d2 = dx * dx + dy2 + dz2;
                        if d2 <= 1.0 {
                            let i = idx(self.nx, self.ny, x, y, z);
                            let w = (1.0 - d2).max(0.0);
                            self.a0[i] = self.a0[i].max(peak * w);
                        }
                    }
                }
            }
        }

        self.a1.copy_from_slice(&self.a0);
        self.v_dirty = true;
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

impl LeniaSimulation {
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
        self.step_kernel(self.dt);
    }

    fn update_v_field(&mut self) {
        // v = A (identity).
        self.v.copy_from_slice(&self.a0);
        self.v_dirty = false;
    }

    fn step_kernel(&mut self, dt: f32) {
        let nx = self.nx;
        let ny = self.ny;
        let nxy = self.nxy;

        let radius = self.radius;
        let nz = self.nz;
        let xwrap = &self.xwrap;
        let ywrap = &self.ywrap;
        let zwrap = &self.zwrap;

        let kernel = &self.kernel;

        let mu = self.params.mu;
        let sigma = self.params.sigma.max(0.00001);
        let inv_2sig2 = 1.0 / (2.0 * sigma * sigma);

        let a0 = &self.a0;
        let a1 = &mut self.a1;

        a1.par_chunks_mut(nxy).enumerate().for_each(|(z, a1z)| {
            let z_off = z * nxy;

            for y in 0..ny {
                let y_off = z_off + y * nx;
                let row = y * nx;
                for x in 0..nx {
                    let c = y_off + x;
                    let a = a0[c];

                    // Convolution: use precomputed wrap tables for dx/dy/dz.
                    let mut u = 0.0f32;
                    for t in kernel {
                        let di_x = (t.dx as isize + radius as isize) as usize;
                        let di_y = (t.dy as isize + radius as isize) as usize;
                        let di_z = (t.dz as isize + radius as isize) as usize;

                        // taps are generated within [-R,R], so indices are valid.
                        let xx = xwrap[di_x * nx + x] as usize;
                        let yy = ywrap[di_y * ny + y] as usize;
                        let zz = zwrap[di_z * nz + z] as usize;

                        u += t.w * a0[idx(nx, ny, xx, yy, zz)];
                    }

                    // Growth function.
                    let du = u - mu;
                    let g = 2.0 * (-(du * du) * inv_2sig2).exp() - 1.0;

                    let mut an = a + dt * g;
                    if !an.is_finite() {
                        an = 0.0;
                    }
                    an = an.max(0.0).min(1.0);

                    a1z[row + x] = an;
                }
            }
        });

        std::mem::swap(&mut self.a0, &mut self.a1);
        self.v_dirty = true;
    }
}
