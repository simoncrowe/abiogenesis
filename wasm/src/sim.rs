use wasm_bindgen::prelude::*;

use crate::meshing::{mesh_region_append, mesh_region_append_lerp, MeshBuffers};
use rayon::prelude::*;

const CHUNK: usize = 16;

fn idx(nx: usize, ny: usize, x: usize, y: usize, z: usize) -> usize {
    x + nx * (y + ny * z)
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

    pub fn push_keyframe_from_gray_scott(&mut self, sim: &crate::gray_scott::Simulation) {
        self.push_keyframe_with_chunk_ranges(
            sim.v_slice(),
            sim.chunk_v_min_slice(),
            sim.chunk_v_max_slice(),
        );
    }

    pub fn push_keyframe_from_stochastic_rdme(
        &mut self,
        sim: &crate::rdme::StochasticRdmeSimulation,
    ) {
        self.push_keyframe_with_chunk_ranges(
            sim.v_slice(),
            sim.chunk_v_min_slice(),
            sim.chunk_v_max_slice(),
        );
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

        #[derive(Clone, Copy)]
        struct Job {
            x0: isize,
            x1: isize,
            y0: isize,
            y1: isize,
            z0: isize,
            z1: isize,
            stride: usize,
        }

        let mut jobs = Vec::new();

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

                    jobs.push(Job {
                        x0,
                        x1,
                        y0,
                        y1,
                        z0,
                        z1,
                        stride,
                    });
                }
            }
        }

        const JOB_BATCH: usize = 8;

        let mesh = jobs
            .par_chunks(JOB_BATCH)
            .map(|chunk| {
                let mut acc = MeshBuffers::new();
                for job in chunk {
                    mesh_region_append(
                        &self.scalars_next,
                        self.nx,
                        self.ny,
                        self.nz,
                        iso,
                        color,
                        &mut acc,
                        job.x0,
                        job.x1,
                        job.y0,
                        job.y1,
                        job.z0,
                        job.z1,
                        job.stride,
                    );
                }
                acc
            })
            .reduce(MeshBuffers::new, |mut a, b| {
                a.append(&b);
                a
            });

        self.mesh = mesh;
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

        #[derive(Clone, Copy)]
        struct Job {
            x0: isize,
            x1: isize,
            y0: isize,
            y1: isize,
            z0: isize,
            z1: isize,
            stride: usize,
        }

        let mut jobs = Vec::new();

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

                    jobs.push(Job {
                        x0,
                        x1,
                        y0,
                        y1,
                        z0,
                        z1,
                        stride,
                    });
                }
            }
        }

        const JOB_BATCH: usize = 8;

        let mesh = jobs
            .par_chunks(JOB_BATCH)
            .map(|chunk| {
                let mut acc = MeshBuffers::new();
                for job in chunk {
                    mesh_region_append_lerp(
                        &self.scalars_prev,
                        &self.scalars_next,
                        lerp_t,
                        self.nx,
                        self.ny,
                        self.nz,
                        iso,
                        color,
                        &mut acc,
                        job.x0,
                        job.x1,
                        job.y0,
                        job.y1,
                        job.z0,
                        job.z1,
                        job.stride,
                    );
                }
                acc
            })
            .reduce(MeshBuffers::new, |mut a, b| {
                a.append(&b);
                a
            });

        self.mesh = mesh;
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
