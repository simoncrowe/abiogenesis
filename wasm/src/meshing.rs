use crate::math::lerp;

pub struct MeshBuffers {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub colors: Vec<f32>,
    pub indices: Vec<u32>,
}

impl MeshBuffers {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.positions.clear();
        self.normals.clear();
        self.colors.clear();
        self.indices.clear();
    }

    fn push_vertex(&mut self, position: [f32; 3], normal: [f32; 3], color: [f32; 4]) -> u32 {
        let vertex_index = (self.positions.len() / 3) as u32;
        self.positions.extend_from_slice(&position);
        self.normals.extend_from_slice(&normal);
        self.colors.extend_from_slice(&color);
        vertex_index
    }

    fn push_triangle(&mut self, a: Vertex, b: Vertex, c: Vertex, color: [f32; 4]) {
        let ia = self.push_vertex(a.position, a.normal, color);
        let ib = self.push_vertex(b.position, b.normal, color);
        let ic = self.push_vertex(c.position, c.normal, color);
        self.indices.extend_from_slice(&[ia, ib, ic]);
    }
}

#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
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

fn gradient(
    scalars: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    x: usize,
    y: usize,
    z: usize,
) -> [f32; 3] {
    let xm = wrap(x as isize - 1, nx);
    let xp = wrap(x as isize + 1, nx);
    let ym = wrap(y as isize - 1, ny);
    let yp = wrap(y as isize + 1, ny);
    let zm = wrap(z as isize - 1, nz);
    let zp = wrap(z as isize + 1, nz);

    let dx = (scalars[idx(nx, ny, xp, y, z)] - scalars[idx(nx, ny, xm, y, z)]) * 0.5;
    let dy = (scalars[idx(nx, ny, x, yp, z)] - scalars[idx(nx, ny, x, ym, z)]) * 0.5;
    let dz = (scalars[idx(nx, ny, x, y, zp)] - scalars[idx(nx, ny, x, y, zm)]) * 0.5;
    [dx, dy, dz]
}

fn sample_lerp(scalars_prev: &[f32], scalars_next: &[f32], lerp_t: f32, index: usize) -> f32 {
    let v_prev = scalars_prev[index];
    let v_next = scalars_next[index];
    v_prev + (v_next - v_prev) * lerp_t
}

fn gradient_lerp(
    scalars_prev: &[f32],
    scalars_next: &[f32],
    lerp_t: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    x: usize,
    y: usize,
    z: usize,
) -> [f32; 3] {
    let x_minus = wrap(x as isize - 1, nx);
    let x_plus = wrap(x as isize + 1, nx);
    let y_minus = wrap(y as isize - 1, ny);
    let y_plus = wrap(y as isize + 1, ny);
    let z_minus = wrap(z as isize - 1, nz);
    let z_plus = wrap(z as isize + 1, nz);

    let dx = (sample_lerp(
        scalars_prev,
        scalars_next,
        lerp_t,
        idx(nx, ny, x_plus, y, z),
    ) - sample_lerp(
        scalars_prev,
        scalars_next,
        lerp_t,
        idx(nx, ny, x_minus, y, z),
    )) * 0.5;
    let dy = (sample_lerp(
        scalars_prev,
        scalars_next,
        lerp_t,
        idx(nx, ny, x, y_plus, z),
    ) - sample_lerp(
        scalars_prev,
        scalars_next,
        lerp_t,
        idx(nx, ny, x, y_minus, z),
    )) * 0.5;
    let dz = (sample_lerp(
        scalars_prev,
        scalars_next,
        lerp_t,
        idx(nx, ny, x, y, z_plus),
    ) - sample_lerp(
        scalars_prev,
        scalars_next,
        lerp_t,
        idx(nx, ny, x, y, z_minus),
    )) * 0.5;
    [dx, dy, dz]
}

fn interp_vertex(
    iso: f32,
    p0: [f32; 3],
    p1: [f32; 3],
    v0: f32,
    v1: f32,
    n0: [f32; 3],
    n1: [f32; 3],
) -> Vertex {
    let denom = v1 - v0;
    let t = if denom.abs() < 1e-12 {
        0.5
    } else {
        ((iso - v0) / denom).max(0.0).min(1.0)
    };

    let position = [
        lerp(p0[0], p1[0], t),
        lerp(p0[1], p1[1], t),
        lerp(p0[2], p1[2], t),
    ];
    let normal = [
        lerp(n0[0], n1[0], t),
        lerp(n0[1], n1[1], t),
        lerp(n0[2], n1[2], t),
    ];

    Vertex { position, normal }
}

fn polygonise_tetra(
    out: &mut MeshBuffers,
    iso: f32,
    p: [[f32; 3]; 4],
    v: [f32; 4],
    n: [[f32; 3]; 4],
    col: [f32; 4],
) {
    let mut inside = [0usize; 4];
    let mut outside = [0usize; 4];
    let mut inside_n = 0usize;
    let mut outside_n = 0usize;

    for i in 0..4 {
        if v[i] >= iso {
            inside[inside_n] = i;
            inside_n += 1;
        } else {
            outside[outside_n] = i;
            outside_n += 1;
        }
    }

    match inside_n {
        0 | 4 => {}
        1 => {
            let i = inside[0];
            let o0 = outside[0];
            let o1 = outside[1];
            let o2 = outside[2];
            let a = interp_vertex(iso, p[i], p[o0], v[i], v[o0], n[i], n[o0]);
            let b = interp_vertex(iso, p[i], p[o1], v[i], v[o1], n[i], n[o1]);
            let c = interp_vertex(iso, p[i], p[o2], v[i], v[o2], n[i], n[o2]);
            out.push_triangle(a, b, c, col);
        }
        3 => {
            let o = outside[0];
            let i0 = inside[0];
            let i1 = inside[1];
            let i2 = inside[2];
            let a = interp_vertex(iso, p[o], p[i0], v[o], v[i0], n[o], n[i0]);
            let b = interp_vertex(iso, p[o], p[i1], v[o], v[i1], n[o], n[i1]);
            let c = interp_vertex(iso, p[o], p[i2], v[o], v[i2], n[o], n[i2]);
            out.push_triangle(a, b, c, col);
        }
        2 => {
            let i0 = inside[0];
            let i1 = inside[1];
            let o0 = outside[0];
            let o1 = outside[1];

            let p0 = interp_vertex(iso, p[i0], p[o0], v[i0], v[o0], n[i0], n[o0]);
            let p1 = interp_vertex(iso, p[i0], p[o1], v[i0], v[o1], n[i0], n[o1]);
            let p2 = interp_vertex(iso, p[i1], p[o0], v[i1], v[o0], n[i1], n[o0]);
            let p3 = interp_vertex(iso, p[i1], p[o1], v[i1], v[o1], n[i1], n[o1]);

            // Quad split into two triangles.
            out.push_triangle(p0, p2, p3, col);
            out.push_triangle(p0, p3, p1, col);
        }
        _ => unreachable!(),
    }
}

pub fn mesh_region_append(
    scalars: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    iso: f32,
    color: [f32; 4],
    out: &mut MeshBuffers,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    z0: usize,
    z1: usize,
    stride: usize,
) {
    if nx < 2 || ny < 2 || nz < 2 {
        return;
    }
    if stride == 0 {
        return;
    }

    let cubes_x = nx - 1;
    let cubes_y = ny - 1;
    let cubes_z = nz - 1;

    let fx = cubes_x as f32;
    let fy = cubes_y as f32;
    let fz = cubes_z as f32;

    let x1 = x1.min(cubes_x);
    let y1 = y1.min(cubes_y);
    let z1 = z1.min(cubes_z);

    if x0 >= x1 || y0 >= y1 || z0 >= z1 {
        return;
    }

    let tetra: [[usize; 4]; 6] = [
        [0, 5, 1, 6],
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
    ];

    // We need x + stride <= x1 etc.
    let x_stop = x1.saturating_sub(stride);
    let y_stop = y1.saturating_sub(stride);
    let z_stop = z1.saturating_sub(stride);

    let mut z = z0;
    while z <= z_stop {
        let z1i = z + stride;
        let mut y = y0;
        while y <= y_stop {
            let y1i = y + stride;
            let mut x = x0;
            while x <= x_stop {
                let x1i = x + stride;

                let x0w = x as f32 / fx - 0.5;
                let y0w = y as f32 / fy - 0.5;
                let z0w = z as f32 / fz - 0.5;
                let x1w = x1i as f32 / fx - 0.5;
                let y1w = y1i as f32 / fy - 0.5;
                let z1w = z1i as f32 / fz - 0.5;

                let corner_positions = [
                    [x0w, y0w, z0w],
                    [x1w, y0w, z0w],
                    [x1w, y1w, z0w],
                    [x0w, y1w, z0w],
                    [x0w, y0w, z1w],
                    [x1w, y0w, z1w],
                    [x1w, y1w, z1w],
                    [x0w, y1w, z1w],
                ];

                let corner_indices = [
                    idx(nx, ny, x, y, z),
                    idx(nx, ny, x1i, y, z),
                    idx(nx, ny, x1i, y1i, z),
                    idx(nx, ny, x, y1i, z),
                    idx(nx, ny, x, y, z1i),
                    idx(nx, ny, x1i, y, z1i),
                    idx(nx, ny, x1i, y1i, z1i),
                    idx(nx, ny, x, y1i, z1i),
                ];

                let corner_values = [
                    scalars[corner_indices[0]],
                    scalars[corner_indices[1]],
                    scalars[corner_indices[2]],
                    scalars[corner_indices[3]],
                    scalars[corner_indices[4]],
                    scalars[corner_indices[5]],
                    scalars[corner_indices[6]],
                    scalars[corner_indices[7]],
                ];

                // Early out for fully inside/outside cube.
                let mut any_below = false;
                let mut any_above = false;
                for &scalar in &corner_values {
                    if scalar < iso {
                        any_below = true;
                    } else {
                        any_above = true;
                    }
                }
                if any_below && any_above {
                    let corner_normals = [
                        gradient(scalars, nx, ny, nz, x, y, z),
                        gradient(scalars, nx, ny, nz, x1i, y, z),
                        gradient(scalars, nx, ny, nz, x1i, y1i, z),
                        gradient(scalars, nx, ny, nz, x, y1i, z),
                        gradient(scalars, nx, ny, nz, x, y, z1i),
                        gradient(scalars, nx, ny, nz, x1i, y, z1i),
                        gradient(scalars, nx, ny, nz, x1i, y1i, z1i),
                        gradient(scalars, nx, ny, nz, x, y1i, z1i),
                    ];

                    for tet in tetra {
                        let p4 = [
                            corner_positions[tet[0]],
                            corner_positions[tet[1]],
                            corner_positions[tet[2]],
                            corner_positions[tet[3]],
                        ];
                        let v4 = [
                            corner_values[tet[0]],
                            corner_values[tet[1]],
                            corner_values[tet[2]],
                            corner_values[tet[3]],
                        ];
                        let n4 = [
                            corner_normals[tet[0]],
                            corner_normals[tet[1]],
                            corner_normals[tet[2]],
                            corner_normals[tet[3]],
                        ];
                        polygonise_tetra(out, iso, p4, v4, n4, color);
                    }
                }

                x += stride;
            }
            y += stride;
        }
        z += stride;
    }
}

pub fn mesh_region_append_lerp(
    scalars_prev: &[f32],
    scalars_next: &[f32],
    lerp_t: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    iso: f32,
    color: [f32; 4],
    out: &mut MeshBuffers,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    z0: usize,
    z1: usize,
    stride: usize,
) {
    if nx < 2 || ny < 2 || nz < 2 {
        return;
    }
    if stride == 0 {
        return;
    }
    if scalars_prev.len() != scalars_next.len() {
        return;
    }

    let cubes_x = nx - 1;
    let cubes_y = ny - 1;
    let cubes_z = nz - 1;

    let fx = cubes_x as f32;
    let fy = cubes_y as f32;
    let fz = cubes_z as f32;

    let x1 = x1.min(cubes_x);
    let y1 = y1.min(cubes_y);
    let z1 = z1.min(cubes_z);

    if x0 >= x1 || y0 >= y1 || z0 >= z1 {
        return;
    }

    let tetra: [[usize; 4]; 6] = [
        [0, 5, 1, 6],
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
    ];

    // We need x + stride <= x1 etc.
    let x_stop = x1.saturating_sub(stride);
    let y_stop = y1.saturating_sub(stride);
    let z_stop = z1.saturating_sub(stride);

    let mut z = z0;
    while z <= z_stop {
        let z1i = z + stride;
        let mut y = y0;
        while y <= y_stop {
            let y1i = y + stride;
            let mut x = x0;
            while x <= x_stop {
                let x1i = x + stride;

                let x0w = x as f32 / fx - 0.5;
                let y0w = y as f32 / fy - 0.5;
                let z0w = z as f32 / fz - 0.5;
                let x1w = x1i as f32 / fx - 0.5;
                let y1w = y1i as f32 / fy - 0.5;
                let z1w = z1i as f32 / fz - 0.5;

                let corner_positions = [
                    [x0w, y0w, z0w],
                    [x1w, y0w, z0w],
                    [x1w, y1w, z0w],
                    [x0w, y1w, z0w],
                    [x0w, y0w, z1w],
                    [x1w, y0w, z1w],
                    [x1w, y1w, z1w],
                    [x0w, y1w, z1w],
                ];

                let corner_indices = [
                    idx(nx, ny, x, y, z),
                    idx(nx, ny, x1i, y, z),
                    idx(nx, ny, x1i, y1i, z),
                    idx(nx, ny, x, y1i, z),
                    idx(nx, ny, x, y, z1i),
                    idx(nx, ny, x1i, y, z1i),
                    idx(nx, ny, x1i, y1i, z1i),
                    idx(nx, ny, x, y1i, z1i),
                ];

                let corner_values = [
                    sample_lerp(scalars_prev, scalars_next, lerp_t, corner_indices[0]),
                    sample_lerp(scalars_prev, scalars_next, lerp_t, corner_indices[1]),
                    sample_lerp(scalars_prev, scalars_next, lerp_t, corner_indices[2]),
                    sample_lerp(scalars_prev, scalars_next, lerp_t, corner_indices[3]),
                    sample_lerp(scalars_prev, scalars_next, lerp_t, corner_indices[4]),
                    sample_lerp(scalars_prev, scalars_next, lerp_t, corner_indices[5]),
                    sample_lerp(scalars_prev, scalars_next, lerp_t, corner_indices[6]),
                    sample_lerp(scalars_prev, scalars_next, lerp_t, corner_indices[7]),
                ];

                // Early out for fully inside/outside cube.
                let mut any_below = false;
                let mut any_above = false;
                for &scalar in &corner_values {
                    if scalar < iso {
                        any_below = true;
                    } else {
                        any_above = true;
                    }
                }
                if any_below && any_above {
                    let corner_normals = [
                        gradient_lerp(scalars_prev, scalars_next, lerp_t, nx, ny, nz, x, y, z),
                        gradient_lerp(scalars_prev, scalars_next, lerp_t, nx, ny, nz, x1i, y, z),
                        gradient_lerp(scalars_prev, scalars_next, lerp_t, nx, ny, nz, x1i, y1i, z),
                        gradient_lerp(scalars_prev, scalars_next, lerp_t, nx, ny, nz, x, y1i, z),
                        gradient_lerp(scalars_prev, scalars_next, lerp_t, nx, ny, nz, x, y, z1i),
                        gradient_lerp(scalars_prev, scalars_next, lerp_t, nx, ny, nz, x1i, y, z1i),
                        gradient_lerp(
                            scalars_prev,
                            scalars_next,
                            lerp_t,
                            nx,
                            ny,
                            nz,
                            x1i,
                            y1i,
                            z1i,
                        ),
                        gradient_lerp(scalars_prev, scalars_next, lerp_t, nx, ny, nz, x, y1i, z1i),
                    ];

                    for tet in tetra {
                        let p4 = [
                            corner_positions[tet[0]],
                            corner_positions[tet[1]],
                            corner_positions[tet[2]],
                            corner_positions[tet[3]],
                        ];
                        let v4 = [
                            corner_values[tet[0]],
                            corner_values[tet[1]],
                            corner_values[tet[2]],
                            corner_values[tet[3]],
                        ];
                        let n4 = [
                            corner_normals[tet[0]],
                            corner_normals[tet[1]],
                            corner_normals[tet[2]],
                            corner_normals[tet[3]],
                        ];
                        polygonise_tetra(out, iso, p4, v4, n4, color);
                    }
                }

                x += stride;
            }
            y += stride;
        }
        z += stride;
    }
}

pub fn generate_isosurface_mesh(
    scalars: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    iso: f32,
    color: [f32; 4],
    out: &mut MeshBuffers,
) {
    out.clear();
    mesh_region_append(
        scalars,
        nx,
        ny,
        nz,
        iso,
        color,
        out,
        0,
        nx - 1,
        0,
        ny - 1,
        0,
        nz - 1,
        1,
    );
}
