pub fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
