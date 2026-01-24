use wasm_bindgen::prelude::*;

mod math;
mod meshing;

mod cahn_hilliard;
mod excitable_media;
mod gray_scott;
#[path = "sim.rs"]
mod mesher;
mod rdme;

pub use cahn_hilliard::{CahnHilliardParams, CahnHilliardSimulation};
pub use excitable_media::{ExcitableMediaParams, ExcitableMediaSimulation};
pub use gray_scott::{GrayScottParams, Simulation};
pub use mesher::ScalarFieldMesher;
pub use rdme::{StochasticRdmeParams, StochasticRdmeSimulation};

#[wasm_bindgen]
pub fn init_thread_pool(num_threads: usize) -> js_sys::Promise {
    wasm_bindgen_rayon::init_thread_pool(num_threads)
}

#[wasm_bindgen]
pub fn rayon_num_threads() -> usize {
    rayon::current_num_threads()
}

#[wasm_bindgen(start)]
pub fn wasm_start() {
    // Keep empty: explicit initialization in JS.
}
