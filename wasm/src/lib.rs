use wasm_bindgen::prelude::*;

mod math;
mod meshing;

mod gray_scott;
#[path = "sim.rs"]
mod mesher;
mod rdme;

pub use gray_scott::{GrayScottParams, Simulation};
pub use mesher::ScalarFieldMesher;
pub use rdme::{StochasticRdmeParams, StochasticRdmeSimulation};

#[wasm_bindgen(start)]
pub fn wasm_start() {
    // Keep empty: explicit initialization in JS.
}
