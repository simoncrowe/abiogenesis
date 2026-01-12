use wasm_bindgen::prelude::*;

mod math;
mod meshing;

mod rdme;
#[path = "sim.rs"]
mod simulation;

pub use rdme::{StochasticRdmeParams, StochasticRdmeSimulation};
pub use simulation::{GrayScottParams, ScalarFieldMesher, Simulation};

#[wasm_bindgen(start)]
pub fn wasm_start() {
    // Keep empty: explicit initialization in JS.
}
