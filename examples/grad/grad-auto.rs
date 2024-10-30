use maidenx::prelude::*;
use std::f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_current_device(Device::cpu())?;

    let x = Tensor::linspace(f32::consts::PI, f32::consts::PI, 2000)?;

    let learning_rate = 1e-6;
    for t in 1..2000 {}

    Ok(())
}
