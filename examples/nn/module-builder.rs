use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let model = nn::ModuleBuilder::new()
        .add_layer(nn::Linear::new_with_bias(2, 3, true)?)
        .add_layer(nn::ReLU::new())
        .add_layer(nn::Sigmoid::new())
        .add_layer(nn::Tanh::new());

    let output = model.forward(&input)?;

    println!("Input:\n{}", input);
    println!("Output:\n{}", output);

    Ok(())
}
