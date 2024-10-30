use maidenx::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    set_current_device(Device::cpu())?;

    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;

    let linear = nn::Linear::new_with_bias(2, 3, true)?;

    let linear_output = linear.forward(&input)?;

    let relu = nn::ReLU::new();
    let relu_output = relu.forward(&linear_output)?;

    let sigmoid = nn::Sigmoid::new();
    let sigmoid_output = sigmoid.forward(&relu_output)?;

    let tanh = nn::Tanh::new();
    let tanh_output = tanh.forward(&sigmoid_output)?;

    println!("Input:\n{}", input);
    println!("Linear output:\n{}", linear_output);
    println!("ReLU output:\n{}", relu_output);
    println!("Sigmoid output:\n{}", sigmoid_output);
    println!("Tanh output:\n{}", tanh_output);

    Ok(())
}
