use maiden_cuda::prelude::Tensor;

fn main() {
    let shape = vec![2, 3, 4];
    let mut data1 = Vec::new();
    for i in 0..24 {
        data1.push(i as f32);
    }
    let tensor1 = Tensor::from_vec(data1, shape.clone()).expect("Failed to create tensor1");

    let mut data2 = Vec::new();
    for i in 0..24 {
        data2.push((i as f32) * 2.0);
    }
    let tensor2 = Tensor::from_vec(data2, shape).expect("Failed to create tensor2");

    println!("Tensor 1:");
    println!("Shape: {:?}", tensor1.shape());
    println!("Strides: {:?}", tensor1.strides());
    println!(
        "Data: {:?}",
        tensor1.to_vec().expect("Failed to get tensor1 data")
    );

    println!("\nTensor 2:");
    println!("Shape: {:?}", tensor2.shape());
    println!("Strides: {:?}", tensor2.strides());
    println!(
        "Data: {:?}",
        tensor2.to_vec().expect("Failed to get tensor2 data")
    );

    let sum_tensor = tensor1.add(&tensor2).expect("Failed to add tensors");
    println!("\nAddition Result:");
    println!("Shape: {:?}", sum_tensor.shape());
    println!(
        "Data: {:?}",
        sum_tensor.to_vec().expect("Failed to get sum data")
    );

    let mul_tensor = tensor1.mul(&tensor2).expect("Failed to multiply tensors");
    println!("\nMultiplication Result:");
    println!("Shape: {:?}", mul_tensor.shape());
    println!(
        "Data: {:?}",
        mul_tensor
            .to_vec()
            .expect("Failed to get multiplication data")
    );

    let sum_data = sum_tensor.to_vec().expect("Failed to get sum data");
    let mul_data = mul_tensor
        .to_vec()
        .expect("Failed to get multiplication data");

    println!("\nValidation:");
    println!("First element: {} + {} = {}", 0.0, 0.0, sum_data[0]);
    println!("First element: {} * {} = {}", 0.0, 0.0, mul_data[0]);
    println!("Last element: {} + {} = {}", 23.0, 46.0, sum_data[23]);
    println!("Last element: {} * {} = {}", 23.0, 46.0, mul_data[23]);
}
