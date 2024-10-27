use maiden_cuda::prelude::*;

fn main() -> CudaResult<()> {
    // Create 3D tensors with shape [2, 3, 4]
    let tensor1 = Tensor::new(vec![
        vec![
            vec![0.0, 1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0, 7.0],
            vec![8.0, 9.0, 10.0, 11.0],
        ],
        vec![
            vec![12.0, 13.0, 14.0, 15.0],
            vec![16.0, 17.0, 18.0, 19.0],
            vec![20.0, 21.0, 22.0, 23.0],
        ],
    ])?;

    let tensor2 = Tensor::new(vec![
        vec![
            vec![0.0, 2.0, 4.0, 6.0],
            vec![8.0, 10.0, 12.0, 14.0],
            vec![16.0, 18.0, 20.0, 22.0],
        ],
        vec![
            vec![24.0, 26.0, 28.0, 30.0],
            vec![32.0, 34.0, 36.0, 38.0],
            vec![40.0, 42.0, 44.0, 46.0],
        ],
    ])?;

    println!("Tensor 1:");
    println!("Shape: {:?}", tensor1.shape());
    println!("Strides: {:?}", tensor1.strides());
    println!("Number of dimensions: {}", tensor1.ndim());
    println!("Data:\n{}", tensor1);

    println!("\nTensor 2:");
    println!("Shape: {:?}", tensor2.shape());
    println!("Strides: {:?}", tensor2.strides());
    println!("Number of dimensions: {}", tensor2.ndim());
    println!("Data:\n{}", tensor2);

    let sum_tensor = tensor1.add(&tensor2)?;
    println!("\nAddition Result:");
    println!("Shape: {:?}", sum_tensor.shape());
    println!("Data:\n{}", sum_tensor);

    let mul_tensor = tensor1.mul(&tensor2)?;
    println!("\nMultiplication Result:");
    println!("Shape: {:?}", mul_tensor.shape());
    println!("Data:\n{}", mul_tensor);

    let sum_data = sum_tensor.to_vec()?;
    let mul_data = mul_tensor.to_vec()?;

    println!("\nValidation:");
    println!("First element: {} + {} = {}", 0.0, 0.0, sum_data[0]);
    println!("First element: {} * {} = {}", 0.0, 0.0, mul_data[0]);
    println!("Last element: {} + {} = {}", 23.0, 46.0, sum_data[23]);
    println!("Last element: {} * {} = {}", 23.0, 46.0, mul_data[23]);

    Ok(())
}
