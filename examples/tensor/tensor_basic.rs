use maiden_cuda::prelude::Tensor;

fn main() {
    let shape = vec![2, 3, 4];
    let mut data = Vec::new();
    for i in 0..24 {
        data.push(i as f32);
    }

    let tensor = Tensor::from_vec(data, shape).expect("Failed to create tensor");

    println!("Tensor shape: {:?}", tensor.shape());
    println!("Tensor strides: {:?}", tensor.strides());
    println!(
        "Tensor data: {:?}",
        tensor.to_vec().expect("Failed to get tensor data")
    );
}
