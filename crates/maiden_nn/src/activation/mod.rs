pub mod relu;
pub mod sigmoid;
pub mod tanh;

pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use tanh::Tanh;

#[cfg(test)]
mod tests {
    use super::*;
    use maiden_cuda_core::error::CudaResult;
    use maiden_tensor::Tensor;

    #[test]
    fn test_activation_shapes() -> CudaResult<()> {
        let input = Tensor::new(vec![vec![1.0, -1.0], vec![-2.0, 2.0]])?;

        // ReLU shape test
        let relu = ReLU::new();
        let relu_output = relu.forward(&input)?;
        assert_eq!(relu_output.shape(), input.shape());

        // Sigmoid shape test
        let sigmoid = Sigmoid::new();
        let sigmoid_output = sigmoid.forward(&input)?;
        assert_eq!(sigmoid_output.shape(), input.shape());

        // Tanh shape test
        let tanh = Tanh::new();
        let tanh_output = tanh.forward(&input)?;
        assert_eq!(tanh_output.shape(), input.shape());

        Ok(())
    }
}
