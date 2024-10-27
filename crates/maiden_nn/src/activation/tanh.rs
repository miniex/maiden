use crate::module::Module;
use maiden_cuda_core::error::CudaResult;
use maiden_cuda_kernels::nn_ops::cuda_tanh_forward;
use maiden_tensor::Tensor;

#[derive(Default, Module)]
pub struct Tanh {
    inplace: bool,
}

impl Tanh {
    pub fn new() -> Self {
        Tanh::default()
    }

    pub fn inplace(mut self, inplace: bool) -> Self {
        self.inplace = inplace;
        self
    }

    pub fn forward(&self, input: &Tensor) -> CudaResult<Tensor> {
        let mut output = if self.inplace {
            input.clone()
        } else {
            Tensor::zeros(input.shape())?
        };

        unsafe {
            cuda_tanh_forward(
                output.data_mut().as_mut_ptr(),
                input.data().as_ptr(),
                input.size(),
            )?;
        }

        output.reshape(input.shape())?;

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maiden_cuda_core::error::CudaResult;
    use maiden_tensor::Tensor;
    use std::f32;

    #[test]
    fn test_tanh_forward() -> CudaResult<()> {
        let tanh = Tanh::new();
        let input = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0])?;
        let output = tanh.forward(&input)?;

        let result = output.to_vec()?;
        let expected: Vec<f32> = input.to_vec()?.iter().map(|&x| x.tanh()).collect();

        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_tanh_inplace() -> CudaResult<()> {
        let tanh = Tanh::new().inplace(true);
        let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = Tensor::new(input_data.clone())?;
        let output = tanh.forward(&input)?;

        assert_eq!(input.data().as_ptr(), output.data().as_ptr());

        let result = output.to_vec()?;
        let expected: Vec<f32> = input_data.iter().map(|&x| x.tanh()).collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
        Ok(())
    }
}
