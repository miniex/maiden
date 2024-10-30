use crate::module::Module;
use maidenx_core::{
    device::Device,
    error::{MaidenXError, Result},
};
#[cfg(feature = "cuda")]
use maidenx_cuda_kernels::nn_ops::cuda_relu_forward;
use maidenx_tensor::Tensor;

#[derive(Default, Module)]
pub struct ReLU {
    inplace: bool,
}

#[allow(unused_assignments)]
impl ReLU {
    pub fn new() -> Self {
        ReLU::default()
    }

    pub fn inplace(mut self, inplace: bool) -> Self {
        self.inplace = inplace;
        self
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = if self.inplace {
            input.clone()
        } else {
            Tensor::zeros(input.shape())?
        };

        let device = maidenx_core::device::get_current_device();
        match device {
            Device::Cpu => {
                let input_data = input.to_vec()?;
                let result = maidenx_cpu_core::ops::nn_ops::activation::relu_forward(&input_data)?;
                output = Tensor::from_vec(result, input.shape())?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_relu_forward(
                        output.data_mut().as_mut_ptr(),
                        input.data().as_ptr(),
                        input.size(),
                    )
                    .map_err(MaidenXError::from)?;
                }
                #[cfg(not(feature = "cuda"))]
                return Err(MaidenXError::UnsupportedOperation(
                    "CUDA operations are not available - feature not enabled".into(),
                ));
            }
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
    use maidenx_core::error::Result;
    use maidenx_tensor::Tensor;

    #[test]
    fn test_relu_forward() -> Result<()> {
        let relu = ReLU::new();
        let input = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0])?;
        let output = relu.forward(&input)?;

        let result = output.to_vec()?;
        let expected = [0.0, 0.0, 0.0, 1.0, 2.0];

        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_relu_inplace() -> Result<()> {
        let relu = ReLU::new().inplace(true);
        let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = Tensor::new(input_data.clone())?;
        let output = relu.forward(&input)?;

        assert_eq!(input.data().as_ptr(), output.data().as_ptr());

        let result = output.to_vec()?;
        let expected = [0.0, 0.0, 0.0, 1.0, 2.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
        Ok(())
    }
}
