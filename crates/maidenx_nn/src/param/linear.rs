use crate::module::Module;
use maidenx_core::{
    device::Device,
    error::{MaidenXError, Result},
};
use maidenx_cuda_kernels::nn_ops::{cuda_bilinear_forward, cuda_linear_forward};
use maidenx_tensor::Tensor;

#[derive(Module)]
pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        Self::new_with_bias(in_features, out_features, false)
    }

    pub fn new_with_bias(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
        let k = 1.0 / (in_features as f32).sqrt();
        let weight = Tensor::randn(&[out_features, in_features])
            .map_err(MaidenXError::from)?
            .mul_scalar(k)
            .map_err(MaidenXError::from)?;

        let bias = if bias {
            Some(Tensor::zeros(&[out_features]).map_err(MaidenXError::from)?)
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.size_dim(0).unwrap_or(1);

        let device = maidenx_core::device::get_current_device();
        match device {
            Device::Cpu => {
                let input_data = input.to_vec()?;
                let weight_data = self.weight.to_vec()?;
                let bias_data = self.bias.as_ref().map(|b| b.to_vec()).transpose()?;

                let result = maidenx_cpu_core::ops::nn_ops::layer::linear::linear_forward(
                    &input_data,
                    &weight_data,
                    bias_data.as_deref(),
                    batch_size as i32,
                    self.out_features as i32,
                    self.in_features as i32,
                )?;

                Tensor::from_vec(result, &[batch_size, self.out_features])
            }
            Device::Cuda(_) => {
                let mut output = Tensor::zeros(&[batch_size, self.out_features])?;
                unsafe {
                    cuda_linear_forward(
                        output.data_mut().as_mut_ptr(),
                        input.data().as_ptr(),
                        self.weight.data().as_ptr(),
                        self.bias.as_ref().map(|b| b.data().as_ptr()),
                        batch_size as i32,
                        self.out_features as i32,
                        self.in_features as i32,
                    )
                    .map_err(MaidenXError::from)?;
                }
                Ok(output)
            }
        }
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.shape() != [self.out_features, self.in_features] {
            return Err(MaidenXError::ShapeMismatch(
                "Weight tensor shape doesn't match layer dimensions".into(),
            ));
        }
        self.weight = weight;
        Ok(())
    }

    pub fn set_bias(&mut self, bias: Tensor) -> Result<()> {
        if bias.shape() != [self.out_features] {
            return Err(MaidenXError::ShapeMismatch(
                "Bias tensor shape doesn't match output dimensions".into(),
            ));
        }
        self.bias = Some(bias);
        Ok(())
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

#[derive(Module)]
pub struct Bilinear {
    in1_features: usize,
    in2_features: usize,
    out_features: usize,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Bilinear {
    pub fn new(in1_features: usize, in2_features: usize, out_features: usize) -> Result<Self> {
        Self::new_with_bias(in1_features, in2_features, out_features, false)
    }

    pub fn new_with_bias(
        in1_features: usize,
        in2_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self> {
        let k = 1.0 / ((in1_features * in2_features) as f32).sqrt();
        let weight = Tensor::randn(&[out_features, in1_features, in2_features])
            .map_err(MaidenXError::from)?
            .mul_scalar(k)
            .map_err(MaidenXError::from)?;

        let bias = if bias {
            Some(Tensor::zeros(&[out_features]).map_err(MaidenXError::from)?)
        } else {
            None
        };

        Ok(Self {
            in1_features,
            in2_features,
            out_features,
            weight,
            bias,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let total_features = self.in1_features + self.in2_features;
        if input.shape()[1] != total_features {
            return Err(MaidenXError::ShapeMismatch(
                "Input tensor shape doesn't match total features".into(),
            ));
        }

        let (input1, input2) = input
            .split_at(1, self.in1_features)
            .map_err(MaidenXError::from)?;
        self.forward_bilinear(&input1, &input2)
    }

    pub fn forward_bilinear(&self, input1: &Tensor, input2: &Tensor) -> Result<Tensor> {
        let batch_size = input1.size_dim(0).unwrap_or(1);

        let device = maidenx_core::device::get_current_device();
        match device {
            Device::Cpu => {
                let input1_data = input1.to_vec()?;
                let input2_data = input2.to_vec()?;
                let weight_data = self.weight.to_vec()?;
                let bias_data = self.bias.as_ref().map(|b| b.to_vec()).transpose()?;

                let result = maidenx_cpu_core::ops::nn_ops::layer::linear::bilinear_forward(
                    &input1_data,
                    &input2_data,
                    &weight_data,
                    bias_data.as_deref(),
                    batch_size as i32,
                    self.out_features as i32,
                    self.in1_features as i32,
                    self.in2_features as i32,
                )?;

                Tensor::from_vec(result, &[batch_size, self.out_features])
            }
            Device::Cuda(_) => {
                let mut output = Tensor::zeros(&[batch_size, self.out_features])?;
                unsafe {
                    cuda_bilinear_forward(
                        output.data_mut().as_mut_ptr(),
                        input1.data().as_ptr(),
                        input2.data().as_ptr(),
                        self.weight.data().as_ptr(),
                        self.bias.as_ref().map(|b| b.data().as_ptr()),
                        batch_size as i32,
                        self.out_features as i32,
                        self.in1_features as i32,
                        self.in2_features as i32,
                    )
                    .map_err(MaidenXError::from)?;
                }
                Ok(output)
            }
        }
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.shape() != [self.out_features, self.in1_features, self.in2_features] {
            return Err(MaidenXError::ShapeMismatch(
                "Weight tensor shape doesn't match layer dimensions".into(),
            ));
        }
        self.weight = weight;
        Ok(())
    }

    pub fn set_bias(&mut self, bias: Tensor) -> Result<()> {
        if bias.shape() != [self.out_features] {
            return Err(MaidenXError::ShapeMismatch(
                "Bias tensor shape doesn't match output dimensions".into(),
            ));
        }
        self.bias = Some(bias);
        Ok(())
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() -> Result<()> {
        let linear = Linear::new(2, 1)?;
        assert_eq!(linear.weight().shape(), &[1, 2]);
        assert!(linear.bias().is_none());

        let linear_with_bias = Linear::new_with_bias(2, 1, true)?;
        assert_eq!(linear_with_bias.weight().shape(), &[1, 2]);
        assert!(linear_with_bias.bias().is_some());
        Ok(())
    }

    #[test]
    fn test_linear_forward() -> Result<()> {
        let linear = Linear::new_with_bias(2, 1, true)?;

        let input = Tensor::from_vec(vec![1.0f32, 1.0], &[1, 2]).map_err(MaidenXError::from)?;
        let output = linear.forward(&input)?;
        assert_eq!(output.shape(), &[1, 1]);
        Ok(())
    }

    #[test]
    fn test_linear_parameters() -> Result<()> {
        let linear = Linear::new(2, 1)?;
        assert_eq!(linear.parameters().len(), 1);

        let linear_with_bias = Linear::new_with_bias(2, 1, true)?;
        assert_eq!(linear_with_bias.parameters().len(), 2);
        Ok(())
    }

    #[test]
    fn test_linear_numerical() -> Result<()> {
        let mut linear = Linear::new_with_bias(1, 1, true)?;

        linear.set_weight(Tensor::from_vec(vec![0.5f32], &[1, 1]).map_err(MaidenXError::from)?)?;
        linear.set_bias(Tensor::from_vec(vec![0.1f32], &[1]).map_err(MaidenXError::from)?)?;

        let input = Tensor::from_vec(vec![2.0f32], &[1, 1]).map_err(MaidenXError::from)?;
        let output = linear.forward(&input)?;

        let output_vec = output.to_vec().map_err(MaidenXError::from)?;
        assert!((output_vec[0] - 1.1).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_bilinear_creation() -> Result<()> {
        let bilinear = Bilinear::new(1, 1, 1)?;
        assert_eq!(bilinear.weight().shape(), &[1, 1, 1]);
        assert!(bilinear.bias().is_none());

        let bilinear_with_bias = Bilinear::new_with_bias(1, 1, 1, true)?;
        assert_eq!(bilinear_with_bias.weight().shape(), &[1, 1, 1]);
        assert!(bilinear_with_bias.bias().is_some());
        Ok(())
    }

    #[test]
    fn test_bilinear_forward() -> Result<()> {
        let bilinear = Bilinear::new(1, 1, 1)?;

        let input1 = Tensor::from_vec(vec![1.0f32], &[1, 1]).map_err(MaidenXError::from)?;
        let input2 = Tensor::from_vec(vec![1.0f32], &[1, 1]).map_err(MaidenXError::from)?;
        let output = bilinear.forward_bilinear(&input1, &input2)?;

        assert_eq!(output.shape(), &[1, 1]);
        Ok(())
    }

    #[test]
    fn test_bilinear_parameters() -> Result<()> {
        let bilinear = Bilinear::new(1, 1, 1)?;
        assert_eq!(bilinear.parameters().len(), 1);

        let bilinear_with_bias = Bilinear::new_with_bias(1, 1, 1, true)?;
        assert_eq!(bilinear_with_bias.parameters().len(), 2);
        Ok(())
    }

    #[test]
    fn test_bilinear_numerical() -> Result<()> {
        let mut bilinear = Bilinear::new_with_bias(1, 1, 1, true)?;

        bilinear
            .set_weight(Tensor::from_vec(vec![0.5f32], &[1, 1, 1]).map_err(MaidenXError::from)?)?;
        bilinear.set_bias(Tensor::from_vec(vec![0.1f32], &[1]).map_err(MaidenXError::from)?)?;

        let input1 = Tensor::from_vec(vec![2.0f32], &[1, 1]).map_err(MaidenXError::from)?;
        let input2 = Tensor::from_vec(vec![2.0f32], &[1, 1]).map_err(MaidenXError::from)?;
        let output = bilinear.forward_bilinear(&input1, &input2)?;

        let output_vec = output.to_vec().map_err(MaidenXError::from)?;
        let expected = 0.1 + (2.0 * 2.0 * 0.5);
        assert!((output_vec[0] - expected).abs() < 1e-5);
        Ok(())
    }
}
