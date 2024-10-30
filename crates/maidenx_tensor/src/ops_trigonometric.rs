use crate::Tensor;
use maidenx_core::{
    buffer::DeviceBuffer,
    device::Device,
    error::{MaidenXError, Result},
};
#[cfg(feature = "cuda")]
use maidenx_cuda_kernels::tensor_ops_trigonometric::{
    cuda_tensor_acos, cuda_tensor_asin, cuda_tensor_atan, cuda_tensor_cos, cuda_tensor_sin,
    cuda_tensor_tan,
};
use std::cell::RefCell;

impl Tensor {
    pub fn sin(&self) -> Result<Self> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let data = self.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::sin(&data)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_sin(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        self.size(),
                    )
                    .map_err(MaidenXError::from)?;
                }
                #[cfg(not(feature = "cuda"))]
                return Err(MaidenXError::UnsupportedOperation(
                    "CUDA operations are not available - feature not enabled".into(),
                ));
            }
        }

        Ok(result)
    }

    pub fn cos(&self) -> Result<Self> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let data = self.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::cos(&data)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_cos(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        self.size(),
                    )
                    .map_err(MaidenXError::from)?;
                }
                #[cfg(not(feature = "cuda"))]
                return Err(MaidenXError::UnsupportedOperation(
                    "CUDA operations are not available - feature not enabled".into(),
                ));
            }
        }

        Ok(result)
    }

    pub fn tan(&self) -> Result<Self> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let data = self.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::tan(&data)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_tan(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        self.size(),
                    )
                    .map_err(MaidenXError::from)?;
                }
                #[cfg(not(feature = "cuda"))]
                return Err(MaidenXError::UnsupportedOperation(
                    "CUDA operations are not available - feature not enabled".into(),
                ));
            }
        }

        Ok(result)
    }

    pub fn asin(&self) -> Result<Self> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let data = self.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::asin(&data)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_asin(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        self.size(),
                    )
                    .map_err(MaidenXError::from)?;
                }
                #[cfg(not(feature = "cuda"))]
                return Err(MaidenXError::UnsupportedOperation(
                    "CUDA operations are not available - feature not enabled".into(),
                ));
            }
        }

        Ok(result)
    }

    pub fn acos(&self) -> Result<Self> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let data = self.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::acos(&data)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_acos(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        self.size(),
                    )
                    .map_err(MaidenXError::from)?;
                }
                #[cfg(not(feature = "cuda"))]
                return Err(MaidenXError::UnsupportedOperation(
                    "CUDA operations are not available - feature not enabled".into(),
                ));
            }
        }

        Ok(result)
    }

    pub fn atan(&self) -> Result<Self> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let data = self.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::atan(&data)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_atan(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        self.size(),
                    )
                    .map_err(MaidenXError::from)?;
                }
                #[cfg(not(feature = "cuda"))]
                return Err(MaidenXError::UnsupportedOperation(
                    "CUDA operations are not available - feature not enabled".into(),
                ));
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_tensor_sin() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, PI / 2.0, PI])?;
        let result = tensor.sin()?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[3]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // sin(0) = 0
        assert!((result_data[1] - 1.0).abs() < 1e-5); // sin(π/2) = 1
        assert!((result_data[2] - 0.0).abs() < 1e-5); // sin(π) = 0

        Ok(())
    }

    #[test]
    fn test_tensor_cos() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, PI / 2.0, PI])?;
        let result = tensor.cos()?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[3]);
        assert!((result_data[0] - 1.0).abs() < 1e-5); // cos(0) = 1
        assert!(result_data[1].abs() < 1e-5); // cos(π/2) = 0
        assert!((result_data[2] + 1.0).abs() < 1e-5); // cos(π) = -1

        Ok(())
    }

    #[test]
    fn test_tensor_tan() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, PI / 4.0])?;
        let result = tensor.tan()?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[2]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // tan(0) = 0
        assert!((result_data[1] - 1.0).abs() < 1e-5); // tan(π/4) = 1

        Ok(())
    }

    #[test]
    fn test_tensor_asin() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, 1.0, -1.0, 0.5])?;
        let result = tensor.asin()?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[4]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // asin(0) = 0
        assert!((result_data[1] - PI / 2.0).abs() < 1e-5); // asin(1) = π/2
        assert!((result_data[2] + PI / 2.0).abs() < 1e-5); // asin(-1) = -π/2
        assert!((result_data[3] - (0.5f32.asin())).abs() < 1e-5); // asin(0.5)

        Ok(())
    }

    #[test]
    fn test_tensor_acos() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 0.0, -1.0, 0.5])?;
        let result = tensor.acos()?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[4]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // acos(1) = 0
        assert!((result_data[1] - PI / 2.0).abs() < 1e-5); // acos(0) = π/2
        assert!((result_data[2] - PI).abs() < 1e-5); // acos(-1) = π
        assert!((result_data[3] - (0.5f32.acos())).abs() < 1e-5); // acos(0.5)

        Ok(())
    }

    #[test]
    fn test_tensor_atan() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, 1.0, -1.0])?;
        let result = tensor.atan()?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[3]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // atan(0) = 0
        assert!((result_data[1] - PI / 4.0).abs() < 1e-5); // atan(1) = π/4
        assert!((result_data[2] + PI / 4.0).abs() < 1e-5); // atan(-1) = -π/4

        Ok(())
    }

    #[test]
    fn test_trig_2d_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![vec![0.0, PI / 2.0], vec![PI, 3.0 * PI / 2.0]])?;
        let sin_result = tensor.sin()?;
        let cos_result = tensor.cos()?;

        assert_eq!(sin_result.shape(), &[2, 2]);
        assert_eq!(cos_result.shape(), &[2, 2]);

        let sin_data = sin_result.to_vec()?;
        let cos_data = cos_result.to_vec()?;

        // Check sin values
        assert!(sin_data[0].abs() < 1e-5); // sin(0) = 0
        assert!((sin_data[1] - 1.0).abs() < 1e-5); // sin(π/2) = 1
        assert!(sin_data[2].abs() < 1e-5); // sin(π) = 0
        assert!((sin_data[3] + 1.0).abs() < 1e-5); // sin(3π/2) = -1

        // Check cos values
        assert!((cos_data[0] - 1.0).abs() < 1e-5); // cos(0) = 1
        assert!(cos_data[1].abs() < 1e-5); // cos(π/2) = 0
        assert!((cos_data[2] + 1.0).abs() < 1e-5); // cos(π) = -1
        assert!(cos_data[3].abs() < 1e-5); // cos(3π/2) = 0

        Ok(())
    }

    #[test]
    fn test_inverse_trig_2d_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![vec![0.0, 0.5], vec![-0.5, -1.0]])?;
        let asin_result = tensor.asin()?;
        let acos_result = tensor.acos()?;

        assert_eq!(asin_result.shape(), &[2, 2]);
        assert_eq!(acos_result.shape(), &[2, 2]);

        let asin_data = asin_result.to_vec()?;
        let acos_data = acos_result.to_vec()?;

        // Check asin values
        assert!(asin_data[0].abs() < 1e-5); // asin(0) = 0
        assert!((asin_data[1] - PI / 6.0).abs() < 1e-5); // asin(0.5) ≈ π/6
        assert!((asin_data[2] + PI / 6.0).abs() < 1e-5); // asin(-0.5) ≈ -π/6
        assert!((asin_data[3] + PI / 2.0).abs() < 1e-5); // asin(-1) = -π/2

        // Check acos values
        assert!((acos_data[0] - PI / 2.0).abs() < 1e-5); // acos(0) = π/2
        assert!((acos_data[1] - PI / 3.0).abs() < 1e-5); // acos(0.5) ≈ π/3
        assert!((acos_data[2] - 2.0 * PI / 3.0).abs() < 1e-5); // acos(-0.5) ≈ 2π/3
        assert!((acos_data[3] - PI).abs() < 1e-5); // acos(-1) = π

        Ok(())
    }

    #[test]
    fn test_large_tensor_trig() -> Result<()> {
        let shape = [100, 100];
        let tensor = Tensor::zeros(&shape)?;

        // Test all trig functions with large tensor
        let sin_result = tensor.sin()?;
        let cos_result = tensor.cos()?;
        let tan_result = tensor.tan()?;
        let asin_result = tensor.asin()?;
        let acos_result = tensor.acos()?;
        let atan_result = tensor.atan()?;

        // Verify shapes are preserved
        assert_eq!(sin_result.shape(), &shape);
        assert_eq!(cos_result.shape(), &shape);
        assert_eq!(tan_result.shape(), &shape);
        assert_eq!(asin_result.shape(), &shape);
        assert_eq!(acos_result.shape(), &shape);
        assert_eq!(atan_result.shape(), &shape);

        // For zeros tensor:
        let sin_data = sin_result.to_vec()?;
        let cos_data = cos_result.to_vec()?;
        let tan_data = tan_result.to_vec()?;

        for i in 0..10 {
            // Check first few elements
            assert!(sin_data[i].abs() < 1e-5); // sin(0) = 0
            assert!((cos_data[i] - 1.0).abs() < 1e-5); // cos(0) = 1
            assert!(tan_data[i].abs() < 1e-5); // tan(0) = 0
        }

        Ok(())
    }
}
