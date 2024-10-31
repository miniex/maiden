use crate::Tensor;
use maidenx_core::{
    buffer::DeviceBuffer,
    device::Device,
    error::{MaidenXError, Result, TensorError},
};
#[cfg(feature = "cuda")]
use maidenx_cuda_kernels::tensor_ops::{
    cuda_tensor_add, cuda_tensor_div, cuda_tensor_mat_mul, cuda_tensor_mean, cuda_tensor_mul,
    cuda_tensor_pow, cuda_tensor_scalar_add, cuda_tensor_scalar_div, cuda_tensor_scalar_mul,
    cuda_tensor_scalar_sub, cuda_tensor_sub, cuda_tensor_sum, cuda_tensor_transpose,
};
use std::cell::RefCell;

impl Tensor {
    #[inline(always)]
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot add tensors with shapes {:?} and {:?}",
                self.shape, other.shape
            ))
            .into());
        }

        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad || other.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let a = self.to_vec()?;
                let b = other.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops::add(&a, &b)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_add(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        other.buffer.as_ptr(),
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

    pub(crate) fn add_internal(&self, other: &Tensor) -> Result<Tensor> {
        self.add(other)
    }

    #[inline(always)]
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot divide tensors with shapes {:?} and {:?}",
                self.shape, other.shape
            ))
            .into());
        }

        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad || other.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let a = self.to_vec()?;
                let b = other.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops::div(&a, &b)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_div(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        other.buffer.as_ptr(),
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

    pub(crate) fn div_internal(&self, other: &Tensor) -> Result<Tensor> {
        self.div(other)
    }

    pub fn mat_mul(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::DimensionMismatch(
                "Both tensors must be 2-dimensional for matrix multiplication".into(),
            )
            .into());
        }

        let m = self.shape[0];
        let k = self.shape[1];

        if k != other.shape[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot multiply matrix with shape {:?} and {:?}",
                self.shape, other.shape
            ))
            .into());
        }
        let n = other.shape[1];

        let result_shape = vec![m, n];
        let result_strides = Self::compute_strides(&result_shape);
        let result_size = m * n * std::mem::size_of::<f32>();

        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(result_size, &device)?,
            shape: result_shape,
            strides: result_strides,
            grad: RefCell::new(None),
            requires_grad: self.requires_grad || other.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let a = self.to_vec()?;
                let b = other.to_vec()?;
                let result_data =
                    maidenx_cpu_core::ops::tensor_ops::mat_mul(&a, &self.shape, &b, &other.shape)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_mat_mul(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        other.buffer.as_ptr(),
                        m as i32,
                        n as i32,
                        k as i32,
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

    pub fn mean(&self) -> Result<Tensor> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(std::mem::size_of::<f32>(), &device)?,
            shape: vec![1],
            strides: vec![1],
            grad: RefCell::new(None),
            requires_grad: self.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let data = self.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops::mean(&data)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_mean(
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

    #[inline(always)]
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot multiply tensors with shapes {:?} and {:?}",
                self.shape, other.shape
            ))
            .into());
        }

        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad || other.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let a = self.to_vec()?;
                let b = other.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops::mul(&a, &b)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_mul(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        other.buffer.as_ptr(),
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

    pub(crate) fn mul_internal(&self, other: &Tensor) -> Result<Tensor> {
        self.mul(other)
    }

    pub fn pow(&self, exponent: f32) -> Result<Self> {
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
                let result_data = maidenx_cpu_core::ops::tensor_ops::pow(&data, exponent)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_pow(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        exponent,
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

    #[inline(always)]
    pub fn scalar_add(&self, scalar: f32) -> Result<Self> {
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
                let result_data = maidenx_cpu_core::ops::tensor_ops::scalar_add(&data, scalar)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_scalar_add(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        scalar,
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

    #[inline(always)]
    pub fn scalar_div(&self, scalar: f32) -> Result<Self> {
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
                let result_data = maidenx_cpu_core::ops::tensor_ops::scalar_div(&data, scalar)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_scalar_div(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        scalar,
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

    #[inline(always)]
    pub fn scalar_mul(&self, scalar: f32) -> Result<Self> {
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
                let result_data = maidenx_cpu_core::ops::tensor_ops::scalar_mul(&data, scalar)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_scalar_mul(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        scalar,
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

    #[inline(always)]
    pub fn scalar_sub(&self, scalar: f32) -> Result<Self> {
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
                let result_data = maidenx_cpu_core::ops::tensor_ops::scalar_sub(&data, scalar)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_scalar_sub(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        scalar,
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

    #[inline(always)]
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot add tensors with shapes {:?} and {:?}",
                self.shape, other.shape
            ))
            .into());
        }

        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            grad: RefCell::new(None),
            requires_grad: self.requires_grad || other.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let a = self.to_vec()?;
                let b = other.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops::sub(&a, &b)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_sub(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        other.buffer.as_ptr(),
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

    pub(crate) fn sub_internal(&self, other: &Tensor) -> Result<Tensor> {
        self.sub(other)
    }

    pub fn sum(&self) -> Result<Tensor> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(std::mem::size_of::<f32>(), &device)?,
            shape: vec![1],
            strides: vec![1],
            grad: RefCell::new(None),
            requires_grad: self.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let data = self.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops::sum(&data)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_sum(
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

    pub fn transpose(&self, rows: usize, cols: usize) -> Result<Tensor> {
        if self.size() != rows * cols {
            return Err(MaidenXError::InvalidArgument(
                "Invalid dimensions for transpose".into(),
            ));
        }

        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: vec![cols, rows],
            strides: vec![rows, 1],
            grad: RefCell::new(None),
            requires_grad: self.requires_grad,
        };

        match &device {
            Device::Cpu => {
                let data = self.to_vec()?;
                let result_data = maidenx_cpu_core::ops::tensor_ops::transpose(&data, rows, cols)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                unsafe {
                    cuda_tensor_transpose(
                        result.buffer.as_mut_ptr(),
                        self.buffer.as_ptr(),
                        rows,
                        cols,
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
    use maidenx_core::error::MaidenXError;

    #[test]
    fn test_tensor_add() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let tensor2 = Tensor::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]])?;

        let result = tensor1.add(&tensor2)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, vec![6.0, 8.0, 10.0, 12.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_add_shape_mismatch() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let tensor2 = Tensor::new(vec![vec![5.0], vec![7.0]])?;

        match tensor1.add(&tensor2) {
            Err(MaidenXError::TensorError(TensorError::ShapeMismatch(_))) => Ok(()),
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_tensor_div() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![8.0, 3.0], vec![4.0, 5.0]])?;
        let tensor2 = Tensor::new(vec![vec![2.0, 3.0], vec![1.0, 2.0]])?;

        let result = tensor1.div(&tensor2)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, vec![4.0, 1.0, 4.0, 2.5]);
        Ok(())
    }

    #[test]
    fn test_tensor_div_shape_mismatch() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]])?;
        let tensor2 = Tensor::new(vec![vec![3.0], vec![1.0]])?;

        match tensor1.div(&tensor2) {
            Err(MaidenXError::TensorError(TensorError::ShapeMismatch(_))) => Ok(()),
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_matrix_multiplication() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let tensor2 = Tensor::new(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]])?;

        let result = tensor1.mat_mul(&tensor2)?;
        assert_eq!(result.shape(), &[2, 2]);

        let result_data = result.to_vec()?;
        let expected = [
            1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0,
            1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0,
            4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0,
            4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0,
        ];

        for (actual, expected) in result_data.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "Expected {}, got {}",
                expected,
                actual
            );
        }

        Ok(())
    }

    #[test]
    fn test_mean() -> Result<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let result = tensor.mean()?;

        assert_eq!(result.shape(), &[1]);
        assert!((result.to_vec()?[0] - 2.5).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_matrix_multiplication_invalid_shape() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let tensor2 = Tensor::new(vec![vec![7.0, 8.0], vec![9.0, 10.0]])?;

        match tensor1.mat_mul(&tensor2) {
            Err(MaidenXError::TensorError(TensorError::ShapeMismatch(_))) => Ok(()),
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_tensor_mul() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]])?;
        let tensor2 = Tensor::new(vec![vec![3.0, 2.0], vec![1.0, 4.0]])?;

        let result = tensor1.mul(&tensor2)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, vec![6.0, 6.0, 4.0, 20.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_mul_shape_mismatch() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]])?;
        let tensor2 = Tensor::new(vec![vec![3.0], vec![1.0]])?;

        match tensor1.mul(&tensor2) {
            Err(MaidenXError::TensorError(TensorError::ShapeMismatch(_))) => Ok(()),
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_pow() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let exponent = 2.0;

        let result = tensor1.pow(exponent)?;

        assert_eq!(result.to_vec()?, vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);

        Ok(())
    }

    #[test]
    fn test_scalar_mul() -> Result<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let result = tensor.scalar_mul(2.0)?;

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, vec![2.0, 4.0, 6.0, 8.0]);
        Ok(())
    }

    #[test]
    fn test_sum() -> Result<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let result = tensor.sum()?;

        assert_eq!(result.shape(), &[1]);
        assert!((result.to_vec()?[0] - 10.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_transpose() -> Result<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let result = tensor.transpose(2, 3)?;

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.to_vec()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_transpose_invalid_shape() -> Result<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;

        match tensor.transpose(3, 3) {
            Err(MaidenXError::InvalidArgument(_)) => Ok(()),
            _ => panic!("Expected InvalidArgument error for incorrect dimensions"),
        }
    }

    #[test]
    fn test_transpose_square() -> Result<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let result = tensor.transpose(2, 2)?;

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, vec![1.0, 3.0, 2.0, 4.0]);
        Ok(())
    }
}
