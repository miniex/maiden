use crate::Tensor;
use maidenx_core::{
    buffer::DeviceBuffer,
    device::Device,
    error::{MaidenXError, Result, TensorError},
};
#[cfg(feature = "cuda")]
use maidenx_cuda_kernels::tensor_ops::{
    cuda_tensor_add, cuda_tensor_div, cuda_tensor_mat_mul, cuda_tensor_mul, cuda_tensor_scalar_mul,
};

impl Tensor {
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

    pub fn scalar_mul(&self, scalar: f32) -> Result<Self> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        };

        match &device {
            Device::Cpu => {
                let data = self
                    .to_vec()?
                    .iter()
                    .map(|x| x * scalar)
                    .collect::<Vec<f32>>();
                result.buffer.copy_from_host(&data)?;
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
}
