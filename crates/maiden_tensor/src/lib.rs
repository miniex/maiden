mod convert;

use convert::TensorData;
use cuda_core::prelude::{CudaBuffer, CudaError, CudaResult};
use cuda_kernels::tensor_ops::{cuda_tensor_add, cuda_tensor_mul};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Tensor {
    buffer: Arc<CudaBuffer>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new<T>(data: T) -> CudaResult<Self>
    where
        T: TensorData,
    {
        let shape = data.to_shape();
        let flat_data = data.to_flat_vec();
        let size = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let buffer = Arc::new(CudaBuffer::new(size)?);
        let strides = Self::compute_strides(&shape);

        let mut tensor = Self {
            buffer,
            shape,
            strides,
        };

        if let Some(buffer) = Arc::get_mut(&mut tensor.buffer) {
            buffer.copy_from_host(&flat_data)?;
        } else {
            return Err(CudaError::AllocationFailed);
        }

        Ok(tensor)
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self, dim: usize) -> Option<usize> {
        self.shape.get(dim).copied()
    }

    pub fn to_vec(&self) -> CudaResult<Vec<f32>> {
        let num_elements = self.shape.iter().product::<usize>();
        let mut result = vec![0.0f32; num_elements];
        self.buffer.copy_to_host(&mut result)?;
        Ok(result)
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn add(&self, other: &Tensor) -> CudaResult<Tensor> {
        if self.shape != other.shape {
            return Err(CudaError::ShapeMismatch);
        }

        let mut result = Self {
            buffer: Arc::new(CudaBuffer::new(self.buffer.len())?),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        };

        let size = self.shape.iter().product::<usize>();

        unsafe {
            cuda_tensor_add(
                Arc::get_mut(&mut result.buffer).unwrap().as_mut_ptr(),
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                size,
            )?;
        }

        Ok(result)
    }

    pub fn mul(&self, other: &Tensor) -> CudaResult<Tensor> {
        if self.shape != other.shape {
            return Err(CudaError::ShapeMismatch);
        }

        let mut result = Self {
            buffer: Arc::new(CudaBuffer::new(self.buffer.len())?),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        };

        let size = self.shape.iter().product::<usize>();

        unsafe {
            cuda_tensor_mul(
                Arc::get_mut(&mut result.buffer).unwrap().as_mut_ptr(),
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                size,
            )?;
        }

        Ok(result)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_tensor() -> CudaResult<()> {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data.clone())?;

        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.to_vec()?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_2d_tensor() -> CudaResult<()> {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let tensor = Tensor::new(data)?;

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_3d_tensor() -> CudaResult<()> {
        let data = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ];
        let tensor = Tensor::new(data)?;

        assert_eq!(tensor.shape(), &[2, 2, 2]);
        assert_eq!(tensor.ndim(), 3);
        assert_eq!(
            tensor.to_vec()?,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        Ok(())
    }

    #[test]
    fn test_4d_tensor() -> CudaResult<()> {
        let data = vec![vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ]];
        let tensor = Tensor::new(data)?;

        assert_eq!(tensor.shape(), &[1, 2, 2, 2]);
        assert_eq!(tensor.ndim(), 4);
        assert_eq!(
            tensor.to_vec()?,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        Ok(())
    }
}
