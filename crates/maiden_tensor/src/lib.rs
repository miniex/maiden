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
    pub fn new(shape: Vec<usize>) -> CudaResult<Self> {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let buffer = Arc::new(CudaBuffer::new(size)?);
        let strides = Self::compute_strides(&shape);

        Ok(Self {
            buffer,
            shape,
            strides,
        })
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> CudaResult<Self> {
        let expected_size = shape.iter().product::<usize>();
        if data.len() != expected_size {
            return Err(CudaError::ShapeMismatch);
        }

        let mut tensor = Self::new(shape)?;

        if let Some(buffer) = Arc::get_mut(&mut tensor.buffer) {
            buffer.copy_from_host(&data)?;
        } else {
            return Err(CudaError::AllocationFailed);
        }

        Ok(tensor)
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

        let mut result = Tensor::new(self.shape.clone())?;
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

        let mut result = Tensor::new(self.shape.clone())?;
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
    fn test_tensor_creation() -> CudaResult<()> {
        let shape = vec![2, 3];
        let tensor = Tensor::new(shape.clone())?;

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.strides(), &vec![3, 1]);
        Ok(())
    }

    #[test]
    fn test_tensor_from_vec() -> CudaResult<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::from_vec(data.clone(), shape)?;

        let result = tensor.to_vec()?;
        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2];

        let result = Tensor::from_vec(data, shape);
        assert!(matches!(result, Err(CudaError::ShapeMismatch)));
    }

    #[test]
    fn test_tensor_addition() -> CudaResult<()> {
        let tensor1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let tensor2 = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;

        let result = tensor1.add(&tensor2)?;
        let data = result.to_vec()?;

        assert_eq!(data, vec![6.0, 8.0, 10.0, 12.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_multiplication() -> CudaResult<()> {
        let tensor1 = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2])?;
        let tensor2 = Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2])?;

        let result = tensor1.mul(&tensor2)?;
        let data = result.to_vec()?;

        assert_eq!(data, vec![6.0, 12.0, 20.0, 30.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_operation_shape_mismatch() -> CudaResult<()> {
        let tensor1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
        let tensor2 = Tensor::from_vec(vec![1.0, 2.0], vec![2])?;

        let add_result = tensor1.add(&tensor2);
        assert!(matches!(add_result, Err(CudaError::ShapeMismatch)));

        let mul_result = tensor1.mul(&tensor2);
        assert!(matches!(mul_result, Err(CudaError::ShapeMismatch)));
        Ok(())
    }

    #[test]
    fn test_tensor_strides() -> CudaResult<()> {
        let tensor = Tensor::new(vec![2, 3, 4])?;
        assert_eq!(tensor.strides(), &vec![12, 4, 1]);

        let tensor2 = Tensor::new(vec![3, 2])?;
        assert_eq!(tensor2.strides(), &vec![2, 1]);
        Ok(())
    }
}
