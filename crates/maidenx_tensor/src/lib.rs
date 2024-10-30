mod convert;
mod display;
mod ops;
mod shape;

use convert::TensorData;
use maidenx_core::buffer::DeviceBuffer;
use maidenx_core::error::{Result, TensorError};
use rand::prelude::*;
use rand_distr::Normal;

#[derive(Debug, Clone)]
pub struct Tensor {
    buffer: DeviceBuffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new<T>(data: T) -> Result<Self>
    where
        T: TensorData,
    {
        let shape = data.to_shape();
        let flat_data = data.to_flat_vec();
        let size = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let strides = Self::compute_strides(&shape);

        let device = maidenx_core::device::get_current_device();
        let mut buffer = DeviceBuffer::new(size, &device)?;

        buffer.copy_from_host(&flat_data)?;

        Ok(Self {
            buffer,
            shape,
            strides,
        })
    }

    pub fn from_vec<T: Into<Vec<f32>>>(vec: T, shape: &[usize]) -> Result<Self> {
        let vec = vec.into();
        let total_size: usize = shape.iter().product();

        if vec.len() != total_size {
            return Err(TensorError::ShapeMismatch(format!(
                "Vector length {} does not match shape {:?}",
                vec.len(),
                shape
            ))
            .into());
        }

        let size = total_size * std::mem::size_of::<f32>();
        let device = maidenx_core::device::get_current_device();
        let mut buffer = DeviceBuffer::new(size, &device)?;
        buffer.copy_from_host(&vec)?;

        Ok(Self {
            buffer,
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
        })
    }

    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0f32; size];
        Self::from_vec(data, shape)
    }

    pub fn randn(shape: &[usize]) -> Result<Self> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).map_err(|e| TensorError::DataError(e.to_string()))?;
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
        Self::from_vec(data, shape)
    }

    pub fn data(&self) -> &DeviceBuffer {
        &self.buffer
    }

    pub fn data_mut(&mut self) -> &mut DeviceBuffer {
        &mut self.buffer
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_dim(&self, dim: usize) -> Option<usize> {
        self.shape.get(dim).copied()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn is_empty(&self) -> bool {
        self.shape.iter().any(|&dim| dim == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_core::error::MaidenXError;

    #[test]
    fn test_1d_tensor() -> Result<()> {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(data.clone())?;

        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.to_vec()?, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_2d_tensor() -> Result<()> {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let tensor = Tensor::new(data)?;

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_3d_tensor() -> Result<()> {
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
    fn test_4d_tensor() -> Result<()> {
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
    fn test_matrix_multiplication_invalid_shape() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let tensor2 = Tensor::new(vec![vec![7.0, 8.0], vec![9.0, 10.0]])?;

        match tensor1.mat_mul(&tensor2) {
            Err(MaidenXError::TensorError(TensorError::ShapeMismatch(_))) => Ok(()),
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_from_vec() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.to_vec()?, vec![1.0, 2.0, 3.0]);

        match Tensor::from_vec(vec![1.0, 2.0], &[3]) {
            Err(MaidenXError::TensorError(TensorError::ShapeMismatch(_))) => Ok(()),
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_zeros() -> Result<()> {
        let tensor = Tensor::zeros(&[2, 3])?;
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.to_vec()?, vec![0.0; 6]);
        Ok(())
    }

    #[test]
    fn test_randn() -> Result<()> {
        let tensor = Tensor::randn(&[2, 3])?;
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.to_vec()?.len(), 6);
        Ok(())
    }

    #[test]
    fn test_reshape() -> Result<()> {
        let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?;
        tensor.reshape(&[2, 2])?;
        assert_eq!(tensor.shape(), &[2, 2]);

        match tensor.reshape(&[3, 3]) {
            Err(MaidenXError::TensorError(TensorError::ShapeMismatch(_))) => Ok(()),
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_split_at() -> Result<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
        let (first, second) = tensor.split_at(1, 2)?;

        assert_eq!(first.shape(), &[2, 2]);
        assert_eq!(second.shape(), &[2, 1]);
        assert_eq!(first.to_vec()?, vec![1.0, 2.0, 4.0, 5.0]);
        assert_eq!(second.to_vec()?, vec![3.0, 6.0]);

        match tensor.split_at(2, 0) {
            Err(MaidenXError::TensorError(TensorError::IndexError(_))) => Ok(()),
            _ => panic!("Expected IndexError"),
        }
    }

    #[test]
    fn test_cat() -> Result<()> {
        let tensor1 = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let tensor2 = Tensor::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]])?;

        let result = Tensor::cat(&[&tensor1, &tensor2], 0)?;
        assert_eq!(result.shape(), &[4, 2]);
        assert_eq!(
            result.to_vec()?,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );

        let result = Tensor::cat(&[&tensor1, &tensor2], 1)?;
        assert_eq!(result.shape(), &[2, 4]);
        assert_eq!(
            result.to_vec()?,
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );

        // Test error cases
        let tensor3 = Tensor::new(vec![vec![1.0]])?;
        match Tensor::cat(&[&tensor1, &tensor3], 0) {
            Err(MaidenXError::TensorError(TensorError::ShapeMismatch(_))) => Ok(()),
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_mul_scalar() -> Result<()> {
        let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let result = tensor.scalar_mul(2.0)?;

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, vec![2.0, 4.0, 6.0, 8.0]);
        Ok(())
    }
}
