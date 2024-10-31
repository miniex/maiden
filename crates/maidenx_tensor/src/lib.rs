mod convert;
mod display;
mod grad;
mod ops;
mod overloading;
mod shape;

use std::cell::RefCell;

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
    grad: RefCell<Option<Box<Tensor>>>,
    requires_grad: bool,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.to_vec().unwrap() == other.to_vec().unwrap()
    }
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
            grad: RefCell::new(None),
            requires_grad: false,
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
            grad: RefCell::new(None),
            requires_grad: false,
        })
    }

    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0f32; size];
        Self::from_vec(data, shape)
    }

    pub fn ones(shape: &[usize]) -> Result<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![1.0f32; size];
        Self::from_vec(data, shape)
    }

    pub fn randn(shape: &[usize]) -> Result<Self> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).map_err(|e| TensorError::DataError(e.to_string()))?;
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
        Self::from_vec(data, shape)
    }

    pub fn linspace(start: f32, end: f32, steps: usize) -> Result<Self> {
        if steps == 0 {
            return Err(TensorError::InvalidOperation(
                "Number of steps must be greater than zero".to_string(),
            )
            .into());
        }

        let step_size = (end - start) / (steps as f32 - 1.0);
        let data: Vec<f32> = (0..steps).map(|i| start + i as f32 * step_size).collect();

        Self::from_vec(data, &[steps])
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

    pub fn item(&self) -> Result<f32> {
        let data = self.to_vec()?;
        if data.is_empty() {
            return Err(
                TensorError::InvalidOperation("Cannot call item() on empty tensor".into()).into(),
            );
        }
        Ok(data[0])
    }

    pub fn item_at(&self, index: usize) -> Result<f32> {
        let data = self.to_vec()?;
        if index >= data.len() {
            return Err(TensorError::IndexError(format!(
                "Index {} is out of bounds for tensor with {} elements",
                index,
                data.len()
            ))
            .into());
        }
        Ok(data[index])
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
    fn test_item() -> Result<()> {
        let tensor = Tensor::new(vec![42.0f32])?;
        assert_eq!(tensor.item()?, 42.0);

        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        assert_eq!(tensor.item()?, 1.0);

        Ok(())
    }

    #[test]
    fn test_item_at() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        assert_eq!(tensor.item_at(0)?, 1.0);
        assert_eq!(tensor.item_at(1)?, 2.0);
        assert_eq!(tensor.item_at(2)?, 3.0);

        match tensor.item_at(3) {
            Err(MaidenXError::TensorError(TensorError::IndexError(_))) => Ok(()),
            _ => panic!("Expected IndexError"),
        }
    }
}
