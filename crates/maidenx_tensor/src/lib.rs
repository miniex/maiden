mod convert;
mod display;

use convert::TensorData;
use maidenx_core::buffer::DeviceBuffer;
use maidenx_core::device::Device;
use maidenx_core::error::{MaidenXError, Result, TensorError};
use maidenx_cuda_kernels::tensor_ops::{cuda_tensor_add, cuda_tensor_matmul, cuda_tensor_mul};
use rand::prelude::*;
use rand_distr::Normal;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Tensor {
    buffer: DeviceBuffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = match self.to_vec() {
            Ok(data) => data,
            Err(_) => return write!(f, "Tensor(Failed to fetch data)"),
        };

        match self.shape.len() {
            1 => display::display_1d(f, &data, &self.shape),
            2 => display::display_2d(f, &data, &self.shape),
            3 => display::display_3d(f, &data, &self.shape),
            4 => display::display_4d(f, &data, &self.shape),
            _ => display::display_nd(f, &data, &self.shape),
        }
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

    // OPS

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
            Device::Cuda(_) => unsafe {
                cuda_tensor_add(
                    result.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    other.buffer.as_ptr(),
                    self.size(),
                )
                .map_err(MaidenXError::from)?;
            },
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
            Device::Cuda(_) => unsafe {
                cuda_tensor_mul(
                    result.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    other.buffer.as_ptr(),
                    self.size(),
                )
                .map_err(MaidenXError::from)?;
            },
        }

        Ok(result)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
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
                    maidenx_cpu_core::ops::tensor_ops::matmul(&a, &self.shape, &b, &other.shape)?;
                result.buffer.copy_from_host(&result_data)?;
            }
            Device::Cuda(_) => unsafe {
                cuda_tensor_matmul(
                    result.buffer.as_mut_ptr(),
                    self.buffer.as_ptr(),
                    other.buffer.as_ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                )
                .map_err(MaidenXError::from)?;
            },
        }

        Ok(result)
    }

    pub fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        let device = maidenx_core::device::get_current_device();
        let mut result = Self {
            buffer: DeviceBuffer::new(self.buffer.len(), &device)?,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        };

        let data = self
            .to_vec()?
            .iter()
            .map(|x| x * scalar)
            .collect::<Vec<f32>>();

        result.buffer.copy_from_host(&data)?;
        Ok(result)
    }

    // Shape

    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<()> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot reshape tensor of size {} into shape {:?}",
                self.size(),
                new_shape
            ))
            .into());
        }

        self.shape = new_shape.to_vec();
        self.strides = Self::compute_strides(&self.shape);
        Ok(())
    }

    pub fn split_at(&self, dim: usize, index: usize) -> Result<(Tensor, Tensor)> {
        if dim >= self.shape.len() {
            return Err(TensorError::IndexError(format!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim,
                self.shape.len()
            ))
            .into());
        }
        if index >= self.shape[dim] {
            return Err(TensorError::IndexError(format!(
                "Split index {} is out of bounds for dimension {} with size {}",
                index, dim, self.shape[dim]
            ))
            .into());
        }

        let mut first_shape = self.shape.clone();
        let mut second_shape = self.shape.clone();
        first_shape[dim] = index;
        second_shape[dim] = self.shape[dim] - index;

        let data = self.to_vec()?;
        let mut first_data = Vec::new();
        let mut second_data = Vec::new();

        if dim == 0 {
            let split_point = index * self.strides[0];
            first_data = data[..split_point].to_vec();
            second_data = data[split_point..].to_vec();
        } else if dim == 1 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            for i in 0..rows {
                let row_start = i * cols;
                first_data.extend_from_slice(&data[row_start..row_start + index]);
                second_data.extend_from_slice(&data[row_start + index..row_start + cols]);
            }
        }

        Ok((
            Tensor::from_vec(first_data, &first_shape)?,
            Tensor::from_vec(second_data, &second_shape)?,
        ))
    }

    pub fn cat(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TensorError::InvalidOperation("Empty tensor list".into()).into());
        }

        let first = &tensors[0];
        let mut target_shape = first.shape.clone();

        // Validate shapes
        for tensor in tensors.iter().skip(1) {
            if tensor.shape.len() != first.shape.len() {
                return Err(TensorError::ShapeMismatch(
                    "All tensors must have the same number of dimensions".into(),
                )
                .into());
            }
            for (i, (&s1, &s2)) in first.shape.iter().zip(tensor.shape.iter()).enumerate() {
                if i != dim && s1 != s2 {
                    return Err(TensorError::ShapeMismatch(format!(
                        "Incompatible shapes for concatenation: {:?} and {:?}",
                        first.shape, tensor.shape
                    ))
                    .into());
                }
            }
        }

        target_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();

        if dim == 0 {
            let mut result_data = Vec::new();
            for tensor in tensors {
                result_data.extend(tensor.to_vec()?);
            }
            Tensor::from_vec(result_data, &target_shape)
        } else if dim == 1 {
            let rows = target_shape[0];
            let mut result_data = Vec::new();
            for row in 0..rows {
                for tensor in tensors {
                    let data = tensor.to_vec()?;
                    let row_start = row * tensor.shape[1];
                    let row_end = row_start + tensor.shape[1];
                    result_data.extend_from_slice(&data[row_start..row_end]);
                }
            }
            Tensor::from_vec(result_data, &target_shape)
        } else {
            Err(TensorError::InvalidOperation(format!(
                "Concatenation along dimension {} is not supported",
                dim
            ))
            .into())
        }
    }

    // Utility methods
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

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let num_elements = self.shape.iter().product::<usize>();
        let mut result = vec![0.0f32; num_elements];
        self.buffer
            .copy_to_host(&mut result)
            .map_err(MaidenXError::from)?;
        Ok(result)
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let result = tensor1.matmul(&tensor2)?;
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

        match tensor1.matmul(&tensor2) {
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
        let result = tensor.mul_scalar(2.0)?;

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, vec![2.0, 4.0, 6.0, 8.0]);
        Ok(())
    }
}
