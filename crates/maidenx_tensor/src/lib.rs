mod convert;

use convert::TensorData;
use maidenx_cuda_core::prelude::{CudaBuffer, CudaError, CudaResult};
use maidenx_cuda_kernels::tensor_ops::{cuda_tensor_add, cuda_tensor_matmul, cuda_tensor_mul};
use rand::prelude::*;
use rand_distr::Normal;
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Tensor {
    buffer: Arc<CudaBuffer>,
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
            1 => display_1d(f, &data, &self.shape),
            2 => display_2d(f, &data, &self.shape),
            3 => display_3d(f, &data, &self.shape),
            4 => display_4d(f, &data, &self.shape),
            _ => display_nd(f, &data, &self.shape),
        }
    }
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

    pub fn from_vec<T: Into<Vec<f32>>>(vec: T, shape: &[usize]) -> CudaResult<Self> {
        let vec = vec.into();
        let total_size: usize = shape.iter().product();

        if vec.len() != total_size {
            return Err(CudaError::ShapeMismatch);
        }

        let size = total_size * std::mem::size_of::<f32>();
        let buffer = Arc::new(CudaBuffer::new(size)?);
        let strides = Self::compute_strides(shape);

        let mut tensor = Self {
            buffer,
            shape: shape.to_vec(),
            strides,
        };

        if let Some(buffer) = Arc::get_mut(&mut tensor.buffer) {
            buffer.copy_from_host(&vec)?;
        } else {
            return Err(CudaError::AllocationFailed);
        }

        Ok(tensor)
    }

    pub fn data(&self) -> &CudaBuffer {
        &self.buffer
    }

    pub fn data_mut(&mut self) -> &mut CudaBuffer {
        if Arc::strong_count(&self.buffer) == 1 {
            Arc::get_mut(&mut self.buffer).unwrap()
        } else {
            let new_buffer = self.buffer.as_ref().clone();
            self.buffer = Arc::new(new_buffer);
            Arc::get_mut(&mut self.buffer).unwrap()
        }
    }

    pub fn zeros(shape: &[usize]) -> CudaResult<Self> {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0f32; size];
        Self::from_vec(data, shape)
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

    pub fn reshape(&mut self, new_shape: &[usize]) -> CudaResult<()> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(CudaError::ShapeMismatch);
        }

        self.shape = new_shape.to_vec();
        self.strides = Self::compute_strides(&self.shape);
        Ok(())
    }

    pub fn randn(shape: &[usize]) -> CudaResult<Self> {
        let size = shape.iter().product::<usize>();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
        Self::from_vec(data, shape)
    }

    pub fn mul_scalar(&self, scalar: f32) -> CudaResult<Self> {
        let mut result = Self {
            buffer: Arc::new(CudaBuffer::new(self.buffer.len())?),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        };

        let scaled_data: Vec<f32> = self.to_vec()?.iter().map(|x| x * scalar).collect();

        if let Some(buffer) = Arc::get_mut(&mut result.buffer) {
            buffer.copy_from_host(&scaled_data)?;
        } else {
            return Err(CudaError::AllocationFailed);
        }

        Ok(result)
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

    pub fn matmul(&self, other: &Tensor) -> CudaResult<Tensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(CudaError::ShapeMismatch);
        }

        let m = self.shape[0];
        let k = self.shape[1];

        if k != other.shape[0] {
            return Err(CudaError::ShapeMismatch);
        }
        let n = other.shape[1];

        let result_shape = vec![m, n];
        let result_strides = Self::compute_strides(&result_shape);
        let result_size = m * n * std::mem::size_of::<f32>();

        let mut result = Self {
            buffer: Arc::new(CudaBuffer::new(result_size)?),
            shape: result_shape,
            strides: result_strides,
        };

        unsafe {
            cuda_tensor_matmul(
                Arc::get_mut(&mut result.buffer).unwrap().as_mut_ptr(),
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                m as i32,
                n as i32,
                k as i32,
            )?;
        }

        Ok(result)
    }
}

// Display

fn display_1d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    write!(f, "Tensor(shape=[{}], data=", shape[0])?;
    write!(f, "[")?;
    for (i, val) in data.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?
        }
        write!(f, "{:.4}", val)?;
    }
    write!(f, "])")
}

fn display_2d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    writeln!(f, "Tensor(shape=[{}, {}], data=", shape[0], shape[1])?;
    for i in 0..shape[0] {
        if i == 0 {
            write!(f, "[")?;
        }
        write!(f, "[")?;
        for j in 0..shape[1] {
            if j > 0 {
                write!(f, ", ")?
            }
            write!(f, "{:.4}", data[i * shape[1] + j])?;
        }
        if i == shape[0] - 1 {
            writeln!(f, "]])")?;
        } else {
            writeln!(f, "]")?;
        }
    }
    Ok(())
}

fn display_3d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    writeln!(
        f,
        "Tensor(shape=[{}, {}, {}], data=",
        shape[0], shape[1], shape[2]
    )?;
    let plane_size = shape[1] * shape[2];

    write!(f, "[")?;
    for i in 0..shape[0] {
        if i > 0 {
            writeln!(f)?;
        }
        write!(f, "[")?;
        for j in 0..shape[1] {
            if j > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for k in 0..shape[2] {
                if k > 0 {
                    write!(f, ", ")?
                }
                write!(f, "{:.4}", data[i * plane_size + j * shape[2] + k])?;
            }
            write!(f, "]")?;
        }
        if i == shape[0] - 1 {
            write!(f, "]])")?;
        } else {
            write!(f, "]")?;
        }
    }
    Ok(())
}

fn display_4d(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    writeln!(
        f,
        "Tensor(shape=[{}, {}, {}, {}], data=",
        shape[0], shape[1], shape[2], shape[3]
    )?;
    let volume_size = shape[1] * shape[2] * shape[3];
    let plane_size = shape[2] * shape[3];

    write!(f, "[")?;
    for w in 0..shape[0] {
        if w > 0 {
            writeln!(f, ",")?;
        }
        write!(f, "[")?;
        for x in 0..shape[1] {
            if x > 0 {
                writeln!(f)?;
            }
            write!(f, "[")?;
            for y in 0..shape[2] {
                if y > 0 {
                    writeln!(f)?;
                }
                write!(f, "[")?;
                for z in 0..shape[3] {
                    if z > 0 {
                        write!(f, ", ")?
                    }
                    write!(
                        f,
                        "{:.4}",
                        data[w * volume_size + x * plane_size + y * shape[3] + z]
                    )?;
                }
                write!(f, "]")?;
            }
            write!(f, "]")?;
        }
        if w == shape[0] - 1 {
            write!(f, "]])")?;
        } else {
            write!(f, "]")?;
        }
    }
    Ok(())
}

fn display_nd(f: &mut fmt::Formatter<'_>, data: &[f32], shape: &[usize]) -> fmt::Result {
    write!(f, "Tensor(shape=[")?;
    for (i, &dim) in shape.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?
        }
        write!(f, "{}", dim)?;
    }
    write!(f, "], data=[")?;
    for (i, val) in data.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?
        }
        if i >= 8 {
            write!(f, "...")?;
            break;
        }
        write!(f, "{:.4}", val)?;
    }
    write!(f, "])")
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

    #[test]
    fn test_tensor_add() -> CudaResult<()> {
        let tensor1 = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
        let tensor2 = Tensor::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]])?;

        let result = tensor1.add(&tensor2)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, [6.0, 8.0, 10.0, 12.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_mul() -> CudaResult<()> {
        let tensor1 = Tensor::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]])?;
        let tensor2 = Tensor::new(vec![vec![3.0, 2.0], vec![1.0, 4.0]])?;

        let result = tensor1.mul(&tensor2)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec()?, [6.0, 6.0, 4.0, 20.0]);
        Ok(())
    }

    #[test]
    fn test_matrix_multiplication() -> CudaResult<()> {
        let data1 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tensor1 = Tensor::new(data1)?;

        let data2 = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let tensor2 = Tensor::new(data2)?;

        let result = tensor1.matmul(&tensor2)?;

        // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
        // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
        assert_eq!(result.shape(), &[2, 2]);

        let result_data = result.to_vec()?;
        let expected = [
            1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0,
            1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0,
            4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0,
            4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0,
        ];

        assert_eq!(result_data.len(), expected.len());
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
    fn test_matrix_multiplication_invalid_shape() -> CudaResult<()> {
        let data1 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tensor1 = Tensor::new(data1)?;

        let data2 = vec![vec![7.0, 8.0], vec![9.0, 10.0]];
        let tensor2 = Tensor::new(data2)?;

        assert!(matches!(
            tensor1.matmul(&tensor2),
            Err(CudaError::ShapeMismatch)
        ));

        Ok(())
    }

    #[test]
    fn test_large_matrix_multiplication() -> CudaResult<()> {
        let size = 32;

        let data1 = vec![vec![1.0; size]; size];
        let data2 = vec![vec![2.0; size]; size];

        let tensor1 = Tensor::new(data1)?;
        let tensor2 = Tensor::new(data2)?;

        let result = tensor1.matmul(&tensor2)?;

        assert_eq!(result.shape(), &[size, size]);

        let result_data = result.to_vec()?;
        let expected_value = 2.0 * size as f32;

        for val in result_data {
            assert!(
                (val - expected_value).abs() < 1e-3,
                "Expected {}, got {}",
                expected_value,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_from_vec() -> CudaResult<()> {
        // 1D tensor
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.to_vec()?, vec![1.0, 2.0, 3.0]);

        // 2D tensor
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);

        // Shape mismatch should fail
        assert!(matches!(
            Tensor::from_vec(vec![1.0, 2.0], &[3]),
            Err(CudaError::ShapeMismatch)
        ));

        Ok(())
    }

    #[test]
    fn test_from_vec_multi_dimensional() -> CudaResult<()> {
        // 3D tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, &[2, 2, 2])?;
        assert_eq!(tensor.shape(), &[2, 2, 2]);

        let result = tensor.to_vec()?;
        assert_eq!(result.len(), 8);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        Ok(())
    }

    #[test]
    fn test_randn() -> CudaResult<()> {
        let shape = &[3, 3];
        let tensor = Tensor::randn(shape)?;

        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.ndim(), shape.len());

        assert_eq!(tensor.to_vec()?.len(), shape.iter().product());

        Ok(())
    }

    #[test]
    fn test_mul_scalar() -> CudaResult<()> {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let scalar = 2.0;
        let result = tensor.mul_scalar(scalar)?;

        let expected = vec![2.0, 4.0, 6.0, 8.0];
        assert_eq!(result.to_vec()?, expected);

        Ok(())
    }
}
