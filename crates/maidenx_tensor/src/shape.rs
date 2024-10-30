use crate::Tensor;
use maidenx_core::error::{Result, TensorError};

impl Tensor {
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

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}
