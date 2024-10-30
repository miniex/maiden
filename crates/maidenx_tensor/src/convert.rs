use crate::Tensor;
use maidenx_core::error::MaidenXError;
use maidenx_core::error::Result;

pub trait TensorData {
    fn to_flat_vec(self) -> Vec<f32>;
    fn to_shape(&self) -> Vec<usize>;
}

impl Tensor {
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let num_elements = self.shape.iter().product::<usize>();
        let mut result = vec![0.0f32; num_elements];
        self.buffer
            .copy_to_host(&mut result)
            .map_err(MaidenXError::from)?;
        Ok(result)
    }
}

impl TensorData for Vec<f32> {
    fn to_flat_vec(self) -> Vec<f32> {
        self
    }

    fn to_shape(&self) -> Vec<usize> {
        vec![self.len()]
    }
}

impl TensorData for Vec<Vec<f32>> {
    fn to_flat_vec(self) -> Vec<f32> {
        self.into_iter()
            .flat_map(|inner| inner.into_iter())
            .collect()
    }

    fn to_shape(&self) -> Vec<usize> {
        if self.is_empty() {
            vec![0, 0]
        } else {
            vec![self.len(), self[0].len()]
        }
    }
}

impl TensorData for Vec<Vec<Vec<f32>>> {
    fn to_flat_vec(self) -> Vec<f32> {
        self.into_iter()
            .flat_map(|matrix| matrix.into_iter().flat_map(|row| row.into_iter()))
            .collect()
    }

    fn to_shape(&self) -> Vec<usize> {
        if self.is_empty() {
            vec![0, 0, 0]
        } else if self[0].is_empty() {
            vec![self.len(), 0, 0]
        } else {
            vec![self.len(), self[0].len(), self[0][0].len()]
        }
    }
}

impl TensorData for Vec<Vec<Vec<Vec<f32>>>> {
    fn to_flat_vec(self) -> Vec<f32> {
        self.into_iter()
            .flat_map(|cube| {
                cube.into_iter()
                    .flat_map(|matrix| matrix.into_iter().flat_map(|row| row.into_iter()))
            })
            .collect()
    }

    fn to_shape(&self) -> Vec<usize> {
        if self.is_empty() {
            vec![0, 0, 0, 0]
        } else if self[0].is_empty() {
            vec![self.len(), 0, 0, 0]
        } else if self[0][0].is_empty() {
            vec![self.len(), self[0].len(), 0, 0]
        } else {
            vec![
                self.len(),
                self[0].len(),
                self[0][0].len(),
                self[0][0][0].len(),
            ]
        }
    }
}
