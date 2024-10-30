use crate::Tensor;
use maidenx_core::error::Result;
use std::ops::{Add, Div, Mul, Sub};

impl Add for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: &Tensor) -> Self::Output {
        if self.shape == [1] {
            let scalar = self.item()?;
            other.scalar_add(scalar)
        } else if other.shape == [1] {
            let scalar = other.item()?;
            self.scalar_add(scalar)
        } else {
            self.add(other)
        }
    }
}
impl Add<f32> for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, scalar: f32) -> Self::Output {
        self.scalar_add(scalar)
    }
}
impl Add<&Tensor> for f32 {
    type Output = Result<Tensor>;

    fn add(self, tensor: &Tensor) -> Self::Output {
        tensor.scalar_add(self)
    }
}

impl Div for &Tensor {
    type Output = Result<Tensor>;

    fn div(self, other: &Tensor) -> Self::Output {
        if self.shape == [1] {
            let scalar = self.item()?;
            other.div(&other.pow(2.0)?)?.scalar_mul(scalar)
        } else if other.shape == [1] {
            let scalar = other.item()?;
            self.scalar_div(scalar)
        } else {
            self.div(other)
        }
    }
}
impl Div<f32> for &Tensor {
    type Output = Result<Tensor>;

    fn div(self, scalar: f32) -> Self::Output {
        self.scalar_div(scalar)
    }
}
impl Div<&Tensor> for f32 {
    type Output = Result<Tensor>;

    fn div(self, tensor: &Tensor) -> Self::Output {
        tensor.div(&tensor.pow(2.0)?)?.scalar_mul(self)
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: &Tensor) -> Self::Output {
        if self.shape == [1] {
            let scalar = self.item()?;
            other.scalar_mul(scalar)
        } else if other.shape == [1] {
            let scalar = other.item()?;
            self.scalar_mul(scalar)
        } else {
            self.mul(other)
        }
    }
}
impl Mul<f32> for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, scalar: f32) -> Self::Output {
        self.scalar_mul(scalar)
    }
}
impl Mul<&Tensor> for f32 {
    type Output = Result<Tensor>;

    fn mul(self, tensor: &Tensor) -> Self::Output {
        tensor.scalar_mul(self)
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: &Tensor) -> Self::Output {
        if self.shape == [1] {
            let scalar = self.item()?;
            other.scalar_mul(-1.0)?.scalar_add(scalar)
        } else if other.shape == [1] {
            let scalar = other.item()?;
            self.scalar_sub(scalar)
        } else {
            self.sub(other)
        }
    }
}
impl Sub<f32> for &Tensor {
    type Output = Result<Tensor>;

    fn sub(self, scalar: f32) -> Self::Output {
        self.scalar_sub(scalar)
    }
}
impl Sub<&Tensor> for f32 {
    type Output = Result<Tensor>;

    fn sub(self, tensor: &Tensor) -> Self::Output {
        tensor.scalar_mul(-1.0)?.scalar_add(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_core::error::Result;

    #[test]
    fn test_tensor_add_tensor() -> Result<()> {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0])?;
        let expected = Tensor::new(vec![5.0, 7.0, 9.0])?;

        let result = (&tensor1 + &tensor2)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_tensor_add_scalar_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar_tensor = Tensor::new(vec![3.0])?;
        let expected = Tensor::new(vec![4.0, 5.0, 6.0])?;

        let result = (&tensor + &scalar_tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_scalar_tensor_add_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar_tensor = Tensor::new(vec![3.0])?;
        let expected = Tensor::new(vec![4.0, 5.0, 6.0])?;

        let result = (&scalar_tensor + &tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_tensor_add_f32() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar = 5.0;
        let expected = Tensor::new(vec![6.0, 7.0, 8.0])?;

        let result = (&tensor + scalar)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_f32_add_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar = 5.0;
        let expected = Tensor::new(vec![6.0, 7.0, 8.0])?;

        let result = (scalar + &tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_div_tensor() -> Result<()> {
        let tensor1 = Tensor::new(vec![8.0, 6.0, 4.0])?;
        let tensor2 = Tensor::new(vec![2.0, 3.0, 4.0])?;
        let expected = Tensor::new(vec![4.0, 2.0, 1.0])?;

        let result = (&tensor1 / &tensor2)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_tensor_div_scalar_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![8.0, 6.0, 4.0])?;
        let scalar_tensor = Tensor::new(vec![2.0])?;
        let expected = Tensor::new(vec![4.0, 3.0, 2.0])?;

        let result = (&tensor / &scalar_tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_scalar_tensor_div_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![8.0, 4.0, 2.0])?;
        let scalar_tensor = Tensor::new(vec![16.0])?;
        let expected = Tensor::new(vec![2.0, 4.0, 8.0])?;

        let result = (&scalar_tensor / &tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_tensor_div_f32() -> Result<()> {
        let tensor = Tensor::new(vec![8.0, 6.0, 4.0])?;
        let scalar = 2.0;
        let expected = Tensor::new(vec![4.0, 3.0, 2.0])?;

        let result = (&tensor / scalar)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_f32_div_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![8.0, 4.0, 2.0])?;
        let scalar = 2.0;
        let expected = Tensor::new(vec![0.25, 0.5, 1.0])?;

        let result = (scalar / &tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_mul_tensor() -> Result<()> {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0])?;
        let expected = Tensor::new(vec![4.0, 10.0, 18.0])?;

        let result = (&tensor1 * &tensor2)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_tensor_mul_scalar_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar_tensor = Tensor::new(vec![3.0])?;
        let expected = Tensor::new(vec![3.0, 6.0, 9.0])?;

        let result = (&tensor * &scalar_tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_scalar_tensor_mul_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar_tensor = Tensor::new(vec![2.0])?;
        let expected = Tensor::new(vec![2.0, 4.0, 6.0])?;

        let result = (&scalar_tensor * &tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_tensor_mul_f32() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar = 3.0;
        let expected = Tensor::new(vec![3.0, 6.0, 9.0])?;

        let result = (&tensor * scalar)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_f32_mul_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar = 2.0;
        let expected = Tensor::new(vec![2.0, 4.0, 6.0])?;

        let result = (scalar * &tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_sub_tensor() -> Result<()> {
        let tensor1 = Tensor::new(vec![5.0, 7.0, 9.0])?;
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0])?;
        let expected = Tensor::new(vec![1.0, 2.0, 3.0])?;

        let result = (&tensor1 - &tensor2)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_tensor_sub_scalar_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![5.0, 7.0, 9.0])?;
        let scalar_tensor = Tensor::new(vec![2.0])?;
        let expected = Tensor::new(vec![3.0, 5.0, 7.0])?;

        let result = (&tensor - &scalar_tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_scalar_tensor_sub_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar_tensor = Tensor::new(vec![5.0])?;
        let expected = Tensor::new(vec![4.0, 3.0, 2.0])?;

        let result = (&scalar_tensor - &tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_tensor_sub_f32() -> Result<()> {
        let tensor = Tensor::new(vec![5.0, 7.0, 9.0])?;
        let scalar = 2.0;
        let expected = Tensor::new(vec![3.0, 5.0, 7.0])?;

        let result = (&tensor - scalar)?;
        assert_eq!(result, expected);

        Ok(())
    }
    #[test]
    fn test_f32_sub_tensor() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0])?;
        let scalar = 5.0;
        let expected = Tensor::new(vec![4.0, 3.0, 2.0])?;

        let result = (scalar - &tensor)?;
        assert_eq!(result, expected);

        Ok(())
    }
}
