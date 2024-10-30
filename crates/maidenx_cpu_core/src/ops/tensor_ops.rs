use super::super::error::{CpuError, CpuResult};

pub fn add(a: &[f32], b: &[f32]) -> CpuResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(CpuError::InvalidValue);
    }

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

pub fn div(a: &[f32], b: &[f32]) -> CpuResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(CpuError::InvalidValue);
    }

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x / y).collect())
}

pub fn mat_mul(a: &[f32], a_shape: &[usize], b: &[f32], b_shape: &[usize]) -> CpuResult<Vec<f32>> {
    if a_shape[1] != b_shape[0] {
        return Err(CpuError::InvalidValue);
    }

    let (m, k) = (a_shape[0], a_shape[1]);
    let n = b_shape[1];
    let mut result = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    Ok(result)
}

pub fn mul(a: &[f32], b: &[f32]) -> CpuResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(CpuError::InvalidValue);
    }

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

pub fn pow(a: &[f32], exponent: f32) -> CpuResult<Vec<f32>> {
    Ok(a.iter().map(|x| x.powf(exponent)).collect())
}

pub fn scalar_mul(a: &[f32], scalar: f32) -> CpuResult<Vec<f32>> {
    Ok(a.iter().map(|x| x * scalar).collect())
}
