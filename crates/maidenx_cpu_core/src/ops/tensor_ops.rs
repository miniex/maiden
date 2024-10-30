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

pub fn mean(a: &[f32]) -> CpuResult<Vec<f32>> {
    if a.is_empty() {
        return Err(CpuError::InvalidValue);
    }
    let sum: f32 = a.iter().sum();
    Ok(vec![sum / a.len() as f32])
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

pub fn sum(a: &[f32]) -> CpuResult<Vec<f32>> {
    if a.is_empty() {
        return Err(CpuError::InvalidValue);
    }
    Ok(vec![a.iter().sum()])
}

pub fn transpose(input: &[f32], rows: usize, cols: usize) -> CpuResult<Vec<f32>> {
    if input.len() != rows * cols {
        return Err(CpuError::InvalidValue);
    }

    let mut output = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            output[j * rows + i] = input[i * cols + j];
        }
    }
    Ok(output)
}
