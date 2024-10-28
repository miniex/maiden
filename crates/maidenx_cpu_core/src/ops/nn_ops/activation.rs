use crate::error::CpuResult;

pub fn relu_forward(src: &[f32]) -> CpuResult<Vec<f32>> {
    Ok(src.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect())
}

pub fn sigmoid_forward(src: &[f32]) -> CpuResult<Vec<f32>> {
    Ok(src.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
}

pub fn tanh_forward(src: &[f32]) -> CpuResult<Vec<f32>> {
    Ok(src.iter().map(|&x| x.tanh()).collect())
}
