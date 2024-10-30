use super::super::error::{CpuError, CpuResult};

pub fn sin(a: &[f32]) -> CpuResult<Vec<f32>> {
    if a.is_empty() {
        return Err(CpuError::InvalidValue);
    }
    Ok(a.iter().map(|x| x.sin()).collect())
}

pub fn cos(a: &[f32]) -> CpuResult<Vec<f32>> {
    if a.is_empty() {
        return Err(CpuError::InvalidValue);
    }
    Ok(a.iter().map(|x| x.cos()).collect())
}

pub fn tan(a: &[f32]) -> CpuResult<Vec<f32>> {
    if a.is_empty() {
        return Err(CpuError::InvalidValue);
    }
    Ok(a.iter().map(|x| x.tan()).collect())
}

pub fn asin(a: &[f32]) -> CpuResult<Vec<f32>> {
    if a.is_empty() {
        return Err(CpuError::InvalidValue);
    }
    if a.iter().any(|&x| !(-1.0..=1.0).contains(&x)) {
        return Err(CpuError::InvalidValue);
    }
    Ok(a.iter().map(|x| x.asin()).collect())
}

pub fn acos(a: &[f32]) -> CpuResult<Vec<f32>> {
    if a.is_empty() {
        return Err(CpuError::InvalidValue);
    }
    if a.iter().any(|&x| !(-1.0..=1.0).contains(&x)) {
        return Err(CpuError::InvalidValue);
    }
    Ok(a.iter().map(|x| x.acos()).collect())
}

pub fn atan(a: &[f32]) -> CpuResult<Vec<f32>> {
    if a.is_empty() {
        return Err(CpuError::InvalidValue);
    }
    Ok(a.iter().map(|x| x.atan()).collect())
}
