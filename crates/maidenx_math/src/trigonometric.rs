use maidenx_core::{
    device::Device,
    error::{MaidenXError, Result},
};
#[cfg(feature = "cuda")]
use maidenx_cuda_kernels::tensor_ops_trigonometric::{
    cuda_tensor_acos, cuda_tensor_asin, cuda_tensor_atan, cuda_tensor_cos, cuda_tensor_sin,
    cuda_tensor_tan,
};
use maidenx_tensor::Tensor;

pub fn sin(tensor: &Tensor) -> Result<Tensor> {
    let device = maidenx_core::device::get_current_device();
    let mut result = Tensor::zeros(tensor.shape())?;

    match &device {
        Device::Cpu => {
            let data = tensor.to_vec()?;
            let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::sin(&data)?;
            result.data_mut().copy_from_host(&result_data)?;
        }
        Device::Cuda(_) => {
            #[cfg(feature = "cuda")]
            unsafe {
                cuda_tensor_sin(
                    result.data_mut().as_mut_ptr(),
                    tensor.data().as_ptr(),
                    tensor.size(),
                )
                .map_err(MaidenXError::from)?;
            }
            #[cfg(not(feature = "cuda"))]
            return Err(MaidenXError::UnsupportedOperation(
                "CUDA operations are not available - feature not enabled".into(),
            ));
        }
    }

    Ok(result)
}

pub fn cos(tensor: &Tensor) -> Result<Tensor> {
    let device = maidenx_core::device::get_current_device();
    let mut result = Tensor::zeros(tensor.shape())?;

    match &device {
        Device::Cpu => {
            let data = tensor.to_vec()?;
            let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::cos(&data)?;
            result.data_mut().copy_from_host(&result_data)?;
        }
        Device::Cuda(_) => {
            #[cfg(feature = "cuda")]
            unsafe {
                cuda_tensor_cos(
                    result.data_mut().as_mut_ptr(),
                    tensor.data().as_ptr(),
                    tensor.size(),
                )
                .map_err(MaidenXError::from)?;
            }
            #[cfg(not(feature = "cuda"))]
            return Err(MaidenXError::UnsupportedOperation(
                "CUDA operations are not available - feature not enabled".into(),
            ));
        }
    }

    Ok(result)
}

pub fn tan(tensor: &Tensor) -> Result<Tensor> {
    let device = maidenx_core::device::get_current_device();
    let mut result = Tensor::zeros(tensor.shape())?;

    match &device {
        Device::Cpu => {
            let data = tensor.to_vec()?;
            let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::tan(&data)?;
            result.data_mut().copy_from_host(&result_data)?;
        }
        Device::Cuda(_) => {
            #[cfg(feature = "cuda")]
            unsafe {
                cuda_tensor_tan(
                    result.data_mut().as_mut_ptr(),
                    tensor.data().as_ptr(),
                    tensor.size(),
                )
                .map_err(MaidenXError::from)?;
            }
            #[cfg(not(feature = "cuda"))]
            return Err(MaidenXError::UnsupportedOperation(
                "CUDA operations are not available - feature not enabled".into(),
            ));
        }
    }

    Ok(result)
}

pub fn asin(tensor: &Tensor) -> Result<Tensor> {
    let device = maidenx_core::device::get_current_device();
    let mut result = Tensor::zeros(tensor.shape())?;

    match &device {
        Device::Cpu => {
            let data = tensor.to_vec()?;
            let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::asin(&data)?;
            result.data_mut().copy_from_host(&result_data)?;
        }
        Device::Cuda(_) => {
            #[cfg(feature = "cuda")]
            unsafe {
                cuda_tensor_asin(
                    result.data_mut().as_mut_ptr(),
                    tensor.data().as_ptr(),
                    tensor.size(),
                )
                .map_err(MaidenXError::from)?;
            }
            #[cfg(not(feature = "cuda"))]
            return Err(MaidenXError::UnsupportedOperation(
                "CUDA operations are not available - feature not enabled".into(),
            ));
        }
    }

    Ok(result)
}

pub fn acos(tensor: &Tensor) -> Result<Tensor> {
    let device = maidenx_core::device::get_current_device();
    let mut result = Tensor::zeros(tensor.shape())?;

    match &device {
        Device::Cpu => {
            let data = tensor.to_vec()?;
            let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::acos(&data)?;
            result.data_mut().copy_from_host(&result_data)?;
        }
        Device::Cuda(_) => {
            #[cfg(feature = "cuda")]
            unsafe {
                cuda_tensor_acos(
                    result.data_mut().as_mut_ptr(),
                    tensor.data().as_ptr(),
                    tensor.size(),
                )
                .map_err(MaidenXError::from)?;
            }
            #[cfg(not(feature = "cuda"))]
            return Err(MaidenXError::UnsupportedOperation(
                "CUDA operations are not available - feature not enabled".into(),
            ));
        }
    }

    Ok(result)
}

pub fn atan(tensor: &Tensor) -> Result<Tensor> {
    let device = maidenx_core::device::get_current_device();
    let mut result = Tensor::zeros(tensor.shape())?;

    match &device {
        Device::Cpu => {
            let data = tensor.to_vec()?;
            let result_data = maidenx_cpu_core::ops::tensor_ops_trigonometric::atan(&data)?;
            result.data_mut().copy_from_host(&result_data)?;
        }
        Device::Cuda(_) => {
            #[cfg(feature = "cuda")]
            unsafe {
                cuda_tensor_atan(
                    result.data_mut().as_mut_ptr(),
                    tensor.data().as_ptr(),
                    tensor.size(),
                )
                .map_err(MaidenXError::from)?;
            }
            #[cfg(not(feature = "cuda"))]
            return Err(MaidenXError::UnsupportedOperation(
                "CUDA operations are not available - feature not enabled".into(),
            ));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_sin() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, PI / 2.0, PI])?;
        let result = sin(&tensor)?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[3]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // sin(0) = 0
        assert!((result_data[1] - 1.0).abs() < 1e-5); // sin(π/2) = 1
        assert!((result_data[2] - 0.0).abs() < 1e-5); // sin(π) = 0

        Ok(())
    }

    #[test]
    fn test_cos() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, PI / 2.0, PI])?;
        let result = cos(&tensor)?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[3]);
        assert!((result_data[0] - 1.0).abs() < 1e-5); // cos(0) = 1
        assert!(result_data[1].abs() < 1e-5); // cos(π/2) = 0
        assert!((result_data[2] + 1.0).abs() < 1e-5); // cos(π) = -1

        Ok(())
    }

    #[test]
    fn test_tan() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, PI / 4.0])?;
        let result = tan(&tensor)?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[2]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // tan(0) = 0
        assert!((result_data[1] - 1.0).abs() < 1e-5); // tan(π/4) = 1

        Ok(())
    }

    #[test]
    fn test_asin() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, 1.0, -1.0, 0.5])?;
        let result = asin(&tensor)?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[4]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // asin(0) = 0
        assert!((result_data[1] - PI / 2.0).abs() < 1e-5); // asin(1) = π/2
        assert!((result_data[2] + PI / 2.0).abs() < 1e-5); // asin(-1) = -π/2
        assert!((result_data[3] - (0.5f32.asin())).abs() < 1e-5); // asin(0.5)

        Ok(())
    }

    #[test]
    fn test_acos() -> Result<()> {
        let tensor = Tensor::new(vec![1.0, 0.0, -1.0, 0.5])?;
        let result = acos(&tensor)?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[4]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // acos(1) = 0
        assert!((result_data[1] - PI / 2.0).abs() < 1e-5); // acos(0) = π/2
        assert!((result_data[2] - PI).abs() < 1e-5); // acos(-1) = π
        assert!((result_data[3] - (0.5f32.acos())).abs() < 1e-5); // acos(0.5)

        Ok(())
    }

    #[test]
    fn test_atan() -> Result<()> {
        let tensor = Tensor::new(vec![0.0, 1.0, -1.0])?;
        let result = atan(&tensor)?;

        let result_data = result.to_vec()?;
        assert_eq!(result.shape(), &[3]);
        assert!((result_data[0] - 0.0).abs() < 1e-5); // atan(0) = 0
        assert!((result_data[1] - PI / 4.0).abs() < 1e-5); // atan(1) = π/4
        assert!((result_data[2] + PI / 4.0).abs() < 1e-5); // atan(-1) = -π/4

        Ok(())
    }
}
