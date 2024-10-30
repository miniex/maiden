use maidenx_cuda_core::prelude::CudaResult;

#[link(name = "tensor_ops")]
extern "C" {
    fn tensor_sin(output: *mut f32, input: *const f32, size: usize);
    fn tensor_cos(output: *mut f32, input: *const f32, size: usize);
    fn tensor_tan(output: *mut f32, input: *const f32, size: usize);
    fn tensor_asin(output: *mut f32, input: *const f32, size: usize);
    fn tensor_acos(output: *mut f32, input: *const f32, size: usize);
    fn tensor_atan(output: *mut f32, input: *const f32, size: usize);
}

/// Performs element-wise sine operation on a tensor on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input` - Pointer to the input buffer on CUDA device
/// * `size` - Number of elements to process
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for `size` elements
/// * `input` buffer contains at least `size` elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
pub unsafe fn cuda_tensor_sin(output: *mut f32, input: *const f32, size: usize) -> CudaResult<()> {
    tensor_sin(output, input, size);
    Ok(())
}

/// Performs element-wise cosine operation on a tensor on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input` - Pointer to the input buffer on CUDA device
/// * `size` - Number of elements to process
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for `size` elements
/// * `input` buffer contains at least `size` elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
pub unsafe fn cuda_tensor_cos(output: *mut f32, input: *const f32, size: usize) -> CudaResult<()> {
    tensor_cos(output, input, size);
    Ok(())
}

/// Performs element-wise tangent operation on a tensor on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input` - Pointer to the input buffer on CUDA device
/// * `size` - Number of elements to process
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for `size` elements
/// * `input` buffer contains at least `size` elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
pub unsafe fn cuda_tensor_tan(output: *mut f32, input: *const f32, size: usize) -> CudaResult<()> {
    tensor_tan(output, input, size);
    Ok(())
}

/// Performs element-wise arcsine operation on a tensor on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input` - Pointer to the input buffer on CUDA device
/// * `size` - Number of elements to process
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for `size` elements
/// * `input` buffer contains at least `size` elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
/// * Input values must be in the range [-1, 1]
pub unsafe fn cuda_tensor_asin(output: *mut f32, input: *const f32, size: usize) -> CudaResult<()> {
    tensor_asin(output, input, size);
    Ok(())
}

/// Performs element-wise arccosine operation on a tensor on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input` - Pointer to the input buffer on CUDA device
/// * `size` - Number of elements to process
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for `size` elements
/// * `input` buffer contains at least `size` elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
/// * Input values must be in the range [-1, 1]
pub unsafe fn cuda_tensor_acos(output: *mut f32, input: *const f32, size: usize) -> CudaResult<()> {
    tensor_acos(output, input, size);
    Ok(())
}

/// Performs element-wise arctangent operation on a tensor on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input` - Pointer to the input buffer on CUDA device
/// * `size` - Number of elements to process
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for `size` elements
/// * `input` buffer contains at least `size` elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
pub unsafe fn cuda_tensor_atan(output: *mut f32, input: *const f32, size: usize) -> CudaResult<()> {
    tensor_atan(output, input, size);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use maidenx_cuda_core::prelude::CudaBuffer;
    use std::f32::consts::PI;

    #[test]
    fn test_tensor_sin() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input_data = vec![0.0; size]; // sin(0) = 0
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cuda_tensor_sin(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                0.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_tensor_cos() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input_data = vec![0.0; size]; // cos(0) = 1
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cuda_tensor_cos(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                1.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_tensor_tan() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input_data = vec![PI / 4.0; size]; // tan(π/4) = 1
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cuda_tensor_tan(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                1.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_tensor_asin() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input_data = vec![1.0; size]; // asin(1) = π/2
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cuda_tensor_asin(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - PI / 2.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                PI / 2.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_tensor_acos() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input_data = vec![1.0; size]; // acos(1) = 0
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cuda_tensor_acos(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                0.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_tensor_atan() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input_data = vec![1.0; size]; // atan(1) = π/4
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            cuda_tensor_atan(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - PI / 4.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                PI / 4.0,
                val
            );
        }

        Ok(())
    }
}
