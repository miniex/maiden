use cuda_core::prelude::CudaResult;

#[link(name = "tensor_ops")]
extern "C" {
    fn tensor_add(output: *mut f32, input1: *const f32, input2: *const f32, size: usize);
    fn tensor_mul(output: *mut f32, input1: *const f32, input2: *const f32, size: usize);
}

/// Performs element-wise addition of two tensors on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input1` - Pointer to the first input buffer on CUDA device
/// * `input2` - Pointer to the second input buffer on CUDA device
/// * `size` - Number of elements to process
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for `size` elements
/// * `input1` and `input2` buffers contain at least `size` elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
pub unsafe fn cuda_tensor_add(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
) -> CudaResult<()> {
    tensor_add(output, input1, input2, size);
    // TODO add CUDA Error check
    Ok(())
}

/// Performs element-wise multiplication of two tensors on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input1` - Pointer to the first input buffer on CUDA device
/// * `input2` - Pointer to the second input buffer on CUDA device
/// * `size` - Number of elements to process
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for `size` elements
/// * `input1` and `input2` buffers contain at least `size` elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
pub unsafe fn cuda_tensor_mul(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    size: usize,
) -> CudaResult<()> {
    tensor_mul(output, input1, input2, size);
    // TODO add CUDA Error check
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_core::prelude::CudaBuffer;

    #[test]
    fn test_tensor_add() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let input1_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let input2_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        unsafe {
            cuda_tensor_add(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                size,
            )?;
        }

        Ok(())
    }

    #[test]
    fn test_tensor_mul() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let input1_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let input2_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        unsafe {
            cuda_tensor_mul(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                size,
            )?;
        }

        Ok(())
    }
}
