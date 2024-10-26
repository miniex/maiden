use cuda_core::prelude::CudaResult;

#[link(name = "tensor_ops")]
extern "C" {
    fn tensor_add(output: *mut f32, input1: *const f32, input2: *const f32, size: usize);
    fn tensor_mul(output: *mut f32, input1: *const f32, input2: *const f32, size: usize);
    fn tensor_matmul(
        output: *mut f32,
        input1: *const f32,
        input2: *const f32,
        m: i32,
        n: i32,
        k: i32,
    );
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

/// Performs matrix multiplication of two tensors on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input1` - Pointer to the first input matrix buffer on CUDA device
/// * `input2` - Pointer to the second input matrix buffer on CUDA device
/// * `m` - Number of rows in the first matrix
/// * `n` - Number of columns in the second matrix
/// * `k` - Number of columns in the first matrix (and rows in the second matrix)
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for `m * n` elements
/// * `input1` buffer contains at least `m * k` elements
/// * `input2` buffer contains at least `k * n` elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
pub unsafe fn cuda_tensor_matmul(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    m: i32,
    n: i32,
    k: i32,
) -> CudaResult<()> {
    tensor_matmul(output, input1, input2, m, n, k);
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
        let mut input1_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input2_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input1_data: Vec<f32> = vec![1.0; size];
        let input2_data: Vec<f32> = vec![2.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        unsafe {
            cuda_tensor_add(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                size,
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - 3.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                3.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_tensor_mul() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input1_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input2_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input1_data: Vec<f32> = vec![2.0; size];
        let input2_data: Vec<f32> = vec![3.0; size];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        unsafe {
            cuda_tensor_mul(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                size,
            )?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - 6.0).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                6.0,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_tensor_matmul() -> CudaResult<()> {
        let m = 32;
        let n = 32;
        let k = 32;

        let mut output_buf = CudaBuffer::new((m * n) as usize * std::mem::size_of::<f32>())?;
        let mut input1_buf = CudaBuffer::new((m * k) as usize * std::mem::size_of::<f32>())?;
        let mut input2_buf = CudaBuffer::new((k * n) as usize * std::mem::size_of::<f32>())?;

        let input1_data: Vec<f32> = vec![1.0; (m * k) as usize];
        let input2_data: Vec<f32> = vec![2.0; (k * n) as usize];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        unsafe {
            cuda_tensor_matmul(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                m,
                n,
                k,
            )?;
        }

        let mut output_data = vec![0.0f32; (m * n) as usize];
        output_buf.copy_to_host(&mut output_data)?;

        let expected_value = 2.0 * k as f32;
        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - expected_value).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                expected_value,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_tensor_matmul_large() -> CudaResult<()> {
        let m = 1024;
        let n = 1024;
        let k = 1024;

        let mut output_buf = CudaBuffer::new((m * n) as usize * std::mem::size_of::<f32>())?;
        let mut input1_buf = CudaBuffer::new((m * k) as usize * std::mem::size_of::<f32>())?;
        let mut input2_buf = CudaBuffer::new((k * n) as usize * std::mem::size_of::<f32>())?;

        let input1_data: Vec<f32> = vec![0.1; (m * k) as usize];
        let input2_data: Vec<f32> = vec![0.1; (k * n) as usize];

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;

        unsafe {
            cuda_tensor_matmul(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                m,
                n,
                k,
            )?;
        }

        let mut output_data = vec![0.0f32; (m * n) as usize];
        output_buf.copy_to_host(&mut output_data)?;

        let expected_value = 0.01 * k as f32;
        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                (val - expected_value).abs() < 1e-3,
                "Mismatch at position {}: expected {}, got {}",
                i,
                expected_value,
                val
            );
        }

        Ok(())
    }
}
