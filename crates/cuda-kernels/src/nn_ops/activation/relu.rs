use maiden_cuda_core::prelude::CudaResult;

#[link(name = "nn_ops")]
extern "C" {
    fn relu_forward(output: *mut f32, input: *const f32, size: usize);
}

/// Performs ReLU forward operation on CUDA device.
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
pub unsafe fn cuda_relu_forward(
    output: *mut f32,
    input: *const f32,
    size: usize,
) -> CudaResult<()> {
    relu_forward(output, input, size);
    // TODO add CUDA Error check
    Ok(())
}
