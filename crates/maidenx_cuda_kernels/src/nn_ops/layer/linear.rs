use maidenx_cuda_core::prelude::CudaResult;

#[link(name = "nn_ops")]
extern "C" {
    fn linear_forward(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        batch_size: i32,
        out_features: i32,
        in_features: i32,
    );

    fn bilinear_forward(
        output: *mut f32,
        input1: *const f32,
        input2: *const f32,
        weight: *const f32,
        bias: *const f32,
        batch_size: i32,
        out_features: i32,
        in1_features: i32,
        in2_features: i32,
    );
}

/// Performs Linear forward operation on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input` - Pointer to the input buffer on CUDA device
/// * `weight` - Pointer to the weight buffer on CUDA device
/// * `bias` - Optional pointer to the bias buffer on CUDA device
/// * `batch_size` - Size of the batch dimension
/// * `out_features` - Number of output features
/// * `in_features` - Number of input features
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * `output` buffer has enough space for batch_size * out_features elements
/// * `input` buffer contains at least batch_size * in_features elements
/// * `weight` buffer contains at least out_features * in_features elements
/// * If bias is Some, it contains at least out_features elements
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
pub unsafe fn cuda_linear_forward(
    output: *mut f32,
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    batch_size: i32,
    out_features: i32,
    in_features: i32,
) -> CudaResult<()> {
    linear_forward(
        output,
        input,
        weight,
        bias.unwrap_or(std::ptr::null()),
        batch_size,
        out_features,
        in_features,
    );
    Ok(())
}

/// Performs Bilinear forward operation on CUDA device.
///
/// # Arguments
///
/// * `output` - Pointer to the output buffer on CUDA device
/// * `input1` - Pointer to the first input buffer on CUDA device
/// * `input2` - Pointer to the second input buffer on CUDA device
/// * `weight` - Pointer to the weight buffer on CUDA device
/// * `bias` - Optional pointer to the bias buffer on CUDA device
///
/// # Safety
///
/// Caller must ensure that:
/// * All pointers point to valid memory on the CUDA device
/// * Memory regions do not overlap
/// * All memory is properly aligned for f32
#[allow(clippy::too_many_arguments)]
pub unsafe fn cuda_bilinear_forward(
    output: *mut f32,
    input1: *const f32,
    input2: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    batch_size: i32,
    out_features: i32,
    in1_features: i32,
    in2_features: i32,
) -> CudaResult<()> {
    bilinear_forward(
        output,
        input1,
        input2,
        weight,
        bias.unwrap_or(std::ptr::null()),
        batch_size,
        out_features,
        in1_features,
        in2_features,
    );
    Ok(())
}
