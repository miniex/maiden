pub(super) mod linear;

#[cfg(test)]
mod tests {
    use super::linear;
    use maidenx_cuda_core::{error::CudaResult, prelude::CudaBuffer};

    #[test]
    fn test_linear_forward() -> CudaResult<()> {
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let weight_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];

        let bias_data = vec![0.1f32, 0.2];

        let mut input_buf = CudaBuffer::new(input_data.len() * std::mem::size_of::<f32>())?;
        let mut weight_buf = CudaBuffer::new(weight_data.len() * std::mem::size_of::<f32>())?;
        let mut bias_buf = CudaBuffer::new(bias_data.len() * std::mem::size_of::<f32>())?;
        let mut output_buf = CudaBuffer::new(4 * std::mem::size_of::<f32>())?; // batch_size * out_features

        input_buf.copy_from_host(&input_data)?;
        weight_buf.copy_from_host(&weight_data)?;
        bias_buf.copy_from_host(&bias_data)?;

        unsafe {
            linear::cuda_linear_forward(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                weight_buf.as_ptr(),
                Some(bias_buf.as_ptr()),
                2, // batch_size
                2, // out_features
                3, // in_features
            )?;
        }

        let mut output_data = vec![0.0f32; 4];
        output_buf.copy_to_host(&mut output_data)?;

        let expected = [
            1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3 + 0.1,
            1.0 * 0.4 + 2.0 * 0.5 + 3.0 * 0.6 + 0.2,
            4.0 * 0.1 + 5.0 * 0.2 + 6.0 * 0.3 + 0.1,
            4.0 * 0.4 + 5.0 * 0.5 + 6.0 * 0.6 + 0.2,
        ];

        const EPSILON: f32 = 1e-3;

        for (i, (actual, expected)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < EPSILON,
                "Mismatch at index {}: Expected {}, got {}",
                i,
                expected,
                actual
            );
        }

        Ok(())
    }

    #[test]
    fn test_bilinear_forward() -> CudaResult<()> {
        let input1_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input2_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];

        let weight_data = vec![
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ];

        let bias_data = vec![0.1f32, 0.2];

        let mut input1_buf = CudaBuffer::new(input1_data.len() * std::mem::size_of::<f32>())?;
        let mut input2_buf = CudaBuffer::new(input2_data.len() * std::mem::size_of::<f32>())?;
        let mut weight_buf = CudaBuffer::new(weight_data.len() * std::mem::size_of::<f32>())?;
        let mut bias_buf = CudaBuffer::new(bias_data.len() * std::mem::size_of::<f32>())?;
        let mut output_buf = CudaBuffer::new(4 * std::mem::size_of::<f32>())?; // batch_size * out_features

        input1_buf.copy_from_host(&input1_data)?;
        input2_buf.copy_from_host(&input2_data)?;
        weight_buf.copy_from_host(&weight_data)?;
        bias_buf.copy_from_host(&bias_data)?;

        unsafe {
            linear::cuda_bilinear_forward(
                output_buf.as_mut_ptr(),
                input1_buf.as_ptr(),
                input2_buf.as_ptr(),
                weight_buf.as_ptr(),
                Some(bias_buf.as_ptr()),
                2, // batch_size
                2, // out_features
                2, // in1_features
                3, // in2_features
            )?;
        }

        let mut output_data = vec![0.0f32; 4];
        output_buf.copy_to_host(&mut output_data)?;

        assert!(output_data.iter().all(|&x| x.is_finite()));

        Ok(())
    }

    #[test]
    fn test_linear_without_bias() -> CudaResult<()> {
        let input_data = vec![1.0f32, 2.0, 3.0];
        let weight_data = vec![0.1f32, 0.2, 0.3];

        let mut input_buf = CudaBuffer::new(input_data.len() * std::mem::size_of::<f32>())?;
        let mut weight_buf = CudaBuffer::new(weight_data.len() * std::mem::size_of::<f32>())?;
        let mut output_buf = CudaBuffer::new(std::mem::size_of::<f32>())?;

        input_buf.copy_from_host(&input_data)?;
        weight_buf.copy_from_host(&weight_data)?;

        unsafe {
            linear::cuda_linear_forward(
                output_buf.as_mut_ptr(),
                input_buf.as_ptr(),
                weight_buf.as_ptr(),
                None, // none bias
                1,    // batch_size
                1,    // out_features
                3,    // in_features
            )?;
        }

        let mut output_data = vec![0.0f32; 1];
        output_buf.copy_to_host(&mut output_data)?;

        let expected = 1.4f32;
        assert!(
            (output_data[0] - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            output_data[0]
        );

        Ok(())
    }
}
