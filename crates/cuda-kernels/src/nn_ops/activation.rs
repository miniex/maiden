pub(super) mod relu;
pub(super) mod sigmoid;
pub(super) mod tanh;

#[cfg(test)]
mod tests {
    use super::{relu, sigmoid, tanh};
    use maiden_cuda_core::{error::CudaResult, prelude::CudaBuffer};

    #[test]
    fn test_relu_forward() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input_data: Vec<f32> = (-512..512).map(|x| x as f32).collect();
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            relu::cuda_relu_forward(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            let expected = input_data[i].max(0.0);
            assert!(
                (val - expected).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_sigmoid_forward() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input_data: Vec<f32> = (-512..512).map(|x| x as f32 * 0.01).collect();
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            sigmoid::cuda_sigmoid_forward(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            let expected = 1.0 / (1.0 + (-input_data[i]).exp());
            assert!(
                (val - expected).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_tanh_forward() -> CudaResult<()> {
        let size = 1024;
        let mut output_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;
        let mut input_buf = CudaBuffer::new(size * std::mem::size_of::<f32>())?;

        let input_data: Vec<f32> = (-512..512).map(|x| x as f32 * 0.01).collect();
        input_buf.copy_from_host(&input_data)?;

        unsafe {
            tanh::cuda_tanh_forward(output_buf.as_mut_ptr(), input_buf.as_ptr(), size)?;
        }

        let mut output_data = vec![0.0f32; size];
        output_buf.copy_to_host(&mut output_data)?;

        for (i, &val) in output_data.iter().enumerate() {
            let expected = input_data[i].tanh();
            assert!(
                (val - expected).abs() < 1e-5,
                "Mismatch at position {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }

        Ok(())
    }
}
