use maidenx_cuda_core::error::CudaResult;
pub use maidenx_nn_macros::Module;
use maidenx_tensor::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> CudaResult<Tensor>;
    fn parameters(&self) -> Vec<Tensor>;
}
