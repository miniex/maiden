use maiden_cuda_core::error::CudaResult;
pub use maiden_nn_macros::Module;
use maiden_tensor::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> CudaResult<Tensor>;
    fn parameters(&self) -> Vec<Tensor>;
}
