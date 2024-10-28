use maidenx_cuda_core::error::CudaResult;
pub use maidenx_nn_macros::Module;
use maidenx_tensor::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> CudaResult<Tensor>;
    fn parameters(&self) -> Vec<Tensor>;
}

#[derive(Default, Module)]
pub struct ModuleBuilder {
    layers: Vec<Box<dyn Module>>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer<M: Module + 'static>(mut self, module: M) -> Self {
        self.layers.push(Box::new(module));
        self
    }

    pub fn forward(&self, input: &Tensor) -> CudaResult<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
