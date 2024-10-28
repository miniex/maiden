use maidenx_core::error::{MaidenXError, Result};
pub use maidenx_nn_macros::Module;
use maidenx_tensor::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
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

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x).map_err(|e| {
                MaidenXError::InvalidOperation(format!("Layer forward pass failed: {}", e))
            })?;
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
