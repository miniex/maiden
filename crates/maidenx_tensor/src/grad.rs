use crate::Tensor;
use maidenx_core::error::Result;

impl Tensor {
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    pub fn backward(&self) -> Result<()> {
        if !self.requires_grad {
            return Ok(());
        }
        if self.grad.borrow().is_none() {
            *self.grad.borrow_mut() = Some(Box::new(Tensor::ones(self.shape())?));
        }
        Ok(())
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.grad.borrow().as_ref().map(|g| (**g).clone())
    }
}
