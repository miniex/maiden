pub mod buffer;
pub mod device;
pub mod error;

pub mod prelude {
    pub use crate::buffer::CudaBuffer;
    pub use crate::error::{CudaError, CudaResult};
}
