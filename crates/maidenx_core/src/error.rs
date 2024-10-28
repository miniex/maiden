use maidenx_cuda_core::error::CudaError;
use std::fmt;

#[derive(Debug)]
pub enum MaidenXError {
    // Shape
    ShapeMismatch(String),
    InvalidShape(String),

    // Operation
    InvalidOperation(String),
    InvalidArgument(String),
    InvalidSize(String),

    // Memory
    OutOfMemory(String),
    BufferSizeMismatch(String),

    // Device
    DeviceError(String),
    UnsupportedDevice(String),

    // Tensor
    TensorError(TensorError),

    // CUDA
    Cuda(CudaError),

    // etc
    IoError(std::io::Error),
    Other(String),
}

#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch(String),
    DimensionMismatch(String),
    InvalidOperation(String),
    DataError(String),
    IndexError(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            TensorError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            TensorError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            TensorError::DataError(msg) => write!(f, "Data error: {}", msg),
            TensorError::IndexError(msg) => write!(f, "Index error: {}", msg),
        }
    }
}

impl std::error::Error for TensorError {}

impl fmt::Display for MaidenXError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MaidenXError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            MaidenXError::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
            MaidenXError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            MaidenXError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            MaidenXError::InvalidSize(msg) => write!(f, "Invalid size: {}", msg),
            MaidenXError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            MaidenXError::BufferSizeMismatch(msg) => write!(f, "Buffer size mismatch: {}", msg),
            MaidenXError::DeviceError(msg) => write!(f, "Device error: {}", msg),
            MaidenXError::UnsupportedDevice(msg) => write!(f, "Unsupported device: {}", msg),
            MaidenXError::TensorError(err) => write!(f, "Tensor error: {}", err),
            MaidenXError::Cuda(err) => write!(f, "CUDA error: {}", err),
            MaidenXError::IoError(err) => write!(f, "IO error: {}", err),
            MaidenXError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for MaidenXError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MaidenXError::Cuda(err) => Some(err),
            MaidenXError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<TensorError> for MaidenXError {
    fn from(err: TensorError) -> Self {
        MaidenXError::TensorError(err)
    }
}

impl From<CudaError> for MaidenXError {
    fn from(err: CudaError) -> Self {
        MaidenXError::Cuda(err)
    }
}

impl From<std::io::Error> for MaidenXError {
    fn from(err: std::io::Error) -> Self {
        MaidenXError::IoError(err)
    }
}

pub type Result<T> = std::result::Result<T, MaidenXError>;
