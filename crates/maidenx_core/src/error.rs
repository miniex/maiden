use crate::device::DeviceError;
use maidenx_cpu_core::error::CpuError;
#[cfg(feature = "cuda")]
use maidenx_cuda_core::error::CudaError;
use std::fmt;

#[derive(Debug)]
pub enum MaidenXError {
    ShapeMismatch(String),
    InvalidShape(String),
    InvalidOperation(String),
    InvalidArgument(String),
    InvalidSize(String),
    OutOfMemory(String),
    BufferSizeMismatch(String),
    Device(DeviceError),
    UnsupportedDevice(String),
    UnsupportedOperation(String),
    TensorError(TensorError),
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
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
    AllocationError(String),
    InvalidValue(String),
    Other(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            TensorError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            TensorError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            TensorError::DataError(msg) => write!(f, "Data error: {}", msg),
            TensorError::IndexError(msg) => write!(f, "Index error: {}", msg),
            TensorError::AllocationError(msg) => write!(f, "Allocation error: {}", msg),
            TensorError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
            TensorError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for TensorError {}

impl fmt::Display for MaidenXError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            Self::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
            Self::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            Self::InvalidSize(msg) => write!(f, "Invalid size: {}", msg),
            Self::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            Self::BufferSizeMismatch(msg) => write!(f, "Buffer size mismatch: {}", msg),
            Self::Device(err) => write!(f, "Device error: {}", err),
            Self::UnsupportedDevice(msg) => write!(f, "Unsupported device: {}", msg),
            Self::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            Self::TensorError(err) => write!(f, "Tensor error: {}", err),
            #[cfg(feature = "cuda")]
            Self::Cuda(err) => write!(f, "CUDA error: {}", err),
            Self::IoError(err) => write!(f, "IO error: {}", err),
            Self::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for MaidenXError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "cuda")]
            MaidenXError::Cuda(err) => Some(err),
            MaidenXError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<DeviceError> for MaidenXError {
    fn from(err: DeviceError) -> Self {
        MaidenXError::Device(err)
    }
}

impl From<TensorError> for MaidenXError {
    fn from(err: TensorError) -> Self {
        MaidenXError::TensorError(err)
    }
}

impl From<CpuError> for MaidenXError {
    fn from(err: CpuError) -> Self {
        match err {
            CpuError::AllocationFailed => MaidenXError::TensorError(TensorError::AllocationError(
                "CPU allocation failed".into(),
            )),
            CpuError::InvalidValue => MaidenXError::TensorError(TensorError::InvalidValue(
                "Invalid CPU buffer value".into(),
            )),
            CpuError::Other(msg) => MaidenXError::TensorError(TensorError::Other(msg)),
        }
    }
}

#[cfg(feature = "cuda")]
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
