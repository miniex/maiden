#[cfg(feature = "cuda")]
use maidenx_cuda_core::error::CudaError;
use std::fmt;

#[derive(Debug)]
pub enum DeviceError {
    DeviceNotFound(String),
    DeviceInitializationFailed(String),
    UnsupportedDevice(String),
    InvalidDeviceIndex(i32),
    DeviceNotAvailable(String),
    DeviceOutOfMemory(String),
    InvalidDeviceProperty(String),
    OperationNotSupported(String),
    DeviceSynchronizationFailed,
    StreamError(String),
    UnsupportedOperation(String),
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::DeviceNotFound(msg) => write!(f, "Device not found: {}", msg),
            Self::DeviceInitializationFailed(msg) => {
                write!(f, "Failed to initialize device: {}", msg)
            }
            Self::UnsupportedDevice(msg) => write!(f, "Unsupported device: {}", msg),
            Self::InvalidDeviceIndex(idx) => write!(f, "Invalid device index: {}", idx),
            Self::DeviceNotAvailable(msg) => write!(f, "Device not available: {}", msg),
            Self::DeviceOutOfMemory(msg) => write!(f, "Device out of memory: {}", msg),
            Self::InvalidDeviceProperty(msg) => write!(f, "Invalid device property: {}", msg),
            Self::OperationNotSupported(msg) => write!(f, "Operation not supported: {}", msg),
            Self::DeviceSynchronizationFailed => write!(f, "Device synchronization failed"),
            Self::StreamError(msg) => write!(f, "Stream error: {}", msg),
            Self::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            #[cfg(feature = "cuda")]
            Self::Cuda(err) => write!(f, "CUDA error: {}", err),
        }
    }
}

impl std::error::Error for DeviceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(err) => Some(err),
            _ => None,
        }
    }
}

#[cfg(feature = "cuda")]
impl From<CudaError> for DeviceError {
    fn from(err: CudaError) -> Self {
        DeviceError::Cuda(err)
    }
}

pub type DeviceResult<T> = Result<T, DeviceError>;

