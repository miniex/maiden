use crate::error::{CudaError, CudaResult};
use maidenx_cuda_sys as sys;
use std::ptr::null_mut;

#[derive(Debug, Clone)]
pub struct CudaDeviceProperties {
    pub name: String,
    pub total_memory: usize,
    pub compute_capability: (i32, i32),
    pub max_threads_per_block: i32,
    pub max_block_dims: [i32; 3],
    pub max_grid_dims: [i32; 3],
    pub warp_size: i32,
}

#[derive(Debug)]
pub struct CudaStream(pub(crate) sys::cudaStream_t);

#[derive(Debug)]
pub struct CudaDevice {
    index: i32,
    properties: CudaDeviceProperties,
    stream: CudaStream,
}

unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    pub fn new(index: i32) -> CudaResult<Self> {
        unsafe {
            // Get device count
            let mut count = 0;
            if sys::cudaGetDeviceCount(&mut count) != 0 {
                return Err(CudaError::DeviceNotFound);
            }

            if index >= count {
                return Err(CudaError::InvalidDevice(index));
            }

            // Set device
            if sys::cudaSetDevice(index) != 0 {
                return Err(CudaError::InvalidValue);
            }

            // Get device properties
            let mut prop = std::mem::zeroed();
            if sys::cudaGetDeviceProperties(&mut prop, index) != 0 {
                return Err(CudaError::InvalidValue);
            }

            // Create stream
            let mut stream = null_mut();
            if sys::cudaStreamCreate(&mut stream) != 0 {
                return Err(CudaError::InvalidOperation(
                    "Failed to create stream".into(),
                ));
            }

            let properties = CudaDeviceProperties {
                name: String::from_utf8_lossy(&prop.name.map(|x| x as u8))
                    .trim_matches(char::from(0))
                    .to_string(),
                total_memory: prop.totalGlobalMem,
                compute_capability: (prop.major, prop.minor),
                max_threads_per_block: prop.maxThreadsPerBlock,
                max_block_dims: prop.maxThreadsDim,
                max_grid_dims: prop.maxGridSize,
                warp_size: prop.warpSize,
            };

            Ok(Self {
                index,
                properties,
                stream: CudaStream(stream),
            })
        }
    }

    pub fn index(&self) -> i32 {
        self.index
    }

    pub fn properties(&self) -> &CudaDeviceProperties {
        &self.properties
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe {
            if sys::cudaDeviceSynchronize() != 0 {
                return Err(CudaError::InvalidOperation("Synchronization failed".into()));
            }
            Ok(())
        }
    }

    pub fn set_current(&self) -> CudaResult<()> {
        unsafe {
            if sys::cudaSetDevice(self.index) != 0 {
                return Err(CudaError::InvalidOperation(
                    "Failed to set current device".into(),
                ));
            }
            Ok(())
        }
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cudaStreamDestroy(self.0);
        }
    }
}
