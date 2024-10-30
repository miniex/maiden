#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod buffer;
pub mod device;
pub mod error;

pub mod prelude {
    pub use crate::buffer::CudaBuffer;
    pub use crate::error::{CudaError, CudaResult};
}

use std::ffi::c_void;

#[link(name = "cudart")]
extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(devPtr: *mut c_void) -> i32;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> i32;

    // Device Management
    pub fn cudaGetDeviceCount(count: *mut i32) -> i32;
    pub fn cudaSetDevice(device: i32) -> i32;
    pub fn cudaGetDevice(device: *mut i32) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: i32) -> i32;

    // Stream Management
    pub fn cudaStreamCreate(pStream: *mut cudaStream_t) -> i32;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> i32;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> i32;
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
}

// Device properties structure
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaDeviceProp {
    pub name: [i8; 256],
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: i32,
    pub warpSize: i32,
    pub maxThreadsPerBlock: i32,
    pub maxThreadsDim: [i32; 3],
    pub maxGridSize: [i32; 3],
    pub clockRate: i32,
    pub totalConstMem: usize,
    pub major: i32,
    pub minor: i32,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub computeMode: i32,
}

// Stream type
pub type cudaStream_t = *mut c_void;

// Error codes
pub const cudaSuccess: i32 = 0;

// Error handling helper
pub fn check_cuda_error(error: i32) -> Result<(), i32> {
    if error == cudaSuccess {
        Ok(())
    } else {
        Err(error)
    }
}
