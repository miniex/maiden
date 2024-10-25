#![allow(non_camel_case_types)]
use std::ffi::c_void;

#[link(name = "cudart")]
extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(ptr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}

pub const cudaMemcpyHostToDevice: i32 = 1;
pub const cudaMemcpyDeviceToHost: i32 = 2;
