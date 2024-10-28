use std::{fmt, mem, ptr::NonNull};

#[derive(Clone)]
pub struct CpuBuffer {
    ptr: NonNull<f32>,
    size: usize,
}

impl fmt::Debug for CpuBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBuffer")
            .field("ptr", &self.ptr)
            .field("size", &self.size)
            .finish()
    }
}

impl CpuBuffer {
    pub fn new(size: usize) -> Result<Self, &'static str> {
        let layout =
            std::alloc::Layout::array::<f32>(size).map_err(|_| "Failed to create memory layout")?;

        let ptr = unsafe {
            let ptr = std::alloc::alloc(layout) as *mut f32;
            if ptr.is_null() {
                return Err("Memory allocation failed");
            }
            // 0으로 초기화
            ptr.write_bytes(0, size);
            NonNull::new_unchecked(ptr)
        };

        Ok(Self { ptr, size })
    }

    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<(), &'static str> {
        if mem::size_of_val(data) > self.size {
            return Err("Invalid size");
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr(), data.len());
        }

        Ok(())
    }

    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<(), &'static str> {
        if mem::size_of_val(data) > self.size {
            return Err("Invalid size");
        }

        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), data.as_mut_ptr(), data.len());
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr.as_ptr()
    }
}

impl Drop for CpuBuffer {
    fn drop(&mut self) {
        unsafe {
            let layout = std::alloc::Layout::array::<f32>(self.size)
                .expect("Failed to create layout for deallocation");
            std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}
