use crate::{
    device::Device,
    error::{MaidenXError, Result},
};
use maidenx_cpu_core::buffer::CpuBuffer;
use maidenx_cuda_core::buffer::CudaBuffer;
use std::sync::Arc;

pub trait Buffer: Send + Sync {
    fn copy_from_host(&mut self, data: &[f32]) -> Result<()>;
    fn copy_to_host(&self, data: &mut [f32]) -> Result<()>;
    fn len(&self) -> usize;
    fn as_ptr(&self) -> *const f32;
    fn as_mut_ptr(&mut self) -> *mut f32;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
pub enum DeviceBuffer {
    Cpu(Arc<CpuBuffer>),
    Cuda(Arc<CudaBuffer>),
}

impl DeviceBuffer {
    pub fn new(size: usize, device: &Device) -> Result<Self> {
        match device {
            Device::Cpu => {
                let buffer = CpuBuffer::new(size).map_err(|e| {
                    MaidenXError::TensorError(crate::error::TensorError::DataError(e.to_string()))
                })?;
                Ok(DeviceBuffer::Cpu(Arc::new(buffer)))
            }
            Device::Cuda(_) => {
                let buffer = CudaBuffer::new(size).map_err(MaidenXError::from)?;
                Ok(DeviceBuffer::Cuda(Arc::new(buffer)))
            }
        }
    }

    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<()> {
        match self {
            DeviceBuffer::Cpu(buf) => {
                if let Some(buf) = Arc::get_mut(buf) {
                    buf.copy_from_host(data).map_err(|e| {
                        MaidenXError::TensorError(crate::error::TensorError::DataError(
                            e.to_string(),
                        ))
                    })
                } else {
                    let mut new_buf = (**buf).clone();
                    new_buf.copy_from_host(data).map_err(|e| {
                        MaidenXError::TensorError(crate::error::TensorError::DataError(
                            e.to_string(),
                        ))
                    })?;
                    *self = DeviceBuffer::Cpu(Arc::new(new_buf));
                    Ok(())
                }
            }
            DeviceBuffer::Cuda(buf) => {
                if let Some(buf) = Arc::get_mut(buf) {
                    buf.copy_from_host(data).map_err(MaidenXError::from)
                } else {
                    let mut new_buf = (**buf).clone();
                    new_buf.copy_from_host(data).map_err(MaidenXError::from)?;
                    *self = DeviceBuffer::Cuda(Arc::new(new_buf));
                    Ok(())
                }
            }
        }
    }

    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<()> {
        match self {
            DeviceBuffer::Cpu(buf) => buf.copy_to_host(data).map_err(|e| {
                MaidenXError::TensorError(crate::error::TensorError::DataError(e.to_string()))
            }),
            DeviceBuffer::Cuda(buf) => buf.copy_to_host(data).map_err(MaidenXError::from),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            DeviceBuffer::Cpu(buf) => buf.len(),
            DeviceBuffer::Cuda(buf) => buf.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_ptr(&self) -> *const f32 {
        match self {
            DeviceBuffer::Cpu(buf) => buf.as_ptr(),
            DeviceBuffer::Cuda(buf) => buf.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        match self {
            DeviceBuffer::Cpu(buf) => {
                if let Some(buf) = Arc::get_mut(buf) {
                    buf.as_mut_ptr()
                } else {
                    let new_buf = (**buf).clone();
                    *self = DeviceBuffer::Cpu(Arc::new(new_buf));
                    Arc::get_mut(match self {
                        DeviceBuffer::Cpu(buf) => buf,
                        _ => unreachable!(),
                    })
                    .unwrap()
                    .as_mut_ptr()
                }
            }
            DeviceBuffer::Cuda(buf) => {
                if let Some(buf) = Arc::get_mut(buf) {
                    buf.as_mut_ptr()
                } else {
                    let new_buf = (**buf).clone();
                    *self = DeviceBuffer::Cuda(Arc::new(new_buf));
                    Arc::get_mut(match self {
                        DeviceBuffer::Cuda(buf) => buf,
                        _ => unreachable!(),
                    })
                    .unwrap()
                    .as_mut_ptr()
                }
            }
        }
    }
}

impl Buffer for DeviceBuffer {
    fn copy_from_host(&mut self, data: &[f32]) -> Result<()> {
        self.copy_from_host(data)
    }

    fn copy_to_host(&self, data: &mut [f32]) -> Result<()> {
        self.copy_to_host(data)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn as_ptr(&self) -> *const f32 {
        self.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.as_mut_ptr()
    }
}
