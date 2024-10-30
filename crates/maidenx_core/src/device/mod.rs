mod error;
pub use error::{DeviceError, DeviceResult};
#[cfg(feature = "cuda")]
use maidenx_cuda_core::device::CudaDevice;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaDevice>),
    #[cfg(not(feature = "cuda"))]
    Cuda(i32),
}

impl Device {
    pub fn cpu() -> Self {
        Device::Cpu
    }

    pub fn cuda(index: i32) -> DeviceResult<Self> {
        #[cfg(feature = "cuda")]
        {
            match CudaDevice::new(index) {
                Ok(device) => Ok(Device::Cuda(Arc::new(device))),
                Err(err) => Err(DeviceError::from(err)),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            log::warn!("CUDA support is not enabled. Using fallback implementation.");
            Ok(Device::Cuda(index))
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    pub fn synchronize(&self) -> DeviceResult<()> {
        match self {
            Device::Cpu => Ok(()),
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => device.synchronize().map_err(DeviceError::from),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                log::warn!("CUDA operations are not available - feature not enabled");
                Ok(())
            }
        }
    }

    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "CPU".to_string(),
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => format!("CUDA Device {}", device.index()),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(index) => format!("CUDA Device {} (disabled)", index),
        }
    }
}

thread_local! {
    static CURRENT_DEVICE: std::cell::RefCell<Device> = const { std::cell::RefCell::new(Device::Cpu) };
}

pub fn get_current_device() -> Device {
    CURRENT_DEVICE.with(|d| d.borrow().clone())
}

pub fn set_current_device(device: Device) -> DeviceResult<()> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda_device) = &device {
        cuda_device.set_current()?;
    }

    CURRENT_DEVICE.with(|d| *d.borrow_mut() = device);
    Ok(())
}

pub struct DeviceGuard {
    prev_device: Device,
}

impl DeviceGuard {
    pub fn new(device: Device) -> DeviceResult<Self> {
        let prev_device = get_current_device();
        set_current_device(device)?;
        Ok(Self { prev_device })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        let _ = set_current_device(self.prev_device.clone());
    }
}

