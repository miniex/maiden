use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(Arc<CudaDevice>),
}

impl Device {
    pub fn cpu() -> Self {
        Device::Cpu
    }

    pub fn index(&self) -> Option<i32> {
        match self {
            Device::Cpu => None,
            Device::Cuda(dev) => Some(dev.index),
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

#[derive(Debug, Clone)]
pub struct DeviceGuard {
    previous: Device,
    current: Device,
}

impl DeviceGuard {
    pub fn new(device: Device) -> Result<Self, DeviceError> {
        let previous = CURRENT_DEVICE.with(|d| d.borrow().clone());
        CURRENT_DEVICE.with(|d| *d.borrow_mut() = device.clone());

        Ok(Self {
            previous,
            current: device,
        })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        CURRENT_DEVICE.with(|d| *d.borrow_mut() = self.previous.clone());
    }
}
