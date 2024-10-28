mod error;
use std::sync::Arc;

pub use error::{DeviceError, DeviceResult};
use maidenx_cuda_core::device::CudaDevice;

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(Arc<CudaDevice>),
}

impl Device {
    pub fn cpu() -> Self {
        Device::Cpu
    }

    pub fn cuda(index: i32) -> DeviceResult<Self> {
        match CudaDevice::new(index) {
            Ok(device) => Ok(Device::Cuda(Arc::new(device))),
            Err(err) => Err(DeviceError::from(err)),
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
            Device::Cuda(device) => device.synchronize().map_err(DeviceError::from),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "CPU".to_string(),
            Device::Cuda(device) => format!("CUDA Device {}", device.index()),
        }
    }
}

// Thread-local current device
thread_local! {
    static CURRENT_DEVICE: std::cell::RefCell<Device> = const { std::cell::RefCell::new(Device::Cpu) };
}

pub fn get_current_device() -> Device {
    CURRENT_DEVICE.with(|d| d.borrow().clone())
}

pub fn set_current_device(device: Device) -> DeviceResult<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device() {
        let device = Device::cpu();
        assert!(device.is_cpu());
        assert!(!device.is_cuda());
        assert_eq!(device.name(), "CPU");
    }

    #[test]
    fn test_cuda_device_creation() -> DeviceResult<()> {
        if let Ok(device) = Device::cuda(0) {
            assert!(!device.is_cpu());
            assert!(device.is_cuda());
            assert!(device.name().starts_with("CUDA Device"));
            device.synchronize()?;
        }
        Ok(())
    }

    #[test]
    fn test_device_guard() -> DeviceResult<()> {
        assert!(get_current_device().is_cpu());

        if let Ok(cuda_device) = Device::cuda(0) {
            {
                let _guard = DeviceGuard::new(cuda_device.clone())?;
                assert!(get_current_device().is_cuda());
            }

            assert!(get_current_device().is_cpu());
        }
        Ok(())
    }

    #[test]
    fn test_current_device_thread_local() -> DeviceResult<()> {
        use std::thread;

        assert!(get_current_device().is_cpu());

        if let Ok(cuda_device) = Device::cuda(0) {
            let handle = thread::spawn(move || -> DeviceResult<()> {
                set_current_device(cuda_device)?;
                assert!(get_current_device().is_cuda());
                Ok(())
            });

            assert!(get_current_device().is_cpu());

            handle.join().unwrap()?;
        }
        Ok(())
    }

    #[test]
    fn test_multiple_cuda_devices() -> DeviceResult<()> {
        if let Ok(device0) = Device::cuda(0) {
            assert_eq!(device0.name(), "CUDA Device 0");

            if let Ok(device1) = Device::cuda(1) {
                assert_eq!(device1.name(), "CUDA Device 1");

                {
                    let _guard0 = DeviceGuard::new(device0.clone())?;
                    assert!(get_current_device().is_cuda());

                    {
                        let _guard1 = DeviceGuard::new(device1.clone())?;
                        assert!(get_current_device().is_cuda());
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_device_synchronization() -> DeviceResult<()> {
        let cpu_device = Device::cpu();
        cpu_device.synchronize()?;

        if let Ok(cuda_device) = Device::cuda(0) {
            cuda_device.synchronize()?;
        }
        Ok(())
    }

    #[test]
    fn test_invalid_cuda_device() {
        let result = Device::cuda(9999);
        assert!(result.is_err());
    }
}
