#[derive(Debug)]
pub enum CpuError {
    AllocationFailed,
    InvalidValue,
    Other(String),
}

pub type CpuResult<T> = Result<T, CpuError>;
