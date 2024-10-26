#[derive(Debug)]
pub enum CudaError {
    AllocationFailed,   // CUDA 메모리 할당 실패
    DeallocationFailed, // CUDA 메모리 해제 실패
    MemcpyFailed,       // 메모리 복사 실패
    KernelLaunchFailed, // CUDA 커널 실행 실패
    InvalidValue,       // 잘못된 값 입력
    ShapeMismatch,      // 텐서 형상 불일치
    InvalidSize,        // 메모리 크기 불일치
}

pub type CudaResult<T> = Result<T, CudaError>;

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CudaError::AllocationFailed => write!(f, "CUDA memory allocation failed"),
            CudaError::DeallocationFailed => write!(f, "CUDA memory deallocation failed"),
            CudaError::MemcpyFailed => write!(f, "CUDA memory copy failed"),
            CudaError::KernelLaunchFailed => write!(f, "CUDA kernel launch failed"),
            CudaError::InvalidValue => write!(f, "Invalid value provided"),
            CudaError::ShapeMismatch => write!(f, "Tensor shapes do not match"),
            CudaError::InvalidSize => write!(f, "Buffer size mismatch"),
        }
    }
}

impl std::error::Error for CudaError {}
