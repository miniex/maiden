#[derive(Debug)]
pub enum CudaError {
    AllocationFailed,         // CUDA 메모리 할당 실패
    DeallocationFailed,       // CUDA 메모리 해제 실패
    MemcpyFailed,             // 메모리 복사 실패
    KernelLaunchFailed,       // CUDA 커널 실행 실패
    InvalidValue,             // 잘못된 값 입력
    ShapeMismatch,            // 텐서 형상 불일치
    InvalidSize,              // 메모리 크기 불일치
    InvalidOperation(String), // 잘못된 연산 수행
    InvalidArgument(String),  // 잘못된 인자

    // 새로 추가된 디바이스 관련 에러들
    DeviceNotFound,              // CUDA 디바이스를 찾을 수 없음
    InvalidDevice(i32),          // 잘못된 디바이스 인덱스
    DeviceInitializationFailed,  // 디바이스 초기화 실패
    DeviceSynchronizationFailed, // 디바이스 동기화 실패
    StreamCreationFailed,        // 스트림 생성 실패
}

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
            CudaError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            CudaError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            CudaError::DeviceNotFound => write!(f, "No CUDA device found"),
            CudaError::InvalidDevice(idx) => write!(f, "Invalid CUDA device index: {}", idx),
            CudaError::DeviceInitializationFailed => write!(f, "Failed to initialize CUDA device"),
            CudaError::DeviceSynchronizationFailed => {
                write!(f, "CUDA device synchronization failed")
            }
            CudaError::StreamCreationFailed => write!(f, "Failed to create CUDA stream"),
        }
    }
}

impl std::error::Error for CudaError {}

pub type CudaResult<T> = Result<T, CudaError>;
