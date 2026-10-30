use cudarc::cublas::result::CublasError;
use cudarc::driver::DriverError;
use cudarc::nvrtc::CompileError;

/// Everything that can go wrong on the GPU side
#[derive(Debug)]
pub enum Error {
    /// Matrix dimensions don't line up for the operation
    Shape(String),
    /// CUDA driver: allocation, copies, kernel launches
    Driver(DriverError),
    /// cuBLAS (matmul)
    Cublas(CublasError),
    /// NVRTC failed to compile the kernel source
    Compile(CompileError),
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Shape(msg) => write!(f, "shape error: {msg}"),
            Error::Driver(e) => write!(f, "cuda driver error: {e}"),
            Error::Cublas(e) => write!(f, "cublas error: {e}"),
            Error::Compile(e) => write!(f, "kernel compile error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<DriverError> for Error {
    fn from(e: DriverError) -> Self {
        Error::Driver(e)
    }
}

impl From<CublasError> for Error {
    fn from(e: CublasError) -> Self {
        Error::Cublas(e)
    }
}

impl From<CompileError> for Error {
    fn from(e: CompileError) -> Self {
        Error::Compile(e)
    }
}
