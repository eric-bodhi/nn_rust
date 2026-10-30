use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::compile_ptx;

use crate::error::Result;

/// CUDA source for every elementwise / reduction kernel Matrix uses.
/// Compiled once at startup by NVRTC. All matrices are row-major.
const KERNELS: &str = r#"
extern "C" __global__ void add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void sub(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

extern "C" __global__ void mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

extern "C" __global__ void scale(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * s;
}

extern "C" __global__ void relu(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

// z is the pre-activation, grad is dL/d(relu(z)). Output is dL/dz.
extern "C" __global__ void relu_backward(const float* z, const float* grad, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = z[i] > 0.0f ? grad[i] : 0.0f;
}

// 2D launch: x = column, y = row of the input.
extern "C" __global__ void transpose(const float* in, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) out[c * rows + r] = in[r * cols + c];
}

// One thread per element; row_vec is [1 x cols].
extern "C" __global__ void add_row(const float* a, const float* row_vec, float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[i] = a[i] + row_vec[i % cols];
}

// One thread per column.
extern "C" __global__ void sum_cols(const float* a, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float acc = 0.0f;
    for (int r = 0; r < rows; r++) acc += a[r * cols + c];
    out[c] = acc;
}

// One thread per row. Subtracting the row max keeps expf from overflowing.
extern "C" __global__ void softmax_rows(const float* a, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* in = a + r * cols;
    float* o = out + r * cols;
    float mx = in[0];
    for (int c = 1; c < cols; c++) mx = fmaxf(mx, in[c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        o[c] = expf(in[c] - mx);
        sum += o[c];
    }
    for (int c = 0; c < cols; c++) o[c] /= sum;
}
"#;

/// Handles to the compiled kernels, looked up by name once.
pub struct Kernels {
    pub add: CudaFunction,
    pub sub: CudaFunction,
    pub mul: CudaFunction,
    pub scale: CudaFunction,
    pub relu: CudaFunction,
    pub relu_backward: CudaFunction,
    pub transpose: CudaFunction,
    pub add_row: CudaFunction,
    pub sum_cols: CudaFunction,
    pub softmax_rows: CudaFunction,
}

impl Kernels {
    fn load(module: &Arc<CudaModule>) -> Result<Self> {
        Ok(Self {
            add: module.load_function("add")?,
            sub: module.load_function("sub")?,
            mul: module.load_function("mul")?,
            scale: module.load_function("scale")?,
            relu: module.load_function("relu")?,
            relu_backward: module.load_function("relu_backward")?,
            transpose: module.load_function("transpose")?,
            add_row: module.load_function("add_row")?,
            sum_cols: module.load_function("sum_cols")?,
            softmax_rows: module.load_function("softmax_rows")?,
        })
    }
}

/// One GPU: its context, a stream that all work is queued on, a cuBLAS handle
/// bound to that stream, and the compiled kernels. Shared by every Matrix via Arc.
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    pub kernels: Kernels,
}

impl Gpu {
    /// Opens device `ordinal` (0 for the first GPU) and compiles the kernels.
    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        let ptx = compile_ptx(KERNELS)?;
        let module = ctx.load_module(ptx)?;
        let kernels = Kernels::load(&module)?;
        Ok(Arc::new(Self {
            ctx,
            stream,
            blas,
            kernels,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    #[test]
    fn matmul_and_kernels_run() -> Result<()> {
        let gpu = Gpu::new(0)?;
        // [2x3] * [3x2]
        let a = Matrix::from_vec(&gpu, 2, 3, &[1., 2., 3., 4., 5., 6.])?;
        let b = Matrix::from_vec(&gpu, 3, 2, &[7., 8., 9., 10., 11., 12.])?;
        assert_eq!(a.matmul(&b)?.to_vec()?, vec![58., 64., 139., 154.]);
        assert_eq!(a.t()?.to_vec()?, vec![1., 4., 2., 5., 3., 6.]);
        assert_eq!(a.sum_cols()?.to_vec()?, vec![5., 7., 9.]);
        let s = a.softmax_rows()?.to_vec()?;
        assert!((s[0] + s[1] + s[2] - 1.0).abs() < 1e-6);
        Ok(())
    }
}
