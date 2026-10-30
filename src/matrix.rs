use std::sync::Arc;

use cudarc::cublas::{Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use crate::error::{Error, Result};
use crate::gpu::Gpu;

/// A row-major f32 matrix that lives in GPU memory.
///
/// Every operation returns a new matrix and leaves its inputs untouched.
/// GPU memory is freed automatically when a Matrix is dropped.
pub struct Matrix {
    data: CudaSlice<f32>,
    rows: usize,
    cols: usize,
    gpu: Arc<Gpu>,
}

const TILE: u32 = 16;

/// One thread per element, 256 threads per block.
fn cfg_1d(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

/// 16x16 thread blocks covering a rows x cols grid (x = column, y = row).
fn cfg_2d(rows: usize, cols: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            (cols as u32).div_ceil(TILE),
            (rows as u32).div_ceil(TILE),
            1,
        ),
        block_dim: (TILE, TILE, 1),
        shared_mem_bytes: 0,
    }
}

fn check_dims(rows: usize, cols: usize) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(Error::Shape(format!("empty matrix {rows}x{cols}")));
    }
    // Kernels index with 32-bit ints.
    if rows
        .checked_mul(cols)
        .map_or(true, |n| n > i32::MAX as usize)
    {
        return Err(Error::Shape(format!("matrix {rows}x{cols} is too large")));
    }
    Ok(())
}

impl Matrix {
    // ---------- creation and transfer ----------

    /// Uploads row-major host data to the GPU.
    pub fn from_vec(gpu: &Arc<Gpu>, rows: usize, cols: usize, data: &[f32]) -> Result<Self> {
        check_dims(rows, cols)?;
        if data.len() != rows * cols {
            return Err(Error::Shape(format!(
                "{} values given for a {rows}x{cols} matrix",
                data.len()
            )));
        }
        let data = gpu.stream.clone_htod(data)?;
        Ok(Self {
            data,
            rows,
            cols,
            gpu: gpu.clone(),
        })
    }

    pub fn zeros(gpu: &Arc<Gpu>, rows: usize, cols: usize) -> Result<Self> {
        check_dims(rows, cols)?;
        let data = gpu.stream.alloc_zeros::<f32>(rows * cols)?;
        Ok(Self {
            data,
            rows,
            cols,
            gpu: gpu.clone(),
        })
    }

    /// Copies the matrix back to the CPU (row-major). This waits for all queued
    /// GPU work, so call it only when you need the numbers, e.g. to print the loss.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.gpu.stream.clone_dtoh(&self.data)?)
    }

    /// A device-to-device copy.
    pub fn try_clone(&self) -> Result<Self> {
        let data = self.gpu.stream.clone_dtod(&self.data)?;
        Ok(self.with_data(data, self.rows, self.cols))
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn len(&self) -> usize {
        self.rows * self.cols
    }

    fn with_data(&self, data: CudaSlice<f32>, rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            gpu: self.gpu.clone(),
        }
    }

    fn alloc(&self, n: usize) -> Result<CudaSlice<f32>> {
        Ok(self.gpu.stream.alloc_zeros::<f32>(n)?)
    }

    // ---------- linear algebra ----------

    /// self [m x k] * other [k x n] -> [m x n]
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix> {
        if self.cols != other.rows {
            return Err(Error::Shape(format!(
                "matmul {}x{} by {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut out = self.alloc(m * n)?;

        // cuBLAS is column-major, and a row-major matrix read as column-major is
        // its transpose. So we ask cuBLAS for C^T = B^T * A^T, which it writes in
        // exactly the memory layout of our row-major C. That's why `other` is
        // passed first and m and n are swapped.
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };
        // Safety: the dimensions above match the buffer sizes, which the shape
        // check guarantees.
        unsafe { self.gpu.blas.gemm(cfg, &other.data, &self.data, &mut out) }?;
        Ok(self.with_data(out, m, n))
    }

    /// Transpose: [rows x cols] -> [cols x rows]
    pub fn t(&self) -> Result<Matrix> {
        let mut out = self.alloc(self.len())?;
        let (r_i, c_i) = (self.rows as i32, self.cols as i32);
        let mut launch = self.gpu.stream.launch_builder(&self.gpu.kernels.transpose);
        launch.arg(&self.data).arg(&mut out).arg(&r_i).arg(&c_i);
        unsafe { launch.launch(cfg_2d(self.rows, self.cols)) }?;
        Ok(self.with_data(out, self.cols, self.rows))
    }

    // ---------- elementwise ----------

    fn binary(&self, other: &Matrix, kernel: &CudaFunction, name: &str) -> Result<Matrix> {
        if self.shape() != other.shape() {
            return Err(Error::Shape(format!(
                "{name} {}x{} with {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let n = self.len();
        let mut out = self.alloc(n)?;
        let n_i = n as i32;
        let mut launch = self.gpu.stream.launch_builder(kernel);
        launch
            .arg(&self.data)
            .arg(&other.data)
            .arg(&mut out)
            .arg(&n_i);
        unsafe { launch.launch(cfg_1d(n)) }?;
        Ok(self.with_data(out, self.rows, self.cols))
    }

    fn unary(&self, kernel: &CudaFunction) -> Result<Matrix> {
        let n = self.len();
        let mut out = self.alloc(n)?;
        let n_i = n as i32;
        let mut launch = self.gpu.stream.launch_builder(kernel);
        launch.arg(&self.data).arg(&mut out).arg(&n_i);
        unsafe { launch.launch(cfg_1d(n)) }?;
        Ok(self.with_data(out, self.rows, self.cols))
    }

    pub fn add(&self, other: &Matrix) -> Result<Matrix> {
        self.binary(other, &self.gpu.kernels.add, "add")
    }

    pub fn sub(&self, other: &Matrix) -> Result<Matrix> {
        self.binary(other, &self.gpu.kernels.sub, "sub")
    }

    /// Elementwise (Hadamard) product.
    pub fn mul(&self, other: &Matrix) -> Result<Matrix> {
        self.binary(other, &self.gpu.kernels.mul, "mul")
    }

    /// Multiplies every element by s.
    pub fn scale(&self, s: f32) -> Result<Matrix> {
        let n = self.len();
        let mut out = self.alloc(n)?;
        let n_i = n as i32;
        let mut launch = self.gpu.stream.launch_builder(&self.gpu.kernels.scale);
        launch.arg(&self.data).arg(&s).arg(&mut out).arg(&n_i);
        unsafe { launch.launch(cfg_1d(n)) }?;
        Ok(self.with_data(out, self.rows, self.cols))
    }

    pub fn relu(&self) -> Result<Matrix> {
        self.unary(&self.gpu.kernels.relu)
    }

    /// Backward pass of relu. Call it on the pre-activation `z` from the
    /// forward pass with the incoming gradient: `z.relu_backward(&grad)`.
    pub fn relu_backward(&self, grad: &Matrix) -> Result<Matrix> {
        self.binary(grad, &self.gpu.kernels.relu_backward, "relu_backward")
    }

    // ---------- broadcasting and reductions ----------

    /// Adds a [1 x cols] row vector to every row (bias add).
    pub fn add_row(&self, row_vec: &Matrix) -> Result<Matrix> {
        if row_vec.rows != 1 || row_vec.cols != self.cols {
            return Err(Error::Shape(format!(
                "add_row needs a 1x{} vector, got {}x{}",
                self.cols, row_vec.rows, row_vec.cols
            )));
        }
        let n = self.len();
        let mut out = self.alloc(n)?;
        let (r_i, c_i) = (self.rows as i32, self.cols as i32);
        let mut launch = self.gpu.stream.launch_builder(&self.gpu.kernels.add_row);
        launch
            .arg(&self.data)
            .arg(&row_vec.data)
            .arg(&mut out)
            .arg(&r_i)
            .arg(&c_i);
        unsafe { launch.launch(cfg_1d(n)) }?;
        Ok(self.with_data(out, self.rows, self.cols))
    }

    /// Sums each column: [rows x cols] -> [1 x cols]. This is the bias gradient.
    pub fn sum_cols(&self) -> Result<Matrix> {
        let mut out = self.alloc(self.cols)?;
        let (r_i, c_i) = (self.rows as i32, self.cols as i32);
        let mut launch = self.gpu.stream.launch_builder(&self.gpu.kernels.sum_cols);
        launch.arg(&self.data).arg(&mut out).arg(&r_i).arg(&c_i);
        unsafe { launch.launch(cfg_1d(self.cols)) }?;
        Ok(self.with_data(out, 1, self.cols))
    }

    /// Softmax applied independently to each row.
    pub fn softmax_rows(&self) -> Result<Matrix> {
        let mut out = self.alloc(self.len())?;
        let (r_i, c_i) = (self.rows as i32, self.cols as i32);
        let mut launch = self
            .gpu
            .stream
            .launch_builder(&self.gpu.kernels.softmax_rows);
        launch.arg(&self.data).arg(&mut out).arg(&r_i).arg(&c_i);
        unsafe { launch.launch(cfg_1d(self.rows)) }?;
        Ok(self.with_data(out, self.rows, self.cols))
    }
}

impl std::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix({}x{})", self.rows, self.cols)
    }
}
