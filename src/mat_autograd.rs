use crate::error::{Error, Result};
use crate::matrix::Matrix;

// automatic back propagation, same idea as autograd.rs but every node is a Matrix
//
// a node's grad has the same shape as its value: grad[r][c] = ∂loss/∂value[r][c]
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Leaf,
    MatMul(usize, usize), // a [m x k] · b [k x n]
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize), // elementwise
    Scale(usize, f32),
    AddRow(usize, usize), // x [m x n] + bias [1 x n] on every row
    Relu(usize),
    T(usize),
}

#[derive(Debug)]
struct Node {
    value: Matrix,
    // None until some gradient flows into this node (saves allocating zeros)
    grad: Option<Matrix>,
    operation: Op,
}

#[derive(Debug, Default)]
pub struct MatTape {
    nodes: Vec<Node>,
}

impl MatTape {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, value: Matrix, operation: Op) -> usize {
        self.nodes.push(Node {
            value,
            grad: None,
            operation,
        });
        self.nodes.len() - 1
    }

    pub fn value(&self, id: usize) -> &Matrix {
        &self.nodes[id].value
    }

    pub fn grad(&self, id: usize) -> Option<&Matrix> {
        self.nodes[id].grad.as_ref()
    }

    // ---------- forward: compute the value now, remember how ----------

    pub fn leaf(&mut self, value: Matrix) -> usize {
        self.push(value, Op::Leaf)
    }

    pub fn matmul(&mut self, a: usize, b: usize) -> Result<usize> {
        let value = self.value(a).matmul(self.value(b))?;
        Ok(self.push(value, Op::MatMul(a, b)))
    }

    pub fn add(&mut self, a: usize, b: usize) -> Result<usize> {
        let value = self.value(a).add(self.value(b))?;
        Ok(self.push(value, Op::Add(a, b)))
    }

    pub fn sub(&mut self, a: usize, b: usize) -> Result<usize> {
        let value = self.value(a).sub(self.value(b))?;
        Ok(self.push(value, Op::Sub(a, b)))
    }

    pub fn mul(&mut self, a: usize, b: usize) -> Result<usize> {
        let value = self.value(a).mul(self.value(b))?;
        Ok(self.push(value, Op::Mul(a, b)))
    }

    pub fn scale(&mut self, a: usize, s: f32) -> Result<usize> {
        let value = self.value(a).scale(s)?;
        Ok(self.push(value, Op::Scale(a, s)))
    }

    pub fn add_row(&mut self, x: usize, bias: usize) -> Result<usize> {
        let value = self.value(x).add_row(self.value(bias))?;
        Ok(self.push(value, Op::AddRow(x, bias)))
    }

    pub fn relu(&mut self, z: usize) -> Result<usize> {
        let value = self.value(z).relu()?;
        Ok(self.push(value, Op::Relu(z)))
    }

    pub fn t(&mut self, a: usize) -> Result<usize> {
        let value = self.value(a).t()?;
        Ok(self.push(value, Op::T(a)))
    }

    // ---------- backward ----------

    // grad += contribution
    fn accumulate(&mut self, id: usize, contribution: Matrix) -> Result<()> {
        let node = &mut self.nodes[id];
        node.grad = Some(match node.grad.take() {
            None => contribution,
            Some(grad) => grad.add(&contribution)?,
        });
        Ok(())
    }

    pub fn zero_grad(&mut self) {
        for node in &mut self.nodes {
            node.grad = None;
        }
    }

    // seed is ∂loss/∂(node `from`), same shape as its value.
    // for a 1x1 loss that's just [[1.0]]; with softmax + cross-entropy you'd
    // seed the logits with (softmax - onehot) / batch instead
    pub fn back_prop(&mut self, from: usize, seed: Matrix) -> Result<()> {
        if seed.shape() != self.value(from).shape() {
            return Err(Error::Shape(format!(
                "seed {:?} for a node of shape {:?}",
                seed.shape(),
                self.value(from).shape()
            )));
        }
        self.accumulate(from, seed)?;

        for i in (0..=from).rev() {
            // nothing flowed into this node, so nothing to pass on
            let Some(g) = &self.nodes[i].grad else {
                continue;
            };

            // work out every (input, contribution) pair while only reading the
            // tape, then add them in below once the borrow of `g` is over
            let contributions: Vec<(usize, Matrix)> = match self.nodes[i].operation {
                Op::Leaf => vec![],
                Op::MatMul(a, b) => {
                    let (a_val, b_val) = (self.value(a), self.value(b));
                    vec![(a, g.matmul(&b_val.t()?)?), (b, a_val.t()?.matmul(g)?)]
                }
                Op::Add(a, b) => {
                    vec![(a, g.try_clone()?), (b, g.try_clone()?)]
                }
                Op::Sub(a, b) => {
                    vec![(a, g.try_clone()?), (b, g.scale(-1.0)?)]
                }
                Op::Mul(a, b) => {
                    vec![(a, g.mul(self.value(b)?)?)]
                }
                Op::Scale(a, s) => todo!(),
                Op::AddRow(x, bias) => todo!(),
                Op::Relu(z) => todo!(),
                Op::T(a) => todo!(),
            };

            for (id, contribution) in contributions {
                self.accumulate(id, contribution)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::Gpu;
    use std::sync::Arc;

    // small graph that uses every op once:
    //   z   = x·w + b
    //   h   = relu(z)
    //   q   = h ⊙ h - 0.5h
    //   out = q·vᵀ + z
    // returns the tape, the leaf ids [x, w, b, v] and the id of out
    fn build(gpu: &Arc<Gpu>, p: &[Vec<f32>; 4]) -> Result<(MatTape, [usize; 4], usize)> {
        let mut tape = MatTape::new();
        let x = tape.leaf(Matrix::from_vec(gpu, 4, 3, &p[0])?);
        let w = tape.leaf(Matrix::from_vec(gpu, 3, 2, &p[1])?);
        let b = tape.leaf(Matrix::from_vec(gpu, 1, 2, &p[2])?);
        let v = tape.leaf(Matrix::from_vec(gpu, 2, 2, &p[3])?);

        let xw = tape.matmul(x, w)?;
        let z = tape.add_row(xw, b)?;
        let h = tape.relu(z)?;
        let hh = tape.mul(h, h)?;
        let half_h = tape.scale(h, 0.5)?;
        let q = tape.sub(hh, half_h)?;
        let vt = tape.t(v)?;
        let qv = tape.matmul(q, vt)?;
        let out = tape.add(qv, z)?;
        Ok((tape, [x, w, b, v], out))
    }

    // loss = sum of every element of out, so ∂loss/∂out is all ones
    fn loss(gpu: &Arc<Gpu>, p: &[Vec<f32>; 4]) -> Result<f32> {
        let (tape, _, out) = build(gpu, p)?;
        Ok(tape.value(out).to_vec()?.iter().sum())
    }

    // nudge every input number by ±eps and check the slope matches back_prop
    #[test]
    fn gradients_match_finite_differences() -> Result<()> {
        let gpu = Gpu::new(0)?;
        let params: [Vec<f32>; 4] = [
            vec![
                0.5, -1.2, 0.8, 1.5, 0.3, -0.7, -0.9, 1.1, 0.4, 0.2, -0.6, 1.3,
            ],
            vec![0.9, -0.4, 0.6, 1.1, -0.8, 0.5],
            vec![0.1, -0.2],
            vec![0.7, -0.3, 0.2, 0.9],
        ];

        let (mut tape, leaves, out) = build(&gpu, &params)?;
        let (rows, cols) = tape.value(out).shape();
        let ones = Matrix::from_vec(&gpu, rows, cols, &vec![1.0; rows * cols])?;
        tape.back_prop(out, ones)?;

        let eps = 1e-2;
        for (which, &leaf) in leaves.iter().enumerate() {
            let analytic = tape
                .grad(leaf)
                .expect("no gradient reached leaf")
                .to_vec()?;
            for k in 0..params[which].len() {
                let mut plus = params.clone();
                plus[which][k] += eps;
                let mut minus = params.clone();
                minus[which][k] -= eps;
                let numeric = (loss(&gpu, &plus)? - loss(&gpu, &minus)?) / (2.0 * eps);
                assert!(
                    (analytic[k] - numeric).abs() < 1e-2 * (1.0 + numeric.abs()),
                    "leaf {which} element {k}: back_prop says {}, finite difference says {numeric}",
                    analytic[k]
                );
            }
        }
        Ok(())
    }
}
