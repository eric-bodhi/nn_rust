use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::Uniform;

use crate::error::Result;
use crate::gpu::Gpu;
use crate::idx::Images;
use crate::mat_autograd::MatTape;
use crate::matrix::Matrix;

// same 784 -> 128 -> 10 network as network.rs, but trained on the GPU in batches
const INPUT: usize = 784;
const HIDDEN: usize = 128;
const OUTPUT: usize = 10;
const BATCH: usize = 100; // divides both 60000 and 10000, so no leftover batch
const EPOCHS: usize = 5;

// weights are [in x out] here, the transpose of network.rs, because each image
// is a row of X and a layer is X·W instead of W·x
pub struct Params {
    w1: Matrix, // [784 x 128]
    b1: Matrix, // [1 x 128]
    w2: Matrix, // [128 x 10]
    b2: Matrix, // [1 x 10]
}

// uniform in ±sqrt(6 / fan_in), same init as Network::new
fn random_matrix(gpu: &Arc<Gpu>, rows: usize, cols: usize, rng: &mut StdRng) -> Result<Matrix> {
    let limit = (6.0 / rows as f32).sqrt();
    let dist = Uniform::new(-limit, limit).unwrap();
    let values: Vec<f32> = (0..rows * cols).map(|_| rng.sample(dist)).collect();
    Matrix::from_vec(gpu, rows, cols, &values)
}

impl Params {
    pub fn new(gpu: &Arc<Gpu>, rng: &mut StdRng) -> Result<Self> {
        Ok(Self {
            w1: random_matrix(gpu, INPUT, HIDDEN, rng)?,
            b1: Matrix::zeros(gpu, 1, HIDDEN)?,
            w2: random_matrix(gpu, HIDDEN, OUTPUT, rng)?,
            b2: Matrix::zeros(gpu, 1, OUTPUT)?,
        })
    }

    // forward pass without a tape, for evaluation: probabilities [batch x 10]
    fn predict(&self, x: &Matrix) -> Result<Matrix> {
        x.matmul(&self.w1)?
            .add_row(&self.b1)?
            .relu()?
            .matmul(&self.w2)?
            .add_row(&self.b2)?
            .softmax_rows()
    }
}

// every pixel of every image, [0, 255] -> [0.0, 1.0], one image after another
fn normalize_all(images: &Images) -> Vec<f32> {
    images.pixels.iter().map(|&p| p as f32 / 255.0).collect()
}

// uploads the images and one-hot labels at `ids` as x [batch x 784], y [batch x 10]
fn make_batch(
    gpu: &Arc<Gpu>,
    pixels: &[f32],
    labels: &[u8],
    ids: &[usize],
) -> Result<(Matrix, Matrix)> {
    let mut x = Vec::with_capacity(ids.len() * INPUT);
    let mut y = vec![0.0; ids.len() * OUTPUT];
    for (row, &id) in ids.iter().enumerate() {
        x.extend_from_slice(&pixels[id * INPUT..(id + 1) * INPUT]);
        y[row * OUTPUT + labels[id] as usize] = 1.0;
    }
    Ok((
        Matrix::from_vec(gpu, ids.len(), INPUT, &x)?,
        Matrix::from_vec(gpu, ids.len(), OUTPUT, &y)?,
    ))
}

// summed cross entropy loss and number of correct guesses for a batch,
// from probabilities copied back to the CPU
fn score(p: &[f32], labels: &[u8], ids: &[usize]) -> (f32, usize) {
    let mut loss = 0.0;
    let mut correct = 0;
    for (row, &id) in ids.iter().enumerate() {
        let probs = &p[row * OUTPUT..(row + 1) * OUTPUT];
        let label = labels[id] as usize;
        loss -= probs[label].max(1e-12).ln();
        let guess = (0..OUTPUT)
            .max_by(|&a, &b| probs[a].total_cmp(&probs[b]))
            .unwrap();
        if guess == label {
            correct += 1;
        }
    }
    (loss, correct)
}

// stochastic gradient descent
// w - lr * grad
fn sgd(w: Matrix, grad: Option<Matrix>, lr: f32) -> Result<Matrix> {
    let grad = grad.expect("no gradient reached a weight");
    w.sub(&grad.scale(lr)?)
}

// one forward + backward + update on a batch. takes the params by value because
// they go onto the tape as leaves, and hands back the updated ones plus the
// probabilities so the caller can score the batch
fn train_step(params: Params, x: Matrix, y: &Matrix, lr: f32) -> Result<(Params, Matrix)> {
    // forward
    let mut tape = MatTape::new();
    let x = tape.leaf(x);
    let w1 = tape.leaf(params.w1);
    let b1 = tape.leaf(params.b1);
    let w2 = tape.leaf(params.w2);
    let b2 = tape.leaf(params.b2);

    let xw1 = tape.matmul(x, w1)?;
    let z1 = tape.add_row(xw1, b1)?;
    let a1 = tape.relu(z1)?;
    let a1w2 = tape.matmul(a1, w2)?;
    let logits = tape.add_row(a1w2, b2)?;

    // softmax + cross entropy stays off the tape: ∂L/∂logits = (p - onehot) / batch
    let p = tape.value(logits).softmax_rows()?;
    let seed = p.sub(y)?.scale(1.0 / BATCH as f32)?;
    tape.back_prop(logits, seed)?;

    // update
    let mut leaves = tape.into_leaves(&[w1, b1, w2, b2]).into_iter();
    let mut next = || {
        let (w, grad) = leaves.next().unwrap();
        sgd(w, grad, lr)
    };
    let params = Params {
        w1: next()?,
        b1: next()?,
        w2: next()?,
        b2: next()?,
    };
    Ok((params, p))
}

// accuracy on a whole dataset, forward pass only
fn evaluate(gpu: &Arc<Gpu>, params: &Params, pixels: &[f32], labels: &[u8]) -> Result<f32> {
    let ids: Vec<usize> = (0..labels.len()).collect();
    let mut correct = 0;
    for batch in ids.chunks(BATCH) {
        let (x, _) = make_batch(gpu, pixels, labels, batch)?;
        let p = params.predict(&x)?.to_vec()?;
        correct += score(&p, labels, batch).1;
    }
    Ok(correct as f32 / labels.len() as f32)
}

pub fn train(
    train_images: &Images,
    train_labels: &[u8],
    test_images: &Images,
    test_labels: &[u8],
) -> Result<Params> {
    let gpu = Gpu::new(0)?;
    let mut rng = StdRng::seed_from_u64(42);
    let mut params = Params::new(&gpu, &mut rng)?;

    let train_pixels = normalize_all(train_images);
    let test_pixels = normalize_all(test_images);

    let mut order: Vec<usize> = (0..train_labels.len()).collect();
    let mut lr = 0.1;
    for epoch in 0..EPOCHS {
        order.shuffle(&mut rng);
        let (mut loss, mut correct, mut seen) = (0.0, 0, 0);

        for (step, batch) in order.chunks(BATCH).enumerate() {
            let (x, y) = make_batch(&gpu, &train_pixels, train_labels, batch)?;
            let p;
            (params, p) = train_step(params, x, &y, lr)?;

            // copying back to the CPU waits for the GPU, so only score now and then
            if step % 100 == 0 {
                let (l, c) = score(&p.to_vec()?, train_labels, batch);
                loss += l;
                correct += c;
                seen += batch.len();
            }
        }

        let test_accuracy = evaluate(&gpu, &params, &test_pixels, test_labels)?;
        println!(
            "epoch {epoch}: train loss {:.4}, train acc {:.2}%, test acc {:.2}%",
            loss / seen as f32,
            correct as f32 / seen as f32 * 100.0,
            test_accuracy * 100.0
        );
        lr /= 2.0;
    }
    Ok(params)
}
