use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;

// neural network main struct
// w1 is input layer weights to hidden layer
// w2 is hidden layer weights to output layer
// b1 is input layer biases to hidden layer
// b2 is hidden layer biases to output layer

// goal is to minimize loss function... all calculus
#[derive(Serialize, Deserialize)]
pub struct Network {
    weights1: Array2<f32>,
    bias1: Array1<f32>,
    weights2: Array2<f32>,
    bias2: Array1<f32>,
}

impl Network {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let w1_limit: f32 = ((6.0 / 784.0) as f32).sqrt();
        let w1_dist = Uniform::new(-w1_limit, w1_limit).unwrap();

        let w2_limit: f32 = ((6.0 / 128.0) as f32).sqrt();
        let w2_dist = Uniform::new(-w2_limit, w2_limit).unwrap();

        // every connection has its own weight
        // but bias are per neuron
        Network {
            weights1: Array2::random_using((128, 784), w1_dist, &mut rng),
            bias1: Array1::zeros(128),
            weights2: Array2::random_using((10, 128), w2_dist, &mut rng),
            bias2: Array1::zeros(10),
        }
    }

    // writes weights and biases to a json file
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    // reads a network back from a file written by save
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let reader = BufReader::new(File::open(path)?);
        Ok(serde_json::from_reader(reader)?)
    }

    pub fn forward(&self, image: Array1<f32>) -> Activations {
        let pre_activations = self.weights1.dot(&image) + &self.bias1;
        let post_activations = pre_activations.mapv(relu);
        let logits = self.weights2.dot(&post_activations) + &self.bias2;
        // logits are the neurons BEFORE softmax turns them into probabilities that sum to 1
        let probs = softmax(&logits);

        Activations {
            x: image,
            z1: pre_activations,
            a1: post_activations,
            p: probs,
        }
    }

    // adjusts weights
    // gradient descent update rule: w_new = curr_w - learning_rate * ∂L/∂w
    // or b_new = curr_b - learning_rate * ∂L/∂b
    pub fn update(&mut self, gradient: &Gradient, learning_rate: f32) {
        self.weights1
            .scaled_add(-learning_rate, &gradient.weights1_gradient);
        self.weights2
            .scaled_add(-learning_rate, &gradient.weights2_gradient);
        self.bias1
            .scaled_add(-learning_rate, &gradient.bias1_gradient);
        self.bias2
            .scaled_add(-learning_rate, &gradient.bias2_gradient);
    }

    // fucking wizard math
    pub fn back_prop(&self, cache: &Activations, label: u8) -> Gradient {
        // bias2 also equals logits_gradient because nudging bias2[digit] by 1
        // moves logit[digit] by 1, so ∂logit/∂b = 1
        // logits_gradient = p - one_hot(label)
        // where one_hot is the length 10 vector where every number is 0 but the label which is 1
        // i.e. [0,0,0,0,1,0...] label = 4
        let mut logits_gradient: Array1<f32> = Array1::zeros(10);

        for (index, item) in logits_gradient.iter_mut().enumerate() {
            if index == label as usize {
                *item = cache.p[index] - 1.0;
            } else {
                *item = cache.p[index];
            }
        }

        let mut weights2_gradient: Array2<f32> = Array2::zeros((10, 128));
        for ((row, col), val) in weights2_gradient.indexed_iter_mut() {
            *val = logits_gradient[row] * cache.a1[col];
        }

        let bias2_gradient = logits_gradient.clone();

        let mut post_activations_gradient: Array1<f32> = Array1::zeros(128);
        for (index, neuron) in post_activations_gradient.indexed_iter_mut() {
            let mut connections: Array1<f32> = Array1::zeros(10);
            for digit in 0..10 {
                connections[digit] = logits_gradient[digit] * self.weights2[(digit, index)]
            }
            *neuron = connections.sum();
        }

        let mut pre_activations_gradient: Array1<f32> = Array1::zeros(128);
        for (index, neuron) in pre_activations_gradient.indexed_iter_mut() {
            if cache.z1[index] > 0.0 {
                *neuron = post_activations_gradient[index];
            } else {
                *neuron = 0.0;
            }
        }

        let mut weights1_gradient: Array2<f32> = Array2::zeros((128, 784));
        for ((hidden_neuron, pixel), slot) in weights1_gradient.indexed_iter_mut() {
            *slot = pre_activations_gradient[hidden_neuron] * cache.x[pixel];
        }

        let bias1_gradient = pre_activations_gradient;
        Gradient {
            weights1_gradient,
            bias1_gradient,
            weights2_gradient,
            bias2_gradient,
        }
    }
}

// basically a cache from the forward propagation
pub struct Activations {
    x: Array1<f32>,     // input pixels [0.0, 1.0]
    z1: Array1<f32>,    // hidden pre activations
    a1: Array1<f32>,    // hidden post activations (relu)
    pub p: Array1<f32>, // output probabilities
}

impl Activations {
    // cross entropy loss function
    // highly correct = low loss
    // very wrong = big loss
    pub fn loss(&self, digit: u8) -> f32 {
        -(self.p[digit as usize].ln())
    }
}
// Sensitivity of weights: ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z*∂w
// gradient parameter tells you how sensitive the loss is to each parameter, i.e. weights, biases
// positive gradient value means turning knob up makes the loss go up so we should turn the knob down
// negative means turn the knob up
pub struct Gradient {
    weights1_gradient: Array2<f32>,
    bias1_gradient: Array1<f32>,
    weights2_gradient: Array2<f32>,
    bias2_gradient: Array1<f32>,
}

// rectified linear unit activation function
// fuck you??
fn relu(number: f32) -> f32 {
    number.max(0_f32)
}

// turns all logits into probabilities
// probability for a logit is e^logit / sum(e^logits)
// this turns negative logits into small positive numbers, and big positive logits into big positive
// numbers, and then normalizes them into a probability by dividing by the sum
fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exps = logits.mapv(|x| (x - max_logit).exp());
    let sum_exps = exps.sum();
    exps / sum_exps
}
