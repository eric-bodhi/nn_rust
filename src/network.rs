use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::Uniform;

mod visualize;

// neural network main struct
// w1 is input layer weights to hidden layer
// w2 is hidden layer weights to output layer
// b1 is input layer biases to hidden layer
// b2 is hidden layer biases to output layer

// goal is to minimize loss function... all calculus
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
        let mut logits_gradient: Array1<f32> = Array1::zeros(10);

        let mut index: usize = 0;
        for prob in &cache.p {
            if index as u8 == label {
                logits_gradient[index] = *prob - 1.0;
            } else {
                logits_gradient[index] = *prob;
            }
            index += 1;
        }

        let bias2_gradient: Array1<f32> = logits_gradient.clone();

        let mut weights2_gradient: Array2<f32> = Array2::zeros((10, 128));
        for ((digit, hidden_neuron), slot) in weights2_gradient.indexed_iter_mut() {
            *slot = logits_gradient[digit] * cache.a1[hidden_neuron];
        }

        let mut hidden_post_activation_gradient: Array1<f32> = Array1::zeros(128);
        for j in 0..128 {
            let mut temp: Array1<f32> = Array1::zeros(10);
            for digit in 0..10 {
                temp[digit] = self.weights2[(digit, j)] * logits_gradient[digit];
            }
            hidden_post_activation_gradient[j] = temp.sum();
        }

        let mut hidden_pre_activation_gradient: Array1<f32> = Array1::zeros(128);
        for (index, val) in cache.a1.indexed_iter() {
            if *val > 0.0 {
                hidden_pre_activation_gradient[index] = hidden_post_activation_gradient[index];
            } else {
                hidden_pre_activation_gradient[index] = 0.0;
            }
        }

        let mut weights1_gradient: Array2<f32> = Array2::zeros((128, 784));
        for ((hidden_neuron, input_pixel), slot) in weights1_gradient.indexed_iter_mut() {
            *slot = hidden_pre_activation_gradient[hidden_neuron] * cache.x[input_pixel];
        }

        let bias1_gradient: Array1<f32> = hidden_pre_activation_gradient;

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
    x: Array1<f32>,  // input pixels [0.0, 1.0]
    z1: Array1<f32>, // hidden pre activations
    a1: Array1<f32>, // hidden post activations (relu)
    p: Array1<f32>,  // output probabilities
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

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exps = logits.mapv(|x| (x - max_logit).exp());
    let sum_exps = exps.sum();
    exps / sum_exps
}
