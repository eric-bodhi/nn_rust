use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::Uniform;

// neural network main struct
// w1 is input layer weights to hidden layer
// w2 is hidden layer weights to output layer
// b1 is input layer biases to hidden layer
// b2 is hidden layer biases to output layer

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
        let probs = softmax(&logits);

        Activations {
            x: image,
            z1: pre_activations,
            a1: post_activations,
            p: probs,
        }
    }

    // adjusts weights
    //pub fn update(&mut self, gradient: Gradient) {}

    pub fn back_prop(&self, cache: Activations, label: u8) -> Gradient {}
}

pub struct Activations {
    x: Array1<f32>,  // input pixels [0.0, 1.0]
    z1: Array1<f32>, // hidden pre activations
    a1: Array1<f32>, // hidden post activations (relu)
    p: Array1<f32>,  // output probabilities
}

pub struct Gradient {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

// rectified linear unit activation function
// fuck you??
fn relu(number: f32) -> f32 {
    number.max(0_f32)
}

// cross entropy loss function
// highly correct = low loss
// very wrong = big loss
fn loss(p: f32) -> f32 {
    -p.ln()
}

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max_logit = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exps = logits.mapv(|x| (x - max_logit).exp());
    let sum_exps = exps.sum();
    exps / sum_exps
}
