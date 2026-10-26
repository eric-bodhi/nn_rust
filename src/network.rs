use std::cmp::max;

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
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl Network {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let w1_limit: f32 = ((6.0 / 784.0) as f32).sqrt();
        let w1_dist = Uniform::new(-w1_limit, w1_limit).unwrap();

        let w2_limit: f32 = ((6.0 / 128.0) as f32).sqrt();
        let w2_dist = Uniform::new(-w2_limit, w2_limit).unwrap();

        Network {
            w1: Array2::random_using((128, 784), w1_dist, &mut rng),
            b1: Array1::zeros(128),
            w2: Array2::random_using((10, 128), w2_dist, &mut rng),
            b2: Array1::zeros(10),
        }
    }
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
    max(0_f32, number)
}
