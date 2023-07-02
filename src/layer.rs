use rand::{distributions::Uniform, Rng};
use std::fmt::Debug;

use crate::matrix::Matrix;

pub struct LayerGrads {
    pub weight_grads: Matrix,
    pub inputs_grad: Matrix,
}

pub struct Layer {
    pub weights: Matrix,
    last_inputs: Matrix,
    input_size: usize,
    output_size: usize,
}

impl Layer {
    pub fn new(input: usize, output: usize) -> Self {
        // Xavier initialization
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (input + output) as f32).sqrt();
        let between = Uniform::new(0.0, limit);

        let mut v_weight = Vec::new();
        for _ in 0..input * output {
            v_weight.push(rng.sample(between));
        }

        Self {
            // weights:Matrix::new_from((input,output),0.001),
            weights: Matrix::new((input, output), v_weight.as_slice()),
            last_inputs: Matrix::new_zeros((input, output)),
            input_size: input,
            output_size: output,
        }
    }

    pub fn forward(&mut self, inputs: Matrix) -> Matrix {
        self.last_inputs = inputs.clone();
        inputs.dot(&self.weights)
    }

    pub fn backward(&mut self, grads: &Matrix) -> LayerGrads {
        LayerGrads {
            weight_grads: self.last_inputs.transpose().dot(grads),
            inputs_grad: grads.dot(&self.weights.transpose()),
        }
    }

    #[inline]
    pub fn apply_gradient(&mut self, grads: &Matrix) {
        self.weights
            .into_iter()
            .zip(grads.iter())
            .for_each(|(w, grad)| *w -= 0.0009 * grad);
    }
}

impl Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layer(input_size={}, output_size={})",
            self.input_size, self.output_size
        )
    }
}
