// Leaky ReLU

use crate::matrix::Matrix;

pub struct LeakyReLU{
    pub last_input:Matrix
}

impl LeakyReLU{
    /// Initialize Our Leaky ReLu. Disponible methods now
    pub fn new()->Self{
        Self{
            last_input:Matrix::default()
        }
    }

    /// f(x) = max(x*0.01,0) = {if x < 0 return x*0.01; if x > 0 return x}
    pub fn forward(&mut self, input:Matrix)->Matrix{
        let shape = input.shape();

        let output_: Vec<f32> = input.iter().map(|&z| if z < 0.0 { 0.01 * z } else { z }).collect();
        let output: Matrix = Matrix::new(shape, &output_);
        self.last_input = output.clone();
        output

    }

    // TODO: is not corret
    pub fn backward(&mut self, grads: Matrix) -> Matrix {
        let shape = grads.shape();
        let mut output: Vec<f32> = Vec::with_capacity(shape.0 * shape.1);

        for (&grad, &input) in grads.iter().zip(self.last_input.iter()) {
            let grad_output = if input < 0.0 { 0.01  *grad } else { grad };
            output.push(grad_output);
        }

        Matrix::new(shape, &output)
    }
}




