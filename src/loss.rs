#![allow(dead_code)]

use crate::matrix::Matrix;
use crate::Softmax;
use crate::matrix::matrix::matrix;

const EULER:f32 = std::f32::consts::E;


pub struct CrossEntropyLoss{
    pub loss:Matrix,
    pub input_grads:Matrix
}
impl CrossEntropyLoss{
    pub fn new()->Self{
        Self{
            loss:Matrix::default(),
            input_grads:Matrix::default()
        }
    }


    pub fn forward(&mut self, inputs: &Matrix, labels: &Matrix){ // WOrking: cloned from tensorflow example. the outputs are a little diferent, but the loss diminue when whe are closer to real value/target/label
        let epsilon = 1e-8; // avoid taking the logarithm to zero
        let mut grads = Matrix::new_zeros(inputs.shape());


        let mut result = 0.0;

        for ((input,label),grad) in inputs.iter().zip(labels.iter()).zip(grads.into_iter()){
            let p = input.max(epsilon).min(1.0-epsilon);
            *grad = p - label; 
            result-= label * p.ln();
        }

        self.input_grads = grads;
        self.loss = matrix![[result]]; // inputs.len() as f32;



    }

}



    /*  Cross Entropy Loss function
     E(y,S(Z)) = -SUM: yi * ln(S(Zi))
     "ln" is the natural logarithm - euler number
     "S(Z)" is output neurons. ex: if we have 3 outputs neurons: S(Z) = [S(Z1), S(Z2),S(Z3)] | the S is of Softmax and Z is the input
     E(y,S(Z)) = -(yi*log(S(Zi)) + y2 * log(S(Z2)) + y3 * log(S(Z3)) )
    */
