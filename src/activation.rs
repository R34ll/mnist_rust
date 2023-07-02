#![allow(dead_code)]

use crate::matrix::Matrix;


pub struct Softmax{
    pub last_input:Matrix
}

impl Softmax{
    #[inline]
    pub fn new()->Self{
        Self { last_input: Matrix::default() }
    }

    /// rescale input to range 0-1
    /// S(Zk) = e^Zk / SUM: e^Zi
    pub fn forward(&mut self, input: &Matrix)->Matrix{
        let e = input.exp();
        self.last_input = e.clone()/e.sum();
        self.last_input.clone()
    }

    pub fn backward(&mut self, grad:Matrix)->Matrix{
        let soft_out = &self.last_input;
        let soft_grad = soft_out.clone() * grad;

        let soft_deri = soft_grad.clone() - (soft_out.clone() * soft_grad.sum());
        let soft_sum = soft_out.sum();
        soft_deri / (soft_sum * soft_sum)

    }
}









