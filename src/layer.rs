#![allow(dead_code)]
use std::fmt::Debug;
use crate::matrix::Matrix;

#[derive(Debug)]
pub struct LayerGrads{
    pub weight_grads:Matrix,
    pub inputs_grad:Matrix,
}

#[derive(Clone)]
pub struct Layer{
    pub weights:Matrix,
    last_inputs:Matrix,
    input_size:usize,
    output_size:usize,
}

impl Layer{
    pub fn new(input:usize,output:usize)->Self{
        Self{
            weights:Matrix::new_from((input,output),0.01), 
            last_inputs:Matrix::new_zeros((input,output)),
            input_size:input,
            output_size:output
        }
    }

    pub fn forward(&mut self, inputs:Matrix)->Matrix{
        self.last_inputs = inputs.clone();
        inputs.dot(&self.weights)
    }

    pub fn backward(&mut self,grads:Matrix)->LayerGrads{
        LayerGrads { 
            weight_grads: self.last_inputs.transpose().dot(&grads), 
            inputs_grad: grads.dot(&self.weights.transpose())
        }
    }

    pub fn apply_gradient(&mut self,grads:Matrix){
        self.weights.into_iter().zip(grads.iter()).for_each(|(w,grad)| {*w -= 0.001 * grad});
    }

    pub fn shape(&self)->(usize,usize){
        (self.input_size, self.output_size)
    }

    pub fn weights(self)->Matrix{
        self.weights
    }

}


impl Debug for Layer {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer(input_size={}, output_size={})",self.input_size,self.output_size)
    }    
}



