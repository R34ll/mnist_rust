use std::fmt::Debug;

use crate::matrix::Matrix;

struct LayerGrads{
    weight_grads:Matrix,
    inputs_grad:Matrix,

}

#[derive(Clone)]
pub struct Layer{
    weights:Matrix,
    last_inputs:Matrix,
    input_size:usize,
    output_size:usize,
}

impl Layer{
    pub fn new(input:usize,output:usize)->Self{
        Self{
            weights:Matrix::new_ones((input,output)),
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
        let inputs = &self.last_inputs;
        
        let weight_grads = inputs.transpose().dot(&grads);
        let inputs_grad = grads.dot(&self.weights.transpose());

        LayerGrads { weight_grads, inputs_grad }

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





// #[cfg(test)]
// mod test{
//     use super::Layer;
//     const INPUT_SIZE:u32=28*28;

//     #[test]
//     fn forward_test(){

//         let test_input = (0..INPUT_SIZE).map(|z| z as f32).collect::<Vec<f32>>();

//         let mut layer1 = Layer::new(INPUT_SIZE as usize,1);
//         let res = layer1.forward(test_input.clone());

//         let python_pytorch_result = 3404738736.0;

//         // assert_eq!(res.iter().sum::<f32>(), python_pytorch_result);
//         assert_eq!(res,vec![0.5]);
//     }
// }


// // pub struct LayerGrads{
// //     pub weight_grads:Vec<f32>,
// //     pub input_grads:Vec<f32>,

// // }


// // pub struct Layer{
    
// //     weights:Array2<f32>,
// //     last_inputs:Array2<f32>,
// // }

// // impl Layer{
// //     pub fn new(weight_shape:(usize,usize))->Self{
// //         Self{
// //             // weights:vec![0.0;I*O],
// //             weights: Array2::zeros(weight_shape),
// //             last_inputs:Array2::zeros(weight_shape),
// //         }
// //     }

// //     pub fn forward(&mut self, inputs:&Array2<f32>)->Array2<f32>{
// //         // let batch_size = inputs.len() / self.I;
// //         // let mut outputs = vec![0.0; batch_size*self.O];

// //         // for b in 0..batch_size{
// //         //     for o in 0..self.O{
// //         //         let mut sum = 0.0;

// //         //         for i in 0..self.I{
// //         //             sum+=inputs[b*self.I+i] * self.weights[self.O * i + o];
// //         //         }
// //         //         outputs[b*self.O+0] = sum;
// //         //     }
//         // }
//         // self.last_inputs = inputs;
//         // return outputs;

//     }


//     pub fn backwards(&mut self, grads:Vec<f32>)->LayerGrads{
//         // let mut weight_grads = vec![0.0;self.I*self.O];

//         // let batch_size = self.last_inputs.len() / self.I;
//         // let mut input_grads = vec![0.0;batch_size*self.I];

//         // for b in 0..batch_size{
//         //     for i in 0..self.I{
//         //         for o in 0..self.O{
//         //             weight_grads[i*self.O+o] += (grads[b * self.O + o] * self.last_inputs[b * self.I + i]) / batch_size as f32;
//         //             input_grads[b*self.I+i] += grads[b*self.O+o] * self.weights[i * self.O +o];

//         //         }
//         //     }
//         // }
//         // return LayerGrads{
//         //     weight_grads,
//         //     input_grads
//         // }

//         todo!()


//     }

//     // pub fn applyGradients(&mut self, grads:Vec<f32>) {
//     //     for i in 0..self.I*self.O{
//     //         self.weights[i] -= 0.01 * grads[i];
//     //     }
//     // }
// }


// #[cfg(test)]
// mod test{

//     use ndarray::arr2;

//     use super::*;

//     #[test]
//     pub fn one(){

//         let data =  arr2(&[[1., 2., 3.],
//             [4., 5., 6.],
//             [7.,8.,9.0],
//             [10.0,11.0,12.]
//             ]);

//         let data_shape = data.shape();

            
//         let mut layer = Layer::new((data_shape[0],data_shape[1]));
//         let out = layer.forward(&data);

//     }

// }



































