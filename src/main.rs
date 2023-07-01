use std::process::exit;

mod dataset;
mod activation;
mod layer;
mod matrix;
mod loss;


#[macro_use]
use matrix::{*,matrix::matrix};

use dataset::{load_dataset,load_label, one_hot_encode};
use layer::Layer;

use loss::CrossEntropyLoss;
use crate::activation::{Softmax, LeakyReLU};



const INPUT_SIZE:usize = 28*28; // 784
const OUTPUT_SIZE:usize = 10;
// const BATCH_SIZE:usize = 32;
const EPOCHS:u32 = 32;

fn main() {
    let labels_: Vec<f32> = load_label("data/train-labels-idx1-ubyte");
    let labels = one_hot_encode(&labels_, OUTPUT_SIZE);
    let dataframe: Matrix = load_dataset("data/train-images-idx3-ubyte");

    let mut layer1 = Layer::new(INPUT_SIZE, 100);
    let mut act = Softmax::new(); // Softmax is returning inf
    let mut layer2 = Layer::new(100,OUTPUT_SIZE);
    let mut loss = CrossEntropyLoss::new();




    // Training
    println!("Initializing training...");
    for epoch in 1..EPOCHS{

        let mut total_loss = 0.0;

        for idx in 0..dataframe.shape().0{
            let xi = dataframe.get_row(idx);
            let yi = Matrix::new((1,OUTPUT_SIZE),&labels[idx]);
            

            let output_layer_1 = layer1.forward(xi);
            let output_act = act.forward(&output_layer_1);
            let output_layer_2 = layer2.forward(output_act);

            loss.forward(&output_layer_2, &yi);


            let grads1 = layer2.backward(loss.input_grads.clone());
            let grads2 = act.backward(grads1.inputs_grad);
            let grads3 = layer1.backward(grads2);

            layer1.apply_gradient(grads3.weight_grads);
            layer2.apply_gradient(grads1.weight_grads);

            // println!("Epoch: {} | Idx: {} | Loss: {:?}",_epoch,idx,loss.loss);
        
            total_loss+=loss.loss.data()[0];
        }

        println!("Epoch: {} | Loss: {:?}",epoch,total_loss);
    }




    for idx in 0..15{
        let xi = dataframe.get_row(idx);
        let yi = Matrix::new((1,OUTPUT_SIZE),&labels[idx]);
        

        let output_layer_1 = layer1.forward(xi);
        let output_act = act.forward(&output_layer_1);
        let output_layer_2 = layer2.forward(output_act);
    
        println!("\nModel: {:?}\nReal:  {:?}\n
        {:?}
        ",output_layer_2.max_index(), yi.max_index(), output_layer_2);

    }




}



