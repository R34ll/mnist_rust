mod dataset;
mod matrix;

mod layer;
mod activation;
mod loss;

use matrix::Matrix;
use dataset::{load_dataset,load_label,one_hot_encode_matrix};
use layer::Layer;
use loss::CrossEntropyLoss;
use activation::Softmax;


const INPUT_SIZE:usize = 28*28; // 784
const OUTPUT_SIZE:usize = 10;
// const BATCH_SIZE:usize = 32;
const EPOCHS:u32 = 3;

fn main() {
    let labels_: Vec<f32> = load_label("data/train-labels-idx1-ubyte");
    let labels = one_hot_encode_matrix(&labels_, OUTPUT_SIZE);
    let dataframe: Matrix = load_dataset("data/train-images-idx3-ubyte");


    let mut layer1 = Layer::new(INPUT_SIZE, 100);
    let mut act = Softmax::new(); 
    let mut layer2 = Layer::new(100,OUTPUT_SIZE);
    let mut loss = CrossEntropyLoss::new();

    // Training
    println!("Initializing training...");
    for epoch in 0..EPOCHS{

        let mut total_loss = 0.0;
        for (xi,yi) in dataframe.iter_row().zip(labels.iter()){

            let output_layer_1 = layer1.forward(xi);
            let output_act = act.forward(&output_layer_1);
            let output_layer_2 = layer2.forward(output_act);

            loss.forward(&output_layer_2, &yi);

            let grads1 = layer2.backward(&loss.input_grads);
            let grads2 = act.backward(grads1.inputs_grad);
            let grads3 = layer1.backward(&grads2);

            layer1.apply_gradient(&grads3.weight_grads);
            layer2.apply_gradient(&grads1.weight_grads);

        
            total_loss+=loss.input_grads.data()[0];
        }

        println!("Epoch: {} | Loss: {:?}",epoch,total_loss);
    }



    let mut corrects = 0.0;
    for (xi,yi) in dataframe.iter_row().zip(labels.iter()){

        let output_layer_1 = layer1.forward(xi);
        let output_act = act.forward(&output_layer_1);
        let output_layer_2 = layer2.forward(output_act);

        if output_layer_2.max_index() == yi.max_index(){
            corrects+=1.0;
        }
    }


    println!("Corrects: {}/{}",corrects,labels.len());
}

















