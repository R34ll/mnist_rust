// use nnlib::{, activations};

mod dataset;
mod activation;
mod layer;
mod matrix;
mod nll;
mod loss;


use dataset::{load_dataset,load_label, data_batch};
use layer::Layer;
use matrix::Matrix;
use activation::LeakyReLU;
use nll::NLL;
use loss::CrossEntropyLoss;

const INPUT_SIZE:u32 = 28*28; // 784
const OUTPUT_SIZE:u32 = 10;
const BATCH_SIZE:usize = 32;
const EPOCHS:u32 = 25;

fn main() {
    // let path_dataset = "data/train-images-idx3-ubyte";
    // let path_label = "data/train-labels-idx1-ubyte";
    // let label = load_label(path_label);
    // let dataframe = load_dataset(path_dataset);


    // let batch:Vec<_> = dataframe // Vec<Matrix>
    // .chunks(28*28)
    // .take(60000)
    // .map(|v|v.to_vec())
    // .collect();

    let data_ = (0..INPUT_SIZE).map(|z| z as f32).collect::<Vec<f32>>();
    let data =  Matrix::new((28,28), &data_);



    let mut loss_fn = CrossEntropyLoss::new();

    // let data = Matrix::new((3,4),&[0.1000, 0.4000, 0.2000, 0.3000, 0.7000, 0.1000, 0.0500, 0.1500, 0.2000,
    // 0.2000, 0.4000, 0.2000]);


    let data = Matrix::new((3,1),&[2.,0.0,1.0]);

    let labels = Matrix::new((1,3), &[2.0,0.0,1.0]);

    let loss = loss_fn.forward(&data, &labels);
    
    println!(">> {:?}",loss);
    


    // println!("{:?}",loss.loss.iter().sum::<f32>());



    // // Perform the forward pass
    // network.forward(&input, &target);

    // // Print the output and loss
    // println!("Rust Output:");
    // println!("{:?}", network.input_grads);
    // println!("Rust Loss:");
    // println!("{:?}", network.loss.data());
































    // let mut relu = ReLU::new();
    // let mut layer2 = Layer::new(100, OUTPUT_SIZE as usize);

    // let shape = dataset.shape().;

    // let mut layer = Layer::new((shape[0],shape[1]));
    // let out = layer.forward(data);
    // println!("{:?}",out);






}

/*
    Dataset:
    - Mnist numbers from 0 to 9
    - Each number we set N/255.0(turning in 0 or 1 the values)
    - vec[ // 60.000 images
            vec[28x28=784] // 784
        ]



    Layer:
    - Struct:
    -- new() Initialize the struct
    -- dot() ndarray Matrix Multiplication
    -- forward() Forward data thought layer weights doin'g matrix multiplication and storing inputs
    -- backward() Use the stored inputs and gradients input to calculate the backward of the data



 */


