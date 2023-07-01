#![allow(dead_code)]

use std::fs::File;
use std::io::Read;

use crate::matrix::Matrix;

pub fn load_dataset(path:&str)->Matrix{
    let mut file = File::open(path).expect("Faile don't was find");

    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();

    // let num_rows = 60000;
    // let num_cols = 784;

    let mut dataset_:Vec<Vec<u8>> = data
        .chunks(784)
        .map(|chunk| chunk.to_vec())
        .collect();
    dataset_.pop();

    let mut dataset:Vec<u8> = dataset_.into_iter().flatten().collect();
    let new:Vec<f32> = dataset.iter_mut().map(|&mut px| px as f32 / 255.0).collect();


    // let batch_dataframe:Vec<Vec<f32>> = new
    // .clone()
    // .chunks(28*28)
    // .take(60000)
    // .map(|v|v.to_vec())
    // .collect();

    Matrix::new((60000,784),new.as_slice())

}

pub fn load_label(path:&str)->Vec<f32>{
    let mut file = File::open(path).expect("Faile don't was find");

    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    data.truncate(data.len()-8);

    data.iter_mut().map(|&mut l| l as f32).collect()
}


pub fn data_batch(batch:usize,input_size:usize, data:Vec<f32>)->Vec<Vec<f32>>{
    let len = data.len();
    let mut res = Vec::new();

    for i in 0..len{
        let inputs:Vec<Vec<f32>> = data
            .chunks(input_size as usize)
            .skip((i * batch) as usize)
            .take(batch as usize)
            .map(|chunk| chunk.to_vec())
            .collect();
        res.push(inputs.into_iter().flatten().collect());
    }
    return res
}


pub fn one_hot_encode(labels: &[f32], num_classes: usize) -> Vec<Vec<f32>> {
    let mut new:Vec<Vec<f32>> = Vec::new();
    for mut label in labels.into_iter(){
        let mut new_label = vec![0.0;num_classes];
        if *label >= num_classes as f32{label = &1.0;}
        new_label[*label as usize] = 1.0;

        new.append(&mut vec![new_label]);
    }
    new
}




















































// ###################################################################################################################

// use ndarray::{arr2, Array2, Axis, Array1, ArrayView, Ix2};

// use std::fs::{read, File};
// use std::io::{self,Read};



// pub fn load_dataset(path:&str)->Array2<u8>{

//     let mut file = File::open(path).expect("Faile don't was find");

//     let mut data = Vec::new();
//     file.read_to_end(&mut data).unwrap();

//     // let data:Vec<u8> = data.iter_mut().map(|&mut px| px/255 ).collect(); // Normalization

//     let num_rows = 60000;
//     let num_cols = 784;

//     let mut dataset_:Vec<Vec<u8>> = data
//         .chunks(784)
//         .map(|chunk| chunk.to_vec())
//         .collect();
//     dataset_.pop();



//     let dataset = Array2::from_shape_vec((num_rows, num_cols), dataset_.into_iter().flatten().collect()).unwrap();



//     dataset
// }

// pub fn load_label(path:&str)->Array2<u8>{
//     let mut file = File::open(path).expect("Faile don't was find");

//     let mut data = Vec::new();
//     file.read_to_end(&mut data).unwrap();
//     data.truncate(data.len()-8);


//     let labels = Array2::<u8>::from_shape_vec((60000,1), data);

//     labels.unwrap()

// }

















