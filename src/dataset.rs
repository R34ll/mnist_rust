#![allow(dead_code)]

use std::fs::File;
use std::io::Read;

use crate::matrix::Matrix;

pub fn load_dataset(path: &str) -> Matrix {
    let mut file = File::open(path).expect("File don't was find");

    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();

    // let num_rows = 60000;
    // let num_cols = 784;

    let mut dataset_: Vec<Vec<u8>> = data.chunks(784).map(|chunk| chunk.to_vec()).collect();
    dataset_.pop();

    let mut dataset: Vec<u8> = dataset_.into_iter().flatten().collect();
    let new: Vec<f32> = dataset
        .iter_mut()
        .map(|&mut px| px as f32 / 255.0)
        .collect();

    // let batch_dataframe:Vec<Vec<f32>> = new
    // .clone()
    // .chunks(28*28)
    // .take(60000)
    // .map(|v|v.to_vec())
    // .collect();

    Matrix::new((60000, 784), new.as_slice())
}

pub fn load_label(path: &str) -> Vec<f32> {
    let mut file = File::open(path).expect("File don't was find");

    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    data.truncate(data.len() - 8);

    data.iter_mut().map(|&mut l| l as f32).collect()
}

pub fn data_batch(batch: usize, input_size: usize, data: Vec<f32>) -> Vec<Vec<f32>> {
    let len = data.len();
    let mut res = Vec::new();

    for i in 0..len {
        let inputs: Vec<Vec<f32>> = data
            .chunks(input_size)
            .skip(i * batch)
            .take(batch)
            .map(|chunk| chunk.to_vec())
            .collect();
        res.push(inputs.into_iter().flatten().collect());
    }
    res
}

pub fn one_hot_encode(labels: &[f32], num_classes: usize) -> Vec<Vec<f32>> {
    let mut new: Vec<Vec<f32>> = Vec::new();
    for mut label in labels.iter() {
        let mut new_label = vec![0.0; num_classes];
        if *label >= num_classes as f32 {
            label = &1.0;
        }
        new_label[*label as usize] = 1.0;

        new.append(&mut vec![new_label]);
    }
    new
}

pub fn one_hot_encode_matrix(labels: &[f32], num_classes: usize) -> Vec<Matrix> {
    let mut new: Vec<Matrix> = Vec::new();
    for mut label in labels.iter() {
        let mut new_label = vec![0.0; num_classes];
        if *label >= num_classes as f32 {
            label = &1.0;
        }
        new_label[*label as usize] = 1.0;
        let a = Matrix::new((1, num_classes), new_label.as_slice());
        new.append(&mut vec![a]);
    }
    new
}
