use ndarray::prelude::*;
use std::error::Error;

pub struct Layer {
    input_shape: usize,
    output_shape: usize,
    weights: Array2::<f32>,
}

impl Layer {
    pub fn new(input_shape: usize, output_shape: usize) -> Self {
        Self {
            input_shape,
            output_shape,
            weights: Array2::<f32>::zeros((output_shape, input_shape+1))
        }
    }
    pub fn forward(&self, input: &Vec<f32>)  -> Result<Array1::<f32>, Box<dyn Error>> {
        let mut i = input.to_vec();
        i.push(1.);
        let input_bias: Array1::<f32> = Array::from_shape_vec(self.input_shape+1, i)?;
        let output: Array1::<f32> = self.weights.dot(&input_bias);
        println!("weights: {}", self.weights);
        return Ok(output);
    }
}

fn main() -> Result<(), Box<dyn Error>>{
    //(0,0) = 0
    //(1,0) = 1
    //(0,1) = 1
    //(1,1) = 0
    //xor example
    let layer: Layer = Layer::new(2, 1);
    let input: Vec<f32> = vec![1., 1.];
    let output: Array1::<f32> = layer.forward(&input)?;
    println!("{:?} -> {}", input, output);
    /*
    let a = arr2(&[[1, 2, 3],
                   [4, 5, 6]]);

    let b = arr2(&[[6, 5, 4],
                   [3, 2, 1]]);

    let sum = &a + &b;

    println!("{}", a);
    println!("+");
    println!("{}", b);
    println!("=");
    println!("{}", sum);
    */
    Ok(())
}

//fn feed_forward(input:Array1::<f32>)