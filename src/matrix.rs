use std::ops::{Sub, Div, Mul};

// #[macro_export]
pub mod matrix{
    macro_rules! matrix{
        ($($elem:expr),*) => {
            {
                let mut data: Vec<Vec<f32>> = Vec::new();
                $(data.push($elem.to_vec());)*
                Matrix::new((data.len(),data[0].len()),data.concat().as_slice())
            }
        };
    }
    pub(crate) use matrix;
}


#[derive(PartialEq,Clone)]
pub struct Matrix{
    pub rows:usize,
    pub(crate) cols:usize,
    data:Vec<f32>
}

impl Matrix{

    /// Creates a new matrix with the specified number of rows and columns.
    ///
    /// # Arguments
    ///
    /// * `(usize,_)` - The number of rows in the matrix.
    /// * `(_,usize)` - The number of columns in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = Matrix::new((3, 2),&[1.0,2.0,
    ///                                   3.0,4.0,
    ///                                   5.0,6.0]);
    /// ```
    #[inline]
    pub fn new(shape:(usize,usize), data:&[f32])->Self{
        assert_eq!(shape.0*shape.1, data.len(), "Shape don't match with data size.");
        Self{
            rows:shape.0,
            cols:shape.1,
            data:data.to_vec(),
        }
    }
    #[inline]
    pub fn new_zeros(shape:(usize,usize))->Self{
        Self { rows: shape.0, cols: shape.1, data: vec![0.0;shape.0*shape.1] }
    }


    #[inline]
    pub fn new_ones(shape:(usize,usize))->Self{
        Self { rows: shape.0, cols: shape.1, data: vec![1.0;shape.0*shape.1] }
    }

    pub fn new_from(shape:(usize,usize), num:f32)->Self{
        Self { rows: shape.0, cols: shape.1, data: vec![num;shape.0*shape.1] }
    }

    #[inline]
    pub fn shape(&self)->(usize,usize){
        (self.rows,self.cols)
    }
    #[inline]
    pub fn len(&self)->usize{
        self.data.len()
    }
    
    pub fn data(&self)->Vec<f32>{
        self.data.clone()
    }

    pub fn get_row(&self, row:usize)->Matrix{
        let start = row * self.shape().1;
        let end = start + self.shape().1;
        let row_data = &self.data[start..end];
        Matrix::new((1,self.cols),row_data)
    }

    pub fn get(&self, row:usize,col:usize)->f32{
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row:usize,col:usize,value:f32){
        self.data[row*self.cols+col] = value
    }

    pub fn iter(&self) -> std::slice::Iter<f32>{
        self.data.iter()
    }


    /// Returns a new matrix with transposed dimensions.
    ///
    /// # Returns
    ///
    /// A new matrix with the number of rows equal to the original number of columns,
    /// and the number of columns equal to the original number of rows. The data is a   
    /// separate copy of the original matrix data.
    ///
    /// # Example
    ///
    /// ```
    /// let matrix = Matrix::new(3, 4);
    /// let transposed = matrix.new_transpose();
    /// ```
    pub fn transpose(&self)->Self{
        let mut transposed_data = vec![0.0; self.rows * self.cols];

        for i in 0..self.rows {
            for j in 0..self.cols {
                let index = j * self.rows + i;
                transposed_data[index] = self.data[i * self.cols + j];
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: transposed_data,
        }
    }

    /// Dot operation
    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows,"Incompatible shapes for dot operation. A={:?} and B={:?}",self.shape(),other.shape());

        let mut result = Matrix::new_zeros((self.rows, other.cols));

    
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * result.cols + j] = sum;
            }
        }
    
        result
    }


    /// Multiply element-wise
    pub fn multiply(&self,other:&Matrix)->Matrix{
        assert!(self.shape() == other.shape(),"Incopatibles shapes for multiply operation: {:?} x {:?}",self.shape(),other.shape());
        let mut new = other.clone();

        for (idx,(row_s,row_o)) in self.iter().zip(other.iter()).enumerate(){
            new.data[idx] = row_s*row_o;
        }
        new
    }

    /// Mean 
    /// Compute the aritimetic mean along the specified axis.
    /// The arithmetic mean is the sum of the elements along the axis divided by the number of elements.
    pub fn mean(&self)->f32{
        self.sum( ) /self.len() as f32
    }

    pub fn scale(&self,input:f32)->Matrix{
        let tmp:Vec<f32> = self.iter().map(|x| x * input).collect();
        Matrix::new(self.shape(), &tmp)
    }

    pub fn exp(&self)->Matrix{
        let tmp:Vec<f32> = self.data.iter().map(|n| n.exp()).collect();
        Matrix::new(self.shape(), &tmp)
    }

    pub fn sum(&self)->f32{
        self.iter().sum::<f32>()
    }

    /// Return the highter number find
    pub fn max(&self)->f32{
        self.iter().fold(std::f32::NEG_INFINITY,|max,&x| max.max(x))
    }

    /// Return the index of the highter number.
    pub fn max_index(&self)->usize{
        let (max_index, _) = self.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
        max_index
    }

    // /// Compares and returns the maximum of two values.
    // pub fn max_cmp(&self, n:f32)->f32{
    //     self.iter().fold(0f32,|max,&x| max.max(x))
    // }

    #[inline]
    pub fn iter_row(&self)->RowIter{
        RowIter{
            matrix_:self,
            current:0
        }
    }
}

pub struct RowIter<'a>{
    matrix_: &'a Matrix,
    current: usize
}

impl<'a> Iterator for RowIter<'a> {
    type Item = Matrix;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.matrix_.rows {
            let row = self.matrix_.get_row(self.current);
            self.current += 1;
            Some(row)
        } else {
            None
        }
    }
}


impl Default for Matrix{
    fn default() -> Self {
        Self { rows: 1, cols: 1, data: vec![0.0] }
    }
}

impl Sub for Matrix{
    type Output = Matrix;
    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.shape() == rhs.shape(),"Incopatibles shapes for subtraction operation: {:?} x {:?}",self.shape(),rhs.shape());
        let mut new = rhs.clone();
        for (idx,(row_s,row_o)) in self.iter().zip(rhs.iter()).enumerate(){
            new.data[idx] = row_s-row_o;
        }
        new
    }
}
impl Div for Matrix{
    type Output = Matrix;
    fn div(self, rhs: Self) -> Self::Output {
        assert!(self.shape() == rhs.shape(),"Incopatibles shapes for division operation: {:?} x {:?}",self.shape(),rhs.shape());
        let mut new = rhs.clone();
        for (idx,(row_s,row_o)) in self.iter().zip(rhs.iter()).enumerate(){
            new.data[idx] = row_s/row_o;
        }
        new
    }
}

impl Mul for Matrix{
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(&rhs)
    }
}

impl Div<f32> for Matrix{
    type Output = Matrix;
    fn div(self, rhs: f32) -> Self::Output {
        let tmp:Vec<f32> = self.iter().map(|v| v/rhs).collect();
        Matrix::new(self.shape(), tmp.as_slice())
    }
}

impl Mul<f32> for Matrix{
    type Output = Matrix;
    fn mul(self, rhs: f32) -> Self::Output {
        let tmp:Vec<f32> = self.iter().map(|v| v*rhs).collect();
        Matrix::new(self.shape(), tmp.as_slice())
    }
}

impl Sub<f32> for Matrix{
    type Output = Matrix;
    fn sub(self, rhs: f32) -> Self::Output {
        let tmp:Vec<f32> = self.iter().map(|v| v-rhs).collect();
        Matrix::new(self.shape(), tmp.as_slice())
    }
}

impl std::fmt::Debug for Matrix{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Matrix[\n")?;

        for row in 0..self.rows {
            f.write_str(" [")?;
            for col in 0..self.cols {write!(f, " {}", self.data[row*self.cols+ col]      )?;}
            f.write_str(" ],\n")?;
        }
        write!(f, "], Shape={:?}", (self.rows,self.cols))
    }
} 

impl<'a> IntoIterator for &'a mut Matrix {
    type Item = &'a mut f32;
    type IntoIter = std::slice::IterMut<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}













#[cfg(test)]
mod matrix_math{
    use super::Matrix;

    #[test]
    pub fn dot_operation(){
        let a = Matrix::new((4,2),&[1.,2.,3.,4.,5.,6.,7.,8.]);
        let b = a.clone().transpose();

        assert_eq!(
            a.dot(&b),
            Matrix{
                rows:4,cols:4,
                data: vec![5.,  11.,  17.,  23.,  11.,  25.,  39.,  53.,  17.,  39.,  61.,  83.,  23.,53.,  83., 113.]
            }
        )

    }

    #[test]
    fn test_dot_operation() {
        let a = Matrix::new((2, 2), &[1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new((2, 2), &[5.0, 6.0, 7.0, 8.0]);

        let result = a.dot(&b);
        let expected = Matrix::new((2, 2), &[19.0, 22.0, 43.0, 50.0]);

        assert_eq!(result.shape(), expected.shape());
        assert_eq!(result.data(), expected.data());
    }

    #[test]
    #[should_panic(expected = "Incompatible shapes for dot operation")]
    fn test_dot_operation_incompatible_shapes() {
        let a = Matrix::new((2, 2), &[1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new((3, 3), &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);

        let _result = a.dot(&b);
    }

    #[test]
    fn test_dot_operation_with_decimals() {
        let a = Matrix::new((2, 3), &[1.5, 2.0, 3.25, 4.75, 5.5, 6.1]);
        let b = Matrix::new((3, 2), &[0.5, 0.75, 1.25, 1.5, 2.0, 2.1]);

        let result = a.dot(&b);
        let expected = Matrix::new((2, 2), &[9.75, 10.95, 21.45, 24.6225]);

        assert_eq!(result.shape(), expected.shape());
        assert_eq!(result.data(), expected.data());
    }


}

#[cfg(test)]
mod matrix_test{
    use super::Matrix;


    #[test]
    pub fn transpose(){
        let a = Matrix::new((2,4),&[1.,2.,3.,4.,5.,6.,7.,8.]);
        assert_eq!(a.transpose(), Matrix{rows:4,cols:2, data:vec![1., 5., 2., 6., 3., 7., 4., 8.]})
    }

    #[test]
    fn transpose_matrix() {
        let test = Matrix::new((2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = test.transpose();
        let expected = Matrix::new((3, 2), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        assert_eq!(result.shape(), expected.shape());
        assert_eq!(result.data(), expected.data());
    }


    #[test]
    pub fn initialization(){
        let v = (0..4*5).map(|z| z as f32).collect::<Vec<f32>>();
        let data = Matrix::new((4,5), &v);
    
        assert_eq!(
            data,
            Matrix{
                rows:4,
                cols:5,
                data:v
            }
        );
    
    }

}


