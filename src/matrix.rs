use std::fmt::Write;


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
        assert_eq!(shape.0*shape.1, data.len(), "Shape don't match.");
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

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows,"Incompatible shapes for dot operation");

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
}

impl Default for Matrix{
    fn default() -> Self {
        Self { rows: 1, cols: 1, data: vec![0.0] }
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


