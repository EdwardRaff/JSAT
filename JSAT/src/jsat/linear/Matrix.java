
package jsat.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;

/**
 *
 * @author Edward Rafff
 */
public abstract class Matrix implements Cloneable
{
    public Matrix add(Matrix b)
    {
        Matrix toReturn = clone();
        toReturn.mutableAdd(b);
        return toReturn;
    }
    public Matrix add(Matrix b, ExecutorService threadPool)
    {
        Matrix toReturn = clone();
        toReturn.mutableAdd(b, threadPool);
        return toReturn;
    }
    public Matrix add(double c)
    {
        Matrix toReturn = clone();
        toReturn.mutableAdd(c);
        return toReturn;
    }
    public Matrix add(double c, ExecutorService threadPool)
    {
        Matrix toReturn = clone();
        toReturn.mutableAdd(c, threadPool);
        return toReturn;
    }
    abstract public void mutableAdd(Matrix b);
    abstract public void mutableAdd(Matrix b, ExecutorService threadPool);
    abstract public void mutableAdd(double c);
    abstract public void mutableAdd(double c, ExecutorService threadPool);
    
    
    public Matrix subtract(Matrix b)
    {
        Matrix toReturn = clone();
        toReturn.mutableSubtract(b);
        return toReturn;
    }
    public Matrix subtract(Matrix b, ExecutorService threadPool)
    {
        Matrix toReturn = clone();
        toReturn.mutableSubtract(b, threadPool);
        return toReturn;
    }
    public Matrix subtract(double c)
    {
        Matrix toReturn = clone();
        toReturn.mutableSubtract(c);
        return toReturn;
    }
    public Matrix subtract(double c, ExecutorService threadPool)
    {
        Matrix toReturn = clone();
        toReturn.mutableSubtract(c, threadPool);
        return toReturn;
    }
    abstract public void mutableSubtract(Matrix b);
    abstract public void mutableSubtract(Matrix b, ExecutorService threadPool);
    public void mutableSubtract(double c)
    {
        mutableAdd(-c);
    }
    public void mutableSubtract(double c, ExecutorService threadPool)
    {
        mutableAdd(-c, threadPool);
    }
    
    /**
     * If this matrix is A_(m x n), and <tt>b</tt> has a length of n, then this will compute the result of A*b 
     * @param b the vector to be treated as a colum vector
     * @return the Vector result of the computation
     */
    abstract public Vec multiply(Vec b);
    abstract public Vec multiply(Vec b, ExecutorService threadPool);
    abstract public Matrix multiply(Matrix b);
    abstract public Matrix multiply(Matrix b, ExecutorService threadPool);
    public Matrix multiply(double c)
    {
        Matrix toReturn = clone();
        toReturn.mutableMultiply(c);
        return toReturn;
    }
    public Matrix multiply(double c, ExecutorService threadPool)
    {
        Matrix toReturn = clone();
        toReturn.mutableMultiply(c, threadPool);
        return toReturn;
    }
    abstract public void mutableMultiply(double c);
    abstract public void mutableMultiply(double c, ExecutorService threadPool);
    
    abstract public Matrix[] lup();
    abstract public Matrix[] lup(ExecutorService threadPool);
    
    abstract public Matrix[] qr();
    abstract public Matrix[] qr(ExecutorService threadPool);
    
    /**
     * Transposes the current matrix in place, altering its value. Only valid for square matrices 
     */
    abstract public void mutableTranspose();
    /**
     * Returns a new matrix that is the transpose of this matrix. 
     * @return a new matrix <tt>A</tt>'
     */
    abstract public Matrix transpose();
    
    /**
     * Computes the result matrix of A'*B, or the same result as <br>
     * <code>
     * A.{@link #transpose() transpose()}.{@link #multiply(jsat.linear.Matrix) multiply(B)}
     * </code>
     * 
     * @param b the other Matrix
     * @return The result of A'*B
     */
    abstract public Matrix transposeMultiply(Matrix b);
    /**
     * Computes the result matrix of A'*B, or the same result as <br>
     * <code>
     * A.{@link #transpose() transpose()}.{@link #multiply(jsat.linear.Matrix) multiply(B)}
     * </code>
     * 
     * @param b the other Matrix
     * @param threadPool the source of threads to run this computation in parallel
     * @return The result of A'*B
     */
    abstract public Matrix transposeMultiply(Matrix b, ExecutorService threadPool);
    
    /**
     * Computes the result of x = A'*b*c
     * 
     * @param c the constant to multiply by
     * @param b the vector to multiply by
     * @return the resulting vector of A'*b*c
     */
    abstract public Vec transposeMultiply(double c, Vec b);
    
    abstract public double get(int i, int j);
    abstract public void set(int i, int j, double value);
    
    abstract public int rows();
    abstract public int cols();
    
    abstract public boolean isSparce();
    abstract public long nnz();
    public boolean isSquare()
    {
        return rows() == cols();
    }
    
    abstract public void swapRows(int r1, int r2);
    
    abstract public Matrix clone();

    /**
     * Creates a clone of the values in column <tt>j</tt> of this matrix. Altering it will not effect the values in the source matrix
     * @param j the column
     * @return a clone of the column as a {@link Vec}
     */
    public Vec getColumn(int j)
    {
        if(j < 0 || j >= cols())
            throw new ArithmeticException("Column was not a valid value " + j + " not in [0," + (cols()-1) + "]");
        DenseVector c = new DenseVector(rows());
        for(int i =0; i < rows(); i++)
            c.set(i, get(i, j));
        return c;
    }
    
    public Vec getRow(int r)
    {
        if(r < 0 || r >= rows())
            throw new ArithmeticException("Row was not a valid value " + r + " not in [0," + (rows()-1) + "]");
        DenseVector c = new DenseVector(cols());
        for(int j =0; j < cols(); j++)
            c.set(j, get(r, j));
        return c;
    }
    
    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder(rows()*cols());
        
        sb.append("[");
        
        for(int i = 0; i < rows(); i++)
        {
            sb.append(get(i, 0));
            for(int j = 1; j < cols(); j++)
            {
                sb.append(", ").append(get(i, j));
            }
            sb.append(";");
        }
        
        sb.append("]");
        return sb.toString();
    }
    
    public static boolean sameDimensions(Matrix a, Matrix b)
    {
        return a.rows() == b.rows() && a.cols() == b.cols();
    }
    
    public static boolean canMultiply(Matrix a, Matrix b)
    {
        return a.cols() == b.rows();
    }

    @Override
    public boolean equals(Object obj)
    {
        if(obj == null || !(obj instanceof Matrix))
            return false;
        Matrix that = (Matrix) obj;
        
        if(this.rows() != that.rows() || this.cols() != that.cols())
            return false;
        
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                if(this.get(i, j) != that.get(i, j))
                    return false;
        
        return true;
    }
    
    /**
     * Performs the same as {@link #equals(java.lang.Object) }, but allows a
     * leniency in the results. This is use full for when some amount of 
     * numerical error in calculation is expected
     * 
     * @param obj the other matrix 
     * @param range the acceptable difference between two cell values
     * @return true if the difference between the values of each matrix element are less than or equal to <tt>range</tt>
     */
    public boolean equals(Object obj, double range)
    {
        if(obj == null || !(obj instanceof Matrix))
            return false;
        Matrix that = (Matrix) obj;
        
        if(this.rows() != that.rows() || this.cols() != that.cols())
            return false;
        
        for(int i = 0; i < rows(); i++)
            for(int j = 0; j < cols(); j++)
                if(Math.abs(this.get(i, j)-that.get(i, j)) > range)
                    return false;
        
        return true;
    }
    
    abstract public void zeroOut();
    
    /**
     * Alters the Matrix C such that, C = C + c * a * b<sup>T</sup>
     * @param C the matrix to update
     * @param a the first vector
     * @param b the second vector
     * @param c the constant to multiply computations by 
     */
    public static void OuterProductUpdate(Matrix C, Vec a, Vec b, double c)
    {
        for(int i = 0; i < a.length(); i++)
        {
            double rowCosnt = c*a.get(i);
            for(int j = 0; j < b.length(); j++)
                C.set(i, j, C.get(i, j) + rowCosnt * b.get(j) ); 
        }
    }
    
    /**
     * Creates a new identity matrix with k rows and columns. 
     * @param k the number of rows / columns
     * @return I_k
     */
    public static DenseMatrix eye(int k)
    {
        DenseMatrix eye = new DenseMatrix(k, k);
        for(int i = 0; i < k; i++ )
            eye.set(i, i, 1);
        return eye;
    }
    
    public static DenseMatrix random(int rows, int cols, Random rand)
    {
        DenseMatrix m = new DenseMatrix(rows, cols);
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                m.set(i, j, rand.nextDouble());
        
        return m;
    }
}
