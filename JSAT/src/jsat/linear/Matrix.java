
package jsat.linear;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.utils.ModifiableCountDownLatch;
import static jsat.utils.SystemInfo.*;

/**
 * Generic class with some pre-implemented methods for a Matrix object. 
 * Throughout the documentation, the object that has its method called on will 
 * be denoted as <i>A</i>. So if you have code that looks like 
 * <br><br><center>
 * {@code Matrix gramHat = gram.subtract(Matrix.eye(gram.rows()));}
 * </center><br><br>
 * Then {@code gram} would be the matrix <i>A</i> in the documentation.
 * <br>
 * Matrices will use a capital letter, vectors a <b>bold</b> lower case letter, 
 * and scalars a normal lower case letter. 
 * 
 * 
 * @author Edward Rafff
 */
public abstract class Matrix implements Cloneable, Serializable
{

	private static final long serialVersionUID = 6888360415978051714L;

	/**
     * Creates a new Matrix that stores the result of {@code A+B}
     * @param B the matrix to add this <i>this</i>
     * @return {@code A+B}
     */
    public Matrix add(Matrix B)
    {
        Matrix toReturn = getThisSideMatrix(B);
        toReturn.mutableAdd(1.0, B);
        return toReturn;
    }
    
    /**
     * Creates a new Matrix that stores the result of {@code A+B}
     * @param B the matrix to add this <i>this</i>
     * @param threadPool the source of threads to do computation in parallel
     * @return {@code A+B}
     */
    public Matrix add(Matrix B, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(B);
        toReturn.mutableAdd(1.0, B, threadPool);
        return toReturn;
    }
    /**
     * Creates a new Matrix that stores the result of {@code A+c}
     * @param c the scalar to add to each value in <i>this</i>
     * @return {@code A+c}
     */
    public Matrix add(double c)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableAdd(c);
        return toReturn;
    }
    
    /**
     * Creates a new Matrix that stores the result of {@code A+c}
     * @param c the scalar to add to each value in <i>this</i>
     * @param threadPool the source of threads to do computation in parallel
     * @return {@code A+B}
     */
    public Matrix add(double c, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableAdd(c, threadPool);
        return toReturn;
    }
    
    /**
     * Alters the current matrix to store the value <i>A+B</i>
     * @param B the matrix to add this <i>this</i>
     */
    public void mutableAdd(Matrix B)
    {
        this.mutableAdd(1.0, B);
    }
    
    /**
     * Alters the current matrix to store the value <i>A+c*B</i>
     * @param c the scalar constant to multiple <i>B</i> by
     * @param B the matrix to add to <i>this</i> 
     */
    abstract public void mutableAdd(double c, Matrix B);
    
    /**
     * Alters the current matrix to store the value <i>A+B</i>
     * @param B the matrix to add to <i>this</i> 
     * @param threadpool the source of threads to do computation in parallel
     */
    public void mutableAdd(Matrix B, ExecutorService threadpool)
    {
        this.mutableAdd(1.0, B, threadpool);
    }
    
    /**
     * Alters the current matrix to store the value <i>A+c*B</i>
     * @param c the scalar constant to multiple <i>B</i> by
     * @param B the matrix to add to <i>this</i> 
     * @param threadPool the source of threads to do computation in parallel
     */
    abstract public void mutableAdd(double c, Matrix B, ExecutorService threadPool);
    
    /**
     * Alters the current matrix to store the value <i>A+c</i>
     * @param c the scalar constant to add to <i>this</i>
     */
    abstract public void mutableAdd(double c);
    
    /**
     * Alters the current matrix to store the value <i>A+c</i>
     * @param c the scalar constant to add to <i>this</i>
     * @param threadPool the source of threads to do computation in parallel
     */
    abstract public void mutableAdd(double c, ExecutorService threadPool);
    
    /**
     * Indicates whether or not this matrix can be mutated. If 
     * {@code false}, any method that contains "mutate" will not work. 
     * <br><br>
     * By default, this returns {@code true}
     * 
     * @return {@code true} if the matrix supports being altered, {@code false} 
     * other wise. 
     */
    public boolean canBeMutated()
    {
        return true;
    }
    
    /**
     * Returns an appropriate matrix to use for some operation A <i>op</i> B, 
     * where {@code A = this }
     * @param B the other matrix, may be null
     * @return a matrix that can be mutated to take the place of A
     */
    private Matrix getThisSideMatrix(Matrix B)
    {
        if(this.canBeMutated())
            return this.clone();
        else//so far, only other option in JSAT is a dense matrix
        {
            DenseMatrix dm = new DenseMatrix(rows(), cols());
            dm.mutableAdd(this);
            return dm;
        }
    }
    
    /**
     * Creates a new Matrix that stores the result of <i>A-B</i> 
     * @param B the matrix to subtract from <i>this</i>. 
     * @return a new matrix equal to <i>A-B</i>
     */
    public Matrix subtract(Matrix B)
    {
        Matrix toReturn = getThisSideMatrix(B);
        toReturn.mutableSubtract(1.0, B);
        return toReturn;
    }
    
    /**
     * Creates a new Matrix that stores the result of <i>A-B</i> 
     * @param B the matrix to subtract from <i>this</i>. 
     * @param threadPool the source of threads to do computation in parallel
     * @return a new matrix equal to <i>A-B</i>
     */
    public Matrix subtract(Matrix B, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(B);
        toReturn.mutableSubtract(1.0, B, threadPool);
        return toReturn;
    }
    
    /**
     * Creates a new Matrix that stores the result of <i>A-c</i> 
     * @param c the scalar constant to subtract from <i>this</i>
     * @return a new matrix equal to <i>A-B</i>
     */
    public Matrix subtract(double c)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableSubtract(c);
        return toReturn;
    }
    
    /**
     * Creates a new Matrix that stores the result of <i>A-c</i> 
     * @param c the scalar constant to subtract from <i>this</i>
     * @param threadPool the source of threads to do computation in parallel
     * @return a new matrix equal to <i>A-B</i>
     */
    public Matrix subtract(double c, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableSubtract(c, threadPool);
        return toReturn;
    }
    
    /**
     * Alters the current matrix to store <i>A-B</i>
     * @param B the matrix to subtract from <i>this</i>.  
     */
    public void mutableSubtract(Matrix B)
    {
        this.mutableSubtract(1.0, B);
    }
    
    /**
     * Alters the current matrix to store <i>A-c*B</i>
     * @param c the scalar constant to multiply <i>B</i> by
     * @param B the matrix to subtract from <i>this</i>.  
     */
    public void mutableSubtract(double c, Matrix B)
    {
        mutableAdd(-c, B);
    }
    
    /**
     * Alters the current matrix to store <i>A-B</i>
     * @param B the matrix to subtract from <i>this</i>.  
     * @param threadpool the source of threads to do computation in parallel
     */
    public void mutableSubtract(Matrix B, ExecutorService threadpool)
    {
        this.mutableSubtract(1.0, B, threadpool);
    }
    
    /**
     * Alters the current matrix to store <i>A-c*B</i>
     * @param c the scalar constant to multiply <i>B</i> by
     * @param B the matrix to subtract from <i>this</i>.  
     * @param threadPool the source of threads to do computation in parallel
     */
    public void mutableSubtract(double c, Matrix B, ExecutorService threadPool)
    {
        mutableAdd(-c, B, threadPool);
    }
    
    /**
     * Alters the current matrix to store <i>A-c</i>
     * @param c the scalar constant to subtract from <i>this</i>
     */
    public void mutableSubtract(double c)
    {
        mutableAdd(-c);
    }
    
    /**
     * Alters the current matrix to store <i>A-c</i>
     * @param c the scalar constant to subtract from <i>this</i>
     * @param threadPool the source of threads to do computation in parallel
     */
    public void mutableSubtract(double c, ExecutorService threadPool)
    {
        mutableAdd(-c, threadPool);
    }
    
    /**
     * If this matrix is <i>A<sub>m x n</sub></i>, and <i><b>b</b></i> has a length of n, and <i><b>c</b></i> has a length of m,
     * then this will mutate c to store <i><b>c</b> = <b>c</b> + A*<b>b</b>*z</i> 
     * @param b the vector to be treated as a colum vector
     * @param z the constant to multiply the <i>A*<b>b</b></i> value by. 
     * @param c where to place the result by addition
     * @throws ArithmeticException if the dimensions of A, <b>b</b>, or <b>c</b> do not all agree
     */
    abstract public void multiply(Vec b, double z, Vec c);
    
    /**
     * Creates a new vector that is equal to <i>A*<b>b</b> </i>
     * @param b the vector to multiply by
     * @return a new vector <i>A*<b>b</b> </i>
     */
    public Vec multiply(Vec b)
    {
        DenseVector result = new  DenseVector(rows());
        multiply(b, 1.0, result);
        return result;
    }
    
    /**
     * Creates a new matrix that stores <i>A*B</i>
     * @param B the matrix to multiply by
     * @return a new matrix <i>A*B</i>
     */
    public Matrix multiply(Matrix B)
    {
        Matrix C = new DenseMatrix(this.rows(), B.cols());
        multiply(B, C);
        return C;
    }
    
    /**
     * Creates a new matrix that stores <i>A*B</i>
     * @param B the matrix to multiply by
     * @param threadPool the source of threads to do computation in parallel
     * @return a new matrix <i>A*B</i>
     */
    public Matrix multiply(Matrix B, ExecutorService threadPool)
    {
        Matrix C = new DenseMatrix(this.rows(), B.cols());
        multiply(B, C, threadPool);
        return C;
    }
    
    /**
     * Alters the matrix <i>C</i> to be equal to <i>C = C+A*B</i>
     * @param B the matrix to multiply <i>this</i> with
     * @param C the matrix to add the result to
     */
    abstract public void multiply(Matrix B, Matrix C);
    
    /**
     * Alters the matrix <i>C</i> to be equal to <i>C = C+A*B</i>
     * @param B the matrix to multiply this with
     * @param C the matrix to add the result to
     * @param threadPool the source of threads to do computation in parallel
     */
    abstract public void multiply(Matrix B, Matrix C, ExecutorService threadPool);
    
    /**
     * Alters the matrix <i>C</i> to be equal to <i>C = C+A*B<sup>T</sup></i>
     * @param B the matrix to multiply <i>this</i> with
     * @param C the matrix to add the result to
     */
    abstract public void multiplyTranspose(final Matrix B, final Matrix C);
    
    /**
     * Returns the new matrix <i>C</i> that is <i>C = A*B<sup>T</sup></i>
     * @param B the matrix to multiply by the transpose of
     * @return the result C
     */
    public Matrix multiplyTranspose(final Matrix B)
    {
        Matrix C = new DenseMatrix(this.rows(), B.rows());
        multiplyTranspose(B, C);
        return C;
    }
    
    /**
     * Alters the matrix <i>C</i> to be equal to <i>C = C+A*B<sup>T</sup></i>
     * @param B the matrix to multiply this with
     * @param C the matrix to add the result to
     * @param threadPool the source of threads to do computation in parallel
     */
    abstract public void multiplyTranspose(final Matrix B, final Matrix C, ExecutorService threadPool);
    
    /**
     * Returns the new matrix <i>C</i> that is <i>C = A*B<sup>T</sup></i>
     * @param B the matrix to multiply by the transpose of
     * @param threadPool the source of threads to do computation in parallel
     * @return the result C
     */
    public Matrix multiplyTranspose(final Matrix B, ExecutorService threadPool)
    {
        Matrix C = new DenseMatrix(this.rows(), B.rows());
        multiplyTranspose(B, C, threadPool);
        return C;
    }
    
    /**
     * Creates a new Matrix that stores <i>A*c</i>
     * @param c the scalar constant to multiply by
     * @return a new vector <i>A*c</i>
     */
    public Matrix multiply(double c)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableMultiply(c);
        return toReturn;
    }
    
    /**
     * Creates a new Matrix that stores <i>A*c</i>
     * @param c the scalar constant to multiply by
     * @param threadPool the source of threads to do computation in parallel
     * @return a new matrix equal to <i>A*c</i>
     */
    public Matrix multiply(double c, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableMultiply(c, threadPool);
        return toReturn;
    }
    
    /**
     * Alters the current matrix to be equal to <i>A*c</i>
     * @param c the scalar constant to multiply by
     */
    abstract public void mutableMultiply(double c);
    
    /**
     * Alters the current matrix to be equal to <i>A*c</i>
     * @param c the scalar constant to multiply by
     * @param threadPool the source of threads to do computation in parallel
     */
    abstract public void mutableMultiply(double c, ExecutorService threadPool);
    
    abstract public Matrix[] lup();
    abstract public Matrix[] lup(ExecutorService threadPool);
    
    abstract public Matrix[] qr();
    abstract public Matrix[] qr(ExecutorService threadPool);
    
    /**
     * This method alters the size of a matrix, either adding or subtracting
     * rows from the internal structure of the matrix. Every resize call may
     * cause a new allocation internally, and should not be called for excessive
     * changing of a matrix. All added rows/ columns will have values of zero.
     * If a row / column is removed, it is always the bottom/right most row /
     * column removed. Values of the removed rows / columns will be lost.
     *
     * @param newRows the new number of rows, must be positive
     * @param newCols the new number of columns, must be positive.
     */
    abstract public void changeSize(int newRows, int newCols);
    
    /**
     * Transposes the current matrix in place, altering its value. 
     * Only valid for square matrices 
     */
    abstract public void mutableTranspose();
    
    /**
     * Returns a new matrix that is the transpose of this matrix. 
     * @return a new matrix <tt>A</tt>'
     */
    public Matrix transpose()
    {
        Matrix toReturn = new DenseMatrix(cols(), rows());
        this.transpose(toReturn);
        return toReturn;
    }
    
    /**
     * Overwrites the values stored in matrix <i>C</i> to store the value of 
     * <i>A'</i>
     * @param C the matrix to store the transpose of the current matrix
     * @throws ArithmeticException if the dimensions of <i>C</i> do not match 
     * the dimensions of <i>this'</i>
     */
    abstract public void transpose(Matrix C);
    
    /**
     * Creates a new matrix equal to <i>A'*B</i>, or the same result as <br>
     * <code>
     * A.{@link #transpose() transpose()}.{@link #multiply(jsat.linear.Matrix) multiply(B)}
     * </code>
     * 
     * @param B the other Matrix
     * @return a new matrix equal to <i>A'*B</i>
     */
    public Matrix transposeMultiply(Matrix B)
    {
        Matrix C = new DenseMatrix(this.cols(), B.cols());
        transposeMultiply(B, C);
        return C;
    }
    
    /**
     * Alters the matrix <i>C</i> so that <i>C = C + A'*B</i>
     * @param B the matrix to multiply by
     * @param C the matrix to add the result to
     */
    abstract public void transposeMultiply(Matrix B, Matrix C);
    
    /**
     * Computes the result matrix of <i>A'*B</i>, or the same result as <br>
     * <code>
     * A.{@link #transpose() transpose()}.{@link #multiply(jsat.linear.Matrix) multiply(B)}
     * </code>
     * 
     * @param B the matrix to multiply by
     * @param threadPool the source of threads to do computation in parallel
     * @return a new matrix equal to <i>A'*B</i>
     */
    public Matrix transposeMultiply(Matrix B, ExecutorService threadPool)
    {
        Matrix C = new DenseMatrix(this.cols(), B.cols());
        transposeMultiply(B, C, threadPool);
        return C;
    }
    
    /**
     * Alters the matrix <i>C</i> so that <i>C = C + A'*B</i>
     * @param B the matrix to multiply by
     * @param C the matrix to place the results in
     * @param threadPool the source of threads to do computation in parallel
     */
    abstract public void transposeMultiply(Matrix B, Matrix C, ExecutorService threadPool);
    
    /**
     * Alters the vector <i><b>x</b></i> to be equal to <i><b>x</b> = <b>x</b> + A'*<b>b</b>*c</i>
     * 
     * @param c the scalar constant to multiply by
     * @param b the vector to multiply by
     * @param x the vector the add the result to 
     */
    abstract public void transposeMultiply(double c, Vec b, Vec x);
    
    /**
     * Creates a new vector equal to <i><b>x</b> = A'*<b>b</b>*c</i>
     * @param c the scalar constant to multiply by
     * @param b the vector to multiply by
     * @return the new vector equal to <i>A'*b*c</i>
     */
    public Vec transposeMultiply(double c, Vec b)
    {
        DenseVector toReturns = new DenseVector(this.cols());
        this.transposeMultiply(c, b, toReturns);
        return toReturns;
    }
    
    /**
     * Returns the value stored at at the matrix position <i>A<sub>i,j</sub></i>
     * @param i the row, starting from 0
     * @param j the column, starting from 0
     * @return the value at <i>A<sub>i,j</sub></i>
     */
    abstract public double get(int i, int j);
    
    /**
     * Sets the value stored at at the matrix position <i>A<sub>i,j</sub></i>
     * @param i the row, starting from 0
     * @param j the column, starting from 0
     * @param value the value to place at <i>A<sub>i,j</sub></i>
     */
    abstract public void set(int i, int j, double value);
    
    /**
     * Alters the current matrix at index <i>(i,j)</i> to be equal to 
     * <i>A<sub>i,j</sub> = A<sub>i,j</sub> + value</i>
     * @param i the row, starting from 0
     * @param j the column, starting from 0
     * @param value the value to add to the matrix coordinate 
     */
    public void increment(int i, int j, double value)
    {
        if(Double.isNaN(value) || Double.isInfinite(value))
            throw new ArithmeticException("Can not add a value " + value);
        set(i, j, get(i, j)+value);
    }
    
    /**
     * Returns the number of rows stored in this matrix
     * @return the number of rows stored in this matrix
     */
    abstract public int rows();
    
    /**
     * Returns the number of columns stored in this matrix
     * @return the number of columns stored in this matrix
     */
    abstract public int cols();
    
    /**
     * Returns {@code true} if the matrix is sparse, {@code false} otherwise
     * @return {@code true} if the matrix is sparse, {@code false} otherwise
     */
    abstract public boolean isSparce();
    
    /**
     * Returns the number of non zero values stored in this matrix. This is 
     * mostly useful for sparse matrices. 
     * 
     * @return the number of non zero values stored in this matrix. 
     */
    public long nnz()
    {
        return ((long)rows())*cols();
    }
    
    /**
     * Returns {@code true} if the matrix is square, meaning it has the same 
     * number of {@link #rows() rows} and {@link #cols() columns}. 
     * @return {@code true} if this matrix is square, {@code false} if it is 
     * rectangular. 
     */
    public boolean isSquare()
    {
        return rows() == cols();
    }
    
    /**
     * Alters the current matrix by swapping the values stored in two different 
     * rows. 
     * @param r1 the first row to swap
     * @param r2 the second row to swap
     */
    abstract public void swapRows(int r1, int r2);

    /**
     * Creates a vector that has a copy of the values in column <i>j</i> of this
     * matrix. Altering it will not effect the values in <i>this</i> matrix
     * @param j the column to copy
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
    
    /**
     * Obtains a vector that is backed by <i>this</i>, at very little memory 
     * cost. Mutations to this vector will alter the values stored in the 
     * matrix, and vice versa. 
     * 
     * @param j the column to obtain a view of
     * @return a vector backed by the specified row of the matrix
     */
    public Vec getColumnView(final int j)
    {
        final Matrix M = this;
        return new Vec() 
        {
            /**
			 * 
			 */
			private static final long serialVersionUID = 7107290189250645384L;

			@Override
            public int length()
            {
                return rows();
            }

            @Override
            public double get(int index)
            {
                return M.get(index, j);
            }

            @Override
            public void set(int index, double val)
            {
                M.set(index, j, val);
            }

            @Override
            public boolean isSparse()
            {
                return M.isSparce();
            }

            @Override
            public Vec clone()
            {
                if(M.isSparce())
                    return new SparseVector(this);
                else
                    return new DenseVector(this);
            }
        };
    }
    
    /**
     * Creates a vector that has a copy of the values in row <i>i</i> of this 
     * matrix. Altering it will not effect the values in <i>this</i> matrix.  
     * @param r the row to copy
     * @return a clone of the row as a {@link Vec}
     */
    public Vec getRow(int r)
    {
        if(r < 0 || r >= rows())
            throw new ArithmeticException("Row was not a valid value " + r + " not in [0," + (rows()-1) + "]");
        DenseVector c = new DenseVector(cols());
        for(int j =0; j < cols(); j++)
            c.set(j, get(r, j));
        return c;
    }
    
    /**
     * Obtains a vector that is backed by <i>this</i>, at very little memory 
     * cost. Mutations to this vector will alter the values stored in the 
     * matrix, and vice versa. 
     * 
     * @param r the row to obtain a view of
     * @return a vector backed by the specified row of the matrix
     */
    public Vec getRowView(final int r)
    {
        final Matrix M = this;
        return new Vec() 
        {

            /**
			 * 
			 */
			private static final long serialVersionUID = 8484494698777822563L;

			@Override
            public int length()
            {
                return M.cols();
            }

            @Override
            public double get(int index)
            {
                return M.get(r, index);
            }

            @Override
            public void set(int index, double val)
            {
                M.set(r, index, val);
            }

            @Override
            public boolean isSparse()
            {
                return M.isSparce();
            }

            @Override
            public Vec clone()
            {
                if(M.isSparce())
                    return new SparseVector(this);
                else
                    return new DenseVector(this);
            }
        };
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
    
    /**
     * Convenience method that will return {@code true} only if the two input
     * matrices have the exact same dimensions. 
     * @param A the first matrix
     * @param B the second matrix
     * @return {@code true} if they have the exact same dimensions, 
     * {@code false} otherwise. 
     */
    public static boolean sameDimensions(Matrix A, Matrix B)
    {
        return A.rows() == B.rows() && A.cols() == B.cols();
    }
    
    /**
     * Convenience method that will return {@code true} only if the two input
     * matrices have dimensions compatible for multiplying <i>A*B</i>
     * @param A the first matrix
     * @param B the second matrix
     * @return {@code true} if they have dimensions allowing multiplication,
     * {@code false} otherwise. 
     */
    public static boolean canMultiply(Matrix A, Matrix B)
    {
        return A.cols() == B.rows();
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
     * leniency in the differences between matrix values. This is useful for 
     * when some amount of numerical error is expected
     * 
     * @param obj the other matrix 
     * @param range the max acceptable difference between two cell values
     * @return {@code true} if the difference between the values of each pair of
     * matrix elements are less than or equal to <i>range</i>
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
    
    /**
     * Alters the current matrix so that all values are equal to zero. 
     */
    abstract public void zeroOut();
    
    /**
     * Copes the values of this matrix into the other matrix of the same dimensions
     * @param other the matrix to overwrite the values of
     */
    public void copyTo(Matrix other)
    {
        if (this.rows() != other.rows() || this.cols() != other.cols())
            throw new ArithmeticException("Matrices are not of the same dimension");
        for(int i = 0; i < rows(); i++)
            this.getRowView(i).copyTo(other.getRowView(i));
    }
    
    /**
     * Alters row i of <i>this</i> matrix, such that 
     * <i>A[i,:] = A[i,:] + c*<b>b</b></i>
     * @param i the index of the row to update
     * @param c the scalar constant to multiply the vector by
     * @param b the vector to add to the specified row
     */
    public void updateRow(int i, double c, Vec b)
    {
        if(b.length() != this.cols())
            throw new ArithmeticException("vector is not of the same column length");
        if (b.isSparse())
            for (IndexValue iv : b)
                this.increment(i, iv.getIndex(), c * iv.getValue());
        else
            for (int j = 0; j < b.length(); j++)
                this.increment(i, j, c * b.get(j));
    }

    /**
     * Alters the matrix <i>A</i> such that, 
     * <i>A = A + c * <b>x</b> * <b>y</b>'</i>
     *
     * @param A the matrix to update
     * @param x the first vector
     * @param y the second vector
     * @param c the scalar constant to multiply the outer product by
     * @throws ArithmeticException if the vector dimensions are not compatible
     * with the matrix <i>A</i>
     */
    public static void OuterProductUpdate(Matrix A, Vec x, Vec y, double c)
    {
        if (x.length() != A.rows() || y.length() != A.cols())
            throw new ArithmeticException("Matrix dimensions do not agree with outer product");
        if (x.isSparse())
            for (IndexValue iv : x)
                A.updateRow(iv.getIndex(), iv.getValue() * c, y);
        else
            for (int i = 0; i < x.length(); i++)
            {
                double rowCosnt = c * x.get(i);
                A.updateRow(i, rowCosnt, y);
            }
    }

    /**
     * Alters the matrix <i>A</i> such that, 
     * <i>A = A + c * <b>x</b> * <b>y</b>'</i>
     * 
     * @param A the matrix to update
     * @param x the first vector
     * @param y the second vector
     * @param c the scalar constant to multiply the outer product by
     * @param threadpool the source of threads to do computation in parallel
     */
    public static void OuterProductUpdate(final Matrix A, final Vec x, final Vec y, final double c, ExecutorService threadpool)
    {
        if(x.length() != A.rows() || y.length() != A.cols())
            throw new ArithmeticException("Matrix dimensions do not agree with outer product");
        
        if (x.isSparse())
        {
            final ModifiableCountDownLatch mcdl = new ModifiableCountDownLatch(1);
            for (final IndexValue iv : x)
            {
                mcdl.countUp();
                threadpool.submit(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        A.updateRow(iv.getIndex(), iv.getValue() * c, y);
                        mcdl.countDown();
                    }
                });
            }
            mcdl.countDown();
            try
            {
                mcdl.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(Matrix.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        else
        {
            final CountDownLatch latch = new CountDownLatch(LogicalCores);
            for(int id = 0; id < LogicalCores; id++)
            {
                final int threadID = id;
                threadpool.submit(new Runnable() 
                {

                    @Override
                    public void run()
                    {
                        for(int i = threadID; i < x.length(); i+=LogicalCores)
                        {
                            double rowCosnt = c*x.get(i);
                            A.updateRow(i, rowCosnt, y);
                        }
                        latch.countDown();
                    }
                });

            }
            try
            {
                latch.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(Matrix.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    /**
     * Creates a new dense identity matrix with <i>k</i> rows and columns. 
     * @param k the number of rows / columns
     * @return a new dense identity matrix <i>I<sub>k</sub></i>
     */
    public static DenseMatrix eye(int k)
    {
        DenseMatrix eye = new DenseMatrix(k, k);
        for(int i = 0; i < k; i++ )
            eye.set(i, i, 1);
        return eye;
    }
    
    /**
     * Creates a new dense matrix filled with random values from 
     * {@link Random#nextDouble() }
     * 
     * @param rows the number of rows for the matrix
     * @param cols the number of columns for the matrix
     * @param rand the source of randomness 
     * @return a new dense matrix full of random values
     */
    public static DenseMatrix random(int rows, int cols, Random rand)
    {
        DenseMatrix m = new DenseMatrix(rows, cols);
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < cols; j++)
                m.set(i, j, rand.nextDouble());
        
        return m;
    }
    
    /**
     * Returns a new dense square matrix such that the main diagonal contains 
     * the values given in <tt>a</tt>
     * @param a the diagonal values of a matrix
     * @return the diagonal matrix represent by <i>a</i>
     */
    public static Matrix diag(Vec a)
    {
        DenseMatrix A = new DenseMatrix(a.length(), a.length());
        for(Iterator<IndexValue> iter = a.getNonZeroIterator(); iter.hasNext();)
        {
            IndexValue iv = iter.next();
            A.set(iv.getIndex(), iv.getIndex(), iv.getValue());
        }
            
        return A;
    }
    
    /**
     * Alters the matrix <i>A</i> so that it contains the result of <i>A</i> 
     * times a sparse matrix represented by only its diagonal values or
     * <i>A = A*diag(<b>b</b>)</i>. This is equivalent to the code
     * <code>
     * A = A{@link #multiply(jsat.linear.Matrix) .multiply}
     * ({@link #diag(jsat.linear.Vec) diag}(b)) 
     * </code>
     * @param A the square matrix to update
     * @param b the diagonal value vector 
     */
    public static void diagMult(Matrix A, Vec b)
    {
        if(A.cols() != b.length())
            throw new ArithmeticException("Could not multiply, matrix dimensions must agree");
        for(int i = 0; i < A.rows(); i++)
            RowColumnOps.multRow(A, i, b);
    }
    
    /**
     * Alters the matrix <i>A</i> so that it contains the result of 
     * sparse matrix represented by only its diagonal values times <i>A</i> or
     * <i>A = diag(<b>b</b>)*A</i>. This is equivalent to the code
     * <code>
     * b{@link Vec#multiply(jsat.linear.Matrix)  .multiply}
     * ({@link #diag(jsat.linear.Vec) diag}(A)) 
     * </code>
     * @param b the diagonal value vector 
     * @param A the square matrix to update
     */
    public static void diagMult(Vec b, Matrix A)
    {
        if(A.rows() != b.length())
            throw new ArithmeticException("Could not multiply, matrix dimensions must agree");
        for(int i = 0; i < A.rows(); i++)
            RowColumnOps.multRow(A, i, b.get(i));
    }
    
    /**
     * Checks to see if the given input is approximately symmetric. Rounding 
     * errors may cause the computation of a matrix to come out non symmetric, 
     * where |a[i,h] - a[j, i]| &lt; eps. Despite these errors, it may be 
     * preferred to treat the matrix as perfectly symmetric regardless. 
     * 
     * @param A the input matrix
     * @param eps the maximum tolerable difference between two entries
     * @return {@code true} if the matrix is approximately symmetric 
     */
    public static boolean isSymmetric(Matrix A, double eps)
    {
        if(!A.isSquare())
            return false;
        for(int i = 0; i < A.rows(); i++)
            for(int j = i+1; j < A.cols(); j++)
                if( Math.abs(A.get(i, j)-A.get(j, i)) > eps)
                    return false;
        return true;
    }
    
    /**
     * Checks to see if the given input is a perfectly symmetric matrix
     * @param A the input matrix
     * @return {@code true} if it is perfectly symmetric. 
     */
    public static boolean isSymmetric(Matrix A)
    {
        return isSymmetric(A, 0.0);
    }
    
    /**
     * Creates a new square matrix that is a pascal matrix. The pascal matrix of
     * size <i>n</i> is <i>n</i> by <i>n</i> and symmetric. 
     * 
     * @param size the number of rows and columns for the matrix
     * @return a pascal matrix of the desired size
     */
    public static Matrix pascal(int size)
    {
        if(size <= 0 )
            throw new ArithmeticException();
        DenseMatrix P = new DenseMatrix(size, size);
        RowColumnOps.fillRow(P, 0, 0, size, 1.0);
        RowColumnOps.fillCol(P, 0, 0, size, 1.0);
        for(int i = 1; i < size; i++)
            for(int j = 1; j < size; j++)
                P.set(i, j, P.get(i-1, j) + P.get(i, j-1));
        return P;
    }

    @Override
    abstract public Matrix clone();
}
