
package jsat.linear;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.utils.ModifiableCountDownLatch;
import jsat.utils.SystemInfo;
import static jsat.utils.SystemInfo.*;

/**
 *
 * @author Edward Rafff
 */
public abstract class Matrix implements Cloneable, Serializable
{
    public Matrix add(Matrix b)
    {
        Matrix toReturn = getThisSideMatrix(b);
        toReturn.mutableAdd(1.0, b);
        return toReturn;
    }
    public Matrix add(Matrix b, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(b);
        toReturn.mutableAdd(1.0, b, threadPool);
        return toReturn;
    }
    public Matrix add(double c)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableAdd(c);
        return toReturn;
    }
    public Matrix add(double c, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableAdd(c, threadPool);
        return toReturn;
    }
    public void mutableAdd(Matrix b)
    {
        this.mutableAdd(1.0, b);
    }
    abstract public void mutableAdd(double c, Matrix b);
    public void mutableAdd(Matrix b, ExecutorService threadpool)
    {
        this.mutableAdd(1.0, b, threadpool);
    }
    abstract public void mutableAdd(double c, Matrix b, ExecutorService threadPool);
    abstract public void mutableAdd(double c);
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
    
    public Matrix subtract(Matrix b)
    {
        Matrix toReturn = getThisSideMatrix(b);
        toReturn.mutableSubtract(-1.0, b);
        return toReturn;
    }
    public Matrix subtract(Matrix b, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(b);
        toReturn.mutableSubtract(1.0, b, threadPool);
        return toReturn;
    }
    public Matrix subtract(double c)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableSubtract(c);
        return toReturn;
    }
    public Matrix subtract(double c, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableSubtract(c, threadPool);
        return toReturn;
    }
    
    public void mutableSubtract(Matrix b)
    {
        this.mutableSubtract(1.0, b);
    }
    
    public void mutableSubtract(double c, Matrix b)
    {
        mutableAdd(-c, b);
    }
    
    public void mutableSubtract(Matrix b, ExecutorService threadpool)
    {
        this.mutableSubtract(1.0, b, threadpool);
    }
    public void mutableSubtract(double c, Matrix b, ExecutorService threadPool)
    {
        mutableAdd(-c, b, threadPool);
    }
    public void mutableSubtract(double c)
    {
        mutableAdd(-c);
    }
    public void mutableSubtract(double c, ExecutorService threadPool)
    {
        mutableAdd(-c, threadPool);
    }
    
    /**
     * If this matrix is A<sub>m x n</sub>, and <tt>b</tt> has a length of n, and <tt>c</tt> has a length of m,
     * then this will compute the result of c = c + A*b*z 
     * @param b the vector to be treated as a colum vector
     * @param z the constant to multiply the <i>A*b</i> value by. 
     * @param c where to place the result by addition
     * @return the Vector result of the computation
     * @throws ArithmeticException if the dimensions of A, b, or c do not all agree
     */
    abstract public void multiply(Vec b, double z, Vec c);
    public Vec multiply(Vec b)
    {
        DenseVector result = new  DenseVector(rows());
        multiply(b, 1.0, result);
        return result;
    }
    
    public Matrix multiply(Matrix b)
    {
        Matrix C = new DenseMatrix(this.rows(), b.cols());
        multiply(b, C);
        return C;
    }
    public Matrix multiply(Matrix b, ExecutorService threadPool)
    {
        Matrix C = new DenseMatrix(this.rows(), b.cols());
        multiply(b, C, threadPool);
        return C;
    }
    
    /**
     * Computes the result of C = C + this*b, the matrix C will be modified
     * @param b the matrix to multiply this with
     * @param C the matrix to add the result to
     */
    abstract public void multiply(Matrix b, Matrix C);
    /**
     * Computes the result of C = C + this*b, the matrix C will be modified
     * @param b the matrix to multiply this with
     * @param threadPool the source of threads for the computation
     * @param C the matrix to add the result to
     */
    abstract public void multiply(Matrix b, Matrix C, ExecutorService threadPool);
    public Matrix multiply(double c)
    {
        Matrix toReturn = getThisSideMatrix(null);
        toReturn.mutableMultiply(c);
        return toReturn;
    }
    public Matrix multiply(double c, ExecutorService threadPool)
    {
        Matrix toReturn = getThisSideMatrix(null);
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
    public Matrix transpose()
    {
        Matrix toReturn = new DenseMatrix(cols(), rows());
        this.transpose(toReturn);
        return toReturn;
    }
    abstract public void transpose(Matrix C);
    
    /**
     * Computes the result matrix of A'*B, or the same result as <br>
     * <code>
     * A.{@link #transpose() transpose()}.{@link #multiply(jsat.linear.Matrix) multiply(B)}
     * </code>
     * 
     * @param b the other Matrix
     * @return The result of A'*B
     */
    public Matrix transposeMultiply(Matrix b)
    {
        Matrix C = new DenseMatrix(this.cols(), b.cols());
        transposeMultiply(b, C);
        return C;
    }
    
    /**
     * Updates the matrix C so that C = C + A'*B*c
     * @param b the other matrix
     * @param C the matrix to place the results in
     */
    abstract public void transposeMultiply(Matrix b, Matrix C);
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
    public Matrix transposeMultiply(Matrix b, ExecutorService threadPool)
    {
        Matrix C = new DenseMatrix(this.cols(), b.cols());
        transposeMultiply(b, C, threadPool);
        return C;
    }
    
    /**
     * Updates the matrix C so that C = C + A'*B
     * @param b the other matrix
     * @param threadPool the source of threads to run this computation in parallel
     * @param C the matrix to place the results in
     */
    abstract public void transposeMultiply(Matrix b, Matrix C, ExecutorService threadPool);
    
    /**
     * Computes the result of x = x + A'*b*c
     * 
     * @param c the constant to multiply by
     * @param b the vector to multiply by
     * @param x the vector the update with the result
     */
    abstract public void transposeMultiply(double c, Vec b, Vec x);
    /**
     * Computes the result of x = A'*b*c
     * @param c the constant to multiply by
     * @param b the vector to multiply by
     * @return the resulting vector of A'*b*c
     */
    public Vec transposeMultiply(double c, Vec b)
    {
        DenseVector toReturns = new DenseVector(this.cols());
        this.transposeMultiply(c, b, toReturns);
        return toReturns;
    }
    
    abstract public double get(int i, int j);
    abstract public void set(int i, int j, double value);
    public void increment(int i, int j, double value)
    {
        if(Double.isNaN(value) || Double.isInfinite(value))
            throw new ArithmeticException("Can not add a value " + value);
        set(i, j, get(i, j)+value);
    }
    
    abstract public int rows();
    abstract public int cols();
    
    abstract public boolean isSparce();
    public long nnz()
    {
        return ((long)rows())*cols();
    }
    public boolean isSquare()
    {
        return rows() == cols();
    }
    
    abstract public void swapRows(int r1, int r2);
    

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
     * Updates row i of this matric, such that A[i,:] = A[i,:] + c*b
     * @param i the index of the row
     * @param c the constant to multiply the vector by
     * @param b the 
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
     * Alters the Matrix A such that, A = A + c * a * b<sup>T</sup>
     *
     * @param A the matrix to update
     * @param a the first vector
     * @param b the second vector
     * @param c the constant to multiply computations by
     */
    public static void OuterProductUpdate(Matrix A, Vec a, Vec b, double c)
    {
        if(a.length() != A.rows() || b.length() != A.cols())
            throw new ArithmeticException("Matrix dimensions do not agree with outer product");
        if(a.isSparse())
            for(IndexValue iv : a)
                A.updateRow(iv.getIndex(), iv.getValue()*c, b);
        else
            for (int i = 0; i < a.length(); i++)
            {
                double rowCosnt = c * a.get(i);
                A.updateRow(i, rowCosnt, b);
            }
    }
    
    public static void OuterProductUpdate(final Matrix A, final Vec a, final Vec b, final double c, ExecutorService threadpool)
    {
        if(a.length() != A.rows() || b.length() != A.cols())
            throw new ArithmeticException("Matrix dimensions do not agree with outer product");
        
        if (a.isSparse())
        {
            final ModifiableCountDownLatch mcdl = new ModifiableCountDownLatch(1);
            for (final IndexValue iv : a)
            {
                mcdl.countUp();
                threadpool.submit(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        A.updateRow(iv.getIndex(), iv.getValue() * c, b);
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
                        for(int i = threadID; i < a.length(); i+=LogicalCores)
                        {
                            double rowCosnt = c*a.get(i);
                            A.updateRow(i, rowCosnt, b);
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
    
    /**
     * Returns a square matrix such that the main diagonal contains the values given in <tt>a</tt>
     * @param a the diagonal values of a matrix
     * @return the diagonal matrix represent by <tt>a</tt>
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
     * Alters the matrix A so that it contains the result of A{@link #multiply(jsat.linear.Matrix) .multiply}({@link #diag(jsat.linear.Vec) diag}(b)) 
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
     * Alters the matrix A so that it contains the result of b{@link Vec#multiply(jsat.linear.Matrix)  .multiply}({@link #diag(jsat.linear.Vec) diag}(A)) 
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
     * @return <tt>true</tt> if the matrix is approximately symmetric 
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
     * @return <tt>true</tt> if it is perfectly symmetric. 
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
