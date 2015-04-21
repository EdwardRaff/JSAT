
package jsat.linear;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import static java.lang.Math.*;
import java.util.concurrent.CountDownLatch;
import java.util.logging.Level;
import java.util.logging.Logger;
import static jsat.linear.LUPDecomposition.*;
import jsat.utils.SystemInfo;

/**
 * The Cholesky Decomposition factors a symmetric positive definite matrix A 
 * into the form A = L L<sup>T</sup>. The Cholesky Decomposition of a matrix is 
 * unique.
 * 
 * @author Edward Raff
 */
public class CholeskyDecomposition implements Serializable
{
    //TODO add block decomposition for efficency 
    

	private static final long serialVersionUID = 8925094456733750112L;
	/**
     * Contains the matrix 'L', but instead of just keeping the lower triangular, we keep it
     * in a symmetric  copy so {@link LUPDecomposition#forwardSub(jsat.linear.Matrix, jsat.linear.Vec) }
     * and backSub can be done without copying. 
     */
    private Matrix L;

    /**
     * Computes the Cholesky Decomposition of the matrix A. The matrix 
     * <tt>A</tt> will be altered to form the decomposition <tt>L</tt>. If A is 
     * still needed after this computation a {@link Matrix#clone() clone} of the
     * matrix should be given instead. <br>
     * NOTE: No check for the symmetric positive definite property will occur. 
     * The results of passing a matrix that does not meet this properties is 
     * undefined. 
     * 
     * @param A the matrix to create the Cholesky Decomposition of
     */
    public CholeskyDecomposition(final Matrix A)
    {
        if(!A.isSquare())
            throw new ArithmeticException("Input matrix must be symmetric positive definite");
        
        L = A;
        final int ROWS = A.rows();

        for (int j = 0; j < ROWS; j++)
        {
            double L_jj = computeLJJ(A, j);
            L.set(j, j, L_jj);
            updateRows(j, j + 1, ROWS, 1, A, L_jj);
        }
        copyUpperToLower(ROWS);
    }

    /**
     * Computes the Cholesky Decomposition of the matrix A. The matrix 
     * <tt>A</tt> will be altered to form the decomposition <tt>L</tt>. If A is 
     * still needed after this computation a {@link Matrix#clone() clone} of the
     * matrix should be given instead. <br>
     * NOTE: No check for the symmetric positive definite property will occur. 
     * The results of passing a matrix that does not meet this properties is 
     * undefined. 
     * 
     * @param A the matrix to create the Cholesky Decomposition of
     * @param threadpool the source of threads for computation
     */
    public CholeskyDecomposition(final Matrix A, ExecutorService threadpool)
    {
        if(!A.isSquare())
            throw new ArithmeticException("Input matrix must be symmetric positive definite");
        
        L = A;
        final int ROWS = A.rows();
        double nextLJJ = computeLJJ(A, 0);
        for (int j = 0; j < ROWS; j++)
        {
            final int J = j;
            final double L_jj = nextLJJ;//computeLJJ(A, j);
            L.set(j, j, L_jj);
            final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores-1);
            for (int i = 1; i < SystemInfo.LogicalCores; i++)
            {
                final int ID = i;
                threadpool.submit(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        updateRows(J, J + 1+ID, ROWS, SystemInfo.LogicalCores, A, L_jj);
                        latch.countDown();
                    }
                });
            }
            try
            {
                updateRows(J, J+1, ROWS, SystemInfo.LogicalCores, A, L_jj);
                if(j+1 < ROWS)
                    nextLJJ = computeLJJ(A, j+1);
                latch.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(CholeskyDecomposition.class.getName()).log(Level.SEVERE, null, ex);
            }

        }
        copyUpperToLower(ROWS);
    }

    /**
     * The Cholesky Decomposition computes the factorization A = L L<sup>T</sup>. This method returns L<sup>T</sup>
     * @return The upper triangular matrix L<sup>T</sup>
     */
    public Matrix getLT()
    {
        Matrix LT = new DenseMatrix(L.rows(), L.cols());
        
        for(int i = 0; i < L.rows(); i++)
            for(int j = i; j < L.rows(); j++)
                LT.set(i, j, L.get(i, j));
        
        return LT;
    }
    
    /**
     * Solves the linear system of equations A x = b 
     * @param b the vectors of values
     * @return the vector x such that A x = b
     */
    public Vec solve(Vec b)
    {
        //Solve  A x = L L^T x = b, for x 
        
        //First solve L y = b
        Vec y = forwardSub(L, b);
        //Sole L^T x = y
        Vec x = backSub(L, y);
        
        return x;
    }
    
    /**
     * Solves the linear system of equations A x = B
     * @param B the matrix of values 
     * @return the matrix c such that A x = B
     */
    public Matrix solve(Matrix B)
    {
        //Solve  A x = L L^T x = b, for x 
        
        //First solve L y = b
        Matrix y = forwardSub(L, B);
        //Sole L^T x = y
        Matrix x = backSub(L, y);
        
        return x;
    }
    
    /**
     * Solves the linear system of equations A x = B
     * @param B the matrix of values 
     * @param threadpool the source of threads for parallel evaluation
     * @return the matrix c such that A x = B
     */
    public Matrix solve(Matrix B, ExecutorService threadpool)
    {
        //Solve  A x = L L^T x = b, for x 
        
        //First solve L y = b
        Matrix y = forwardSub(L, B, threadpool);
        //Sole L^T x = y
        Matrix x = backSub(L, y, threadpool);
        
        return x;
    }
    
    /**
     * Computes the determinant of A
     * @return the determinant of A
     */
    public double getDet()
    {
        double det = 1;
        for(int i = 0; i < L.rows(); i++)
            det *= L.get(i, i);
        return det;
    }

    private double computeLJJ(final Matrix A, final int j)
    {
        /**
         *                _________________
         *               /       j - 1
         *              /        =====
         *             /         \      2
         * L    =     /   A    -  >    L
         *  j j      /     j j   /      j k
         *          /            =====
         *        \/             k = 1
         */
        double L_jj = A.get(j, j);
        for(int k = 0; k < j; k++)
            L_jj -= pow(L.get(j, k), 2);
        final double result = sqrt(L_jj);
        if(Double.isNaN(result))
            throw new ArithmeticException("input matrix is not positive definite");
        return result;
    }

    private void updateRows(final int j, final int start, final int end, final int skip, final Matrix A, final double L_jj)
    {
        /*
         * 
         *             /       j - 1          \
         *             |       =====          |
         *          1  |       \              |
         * L    = ---- |A    -  >    L    L   |
         *  i j   L    | i j   /      i k  j k|
         *         j j |       =====          |
         *             \       k = 1          /
         */
        for(int i = start; i < end; i+=skip)
        {
            double L_ij = A.get(i, j);
            for(int k = 0; k < j; k++)
                L_ij -= L.get(i, k)*L.get(j, k);
            L.set(i, j, L_ij/L_jj);
        }
    }

    private void copyUpperToLower(final int ROWS)
    {
        //Now copy so that All of L is filled 
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < i; j++)
                L.set(j, i, L.get(i, j));
    }
}
