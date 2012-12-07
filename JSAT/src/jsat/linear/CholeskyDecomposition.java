
package jsat.linear;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import static java.lang.Math.*;
import static jsat.linear.LUPDecomposition.*;

/**
 * The Cholesky Decomposition factors a symmetric positive definite matrix A into the form A = L L<sup>T</sup>. The Cholesky Decomposition of a matrix is unique 
 * @author Edward Raff
 */
public class CholeskyDecomposition implements Serializable
{
    /**
     * Contains the matrix 'L', but instead of just keeping the lower triangular, we keep it
     * in a symmetric  copy so {@link LUPDecomposition#forwardSub(jsat.linear.Matrix, jsat.linear.Vec) }
     * and backSub can be done without copying. 
     */
    private Matrix L;

    /**
     * Computes the Cholesky Decomposition of the matrix A. This matrix A will be altered. 
     * If A is still needed after this computation a {@link Matrix#cols() } of the matrix 
     * should be given instead. <br>
     * NOTE: No check for the symmetric positive definite property will occur. The results
     * of passing a matrix that does not meet this properties is undefined. 
     * 
     * @param A the matrix to create the Cholesky Decomposition of
     */
    public CholeskyDecomposition(Matrix A)
    {
        if(!A.isSquare())
            throw new ArithmeticException("Input matrix must be symmetric positive definite");
        
        L = A;
       
        for(int j = 0; j < A.rows(); j++)
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
            L.set(j, j, sqrt(L_jj));
            
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
            for(int i = j+1; i < L.rows(); i++)
            {
                double L_ij = A.get(i, j);
                for(int k = 0; k < j; k++)
                    L_ij -= L.get(i, k)*L.get(j, k);
                L.set(i, j, L_ij/L_jj);
            }
            
        }
        
        //Now copy so that All of L is filled 
        for(int i = 0; i < L.rows(); i++)
            for(int j = 0; j < i; j++)
                L.set(j, i, L.get(i, j));
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
}
