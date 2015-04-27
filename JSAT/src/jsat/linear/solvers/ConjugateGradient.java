
package jsat.linear.solvers;

import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * Provides an iterative implementation of the COnjugate Gradient Method. 
 * <br><br>
 * The Conjugate method, if using exact arithmetic, produces the exact result after a finite 
 * number of iterations that is no more then the number of rows in the matrix. Because of this,
 * no max iteration parameter is given.
 * <br><br>
 * 
 * 
 * @author Edward Raff
 */
public class ConjugateGradient
{
    /**
     * Uses the Conjugate Gradient method to solve a linear system of 
     * equations involving a symmetric positive definite matrix.<br><br>
     * A symmetric positive definite matrix is a matrix A such that: <br>
     * <ul>
     * <li>A<sup>T</sup> = A</li>
     * <li>x<sup>T</sup> * A * x &gt; 0 for all x != 0</li>
     * </ul>
     * <br><br>
     * NOTE: No checks will be performed to confirm these properties of the given matrix. 
     * If a matrix is given that does not meet this requirements, invalid results may be returned. 
     * 
     * @param eps the precision of the desired result.
     * @param A the symmetric positive definite matrix
     * @param x an initial guess for x, can be all zeros. This vector will be altered
     * @param b the target values
     * @return the approximate solution to the equation <i>A x = b</i>
     */
    public static Vec solve(double eps, Matrix A, Vec x, Vec b)
    {
        if(!A.isSquare())
            throw new ArithmeticException("A must be a square (symmetric & positive definite) matrix");
        else if(A.rows() != b.length() || A.rows() != x.length())
            throw new ArithmeticException("Matrix A dimensions do not agree with x and b");
        int k = 0;
        Vec r_k = b.subtract(A.multiply(x));
        Vec p_k = r_k.clone();
        Vec Apk;
        
        double RdR = r_k.dot(r_k);
        do
        {
            Apk = A.multiply(p_k);
            double alpha_k = RdR /  p_k.dot(Apk) ;
            
            x.mutableAdd(alpha_k, p_k);
            
            r_k.mutableAdd(-alpha_k, Apk);
            
            double newRdR = r_k.dot(r_k);
            
            //Stop when we are close enough
            if(newRdR < eps*eps)
                return x;
            
            double beta_k = newRdR/RdR;
            
            p_k.mutableMultiply(beta_k);
            p_k.mutableAdd(r_k);
            
            //Set up for next ste
            RdR = newRdR;
        }
        while(k++ < A.rows());
        
        return x;
    }
    
    public static Vec solve(Matrix A, Vec b)
    {
        DenseVector x = new DenseVector(b.length());
        return solve(1e-10, A, x, b);
    }
    
    /**
     * Uses the Conjugate Gradient method to solve a linear system of 
     * equations involving a symmetric positive definite matrix.<br><br>
     * A symmetric positive definite matrix is a matrix A such that: <br>
     * <ul>
     * <li>A<sup>T</sup> = A</li>
     * <li>x<sup>T</sup> * A * x &gt; 0 for all x != 0</li>
     * </ul>
     * <br><br>
     * NOTE: No checks will be performed to confirm these properties of the given matrix. 
     * If a matrix is given that does not meet this requirements, invalid results may be returned. 
     * 
     * @param eps the precision of the desired result.
     * @param A the symmetric positive definite matrix
     * @param x an initial guess for x, can be all zeros. This vector will be altered
     * @param b the target values
     * @param Minv the of a matric M, such that M is a symmetric positive definite matrix.
     * Is applied as M<sup>-1</sup>( A x - b = 0) to increase convergence and stability. 
     * These increases are soley a property of M<sup>-1</sup>
     * 
     * @return the approximate solution to the equation <i>A x = b</i>
     */
    public static Vec solve(double eps, Matrix A, Vec x, Vec b, Matrix Minv)
    {
        if(!A.isSquare() || !Minv.isSquare())
            throw new ArithmeticException("A and Minv must be square (symmetric & positive definite) matrix");
        else if(A.rows() != b.length() || A.rows() != x.length())
            throw new ArithmeticException("Matrix A dimensions do not agree with x and b");
        else if(A.rows() != Minv.rows() || A.cols() != Minv.cols())
            throw new ArithmeticException("Matrix A and Minv do not have the same dimmentions");
        
        int k = 0;
        Vec r_k = b.subtract(A.multiply(x));
        Vec z_k = Minv.multiply(r_k);
        Vec p_k = z_k.clone();
        Vec Apk;
        double rkzk = r_k.dot(z_k);
        
        do
        {
            Apk = A.multiply(p_k);
            
            double alpha = rkzk/p_k.dot(Apk);
            x.mutableAdd(alpha, p_k);
            r_k.mutableSubtract(alpha, Apk);
            
            if(r_k.dot(r_k) < eps*eps)
                return x;
            
            z_k = Minv.multiply(r_k);//TODO implement method so we can reuse z_k space, instead of creating a new vector
            
            double newRkZk = r_k.dot(z_k);
            double beta = newRkZk/rkzk;
            rkzk = newRkZk;
            
            p_k.mutableMultiply(beta);
            p_k.mutableAdd(z_k);
            
            
        }
        while(k++ < A.rows());
        
        return x;
    }
    
    /**
     * Uses the Conjugate Gradient method to compute the least squares solution to a system 
     * of linear equations.<br>
     * Computes the least squares solution to A x = b. Where A is an m x n matrix and b is 
     * a vector of length m and x is a vector of length n
     * 
     * <br><br>
     * NOTE: Unlike {@link #solve(double, jsat.linear.Matrix, jsat.linear.Vec, jsat.linear.Vec) }, 
     * the CGNR method does not need any special properties of the matrix. Because of this, slower
     * convergence or numerical error can occur. 
     * 
     * @param eps the desired precision for the result
     * @param A any m x n matrix
     * @param x the initial guess for x, can be all zeros. This vector will be altered
     * @param b the target values
     * @return the least squares solution to A x = b
     */
    public static Vec solveCGNR(double eps, Matrix A, Vec x, Vec b)
    {
        if(A.rows() != b.length())
            throw new ArithmeticException("Dimensions do not agree for Matrix A and Vector b");
        else if(A.cols() != x.length())
            throw new ArithmeticException("Dimensions do not agree for Matrix A and Vector x");
        
        //TODO write a version that does not explicityly form the transpose matrix
        Matrix At = A.transpose();
        Matrix AtA = At.multiply(A);
        Vec AtB = At.multiply(b);
        
        return solve(eps, AtA, x, AtB);
    }
    
    public static Vec solveCGNR(Matrix A, Vec b)
    {
        DenseVector x = new DenseVector(A.cols());
        return solveCGNR(1e-10, A, x, b);
    }
}
