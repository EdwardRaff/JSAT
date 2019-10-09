/*
 * This implementation contributed under the Public Domain. 
 */
package jsat.linear;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.nextUp;
import java.util.Arrays;

/**
 * Computes the Truncated version of the Singular Value Decomposition (SVD).
 * Given a rectangular matrix <b>A</b><sub>m,n </sub>, and a desired number of
 * singular values <i>k</i>, the Truncated SVD computes <b>A</b><sub>m,n </sub>
 * &asymp; <b>U</b><sub>m,k </sub> <b>&Sigma;</b><sub>k,k </sub>
 * <b>V</b><sup>T</sup><sub>k,n </sub>. This is faster than computing the full
 * SVD using {@link SingularValueDecomposition}, as only the top-<i>k</i>
 * singular values and associated data will be computed. This implementation
 * also
 * supports sparse inputs.
 *
 * @author Edward Raff
 */
public class TruncatedSVD 
{
    private Matrix U, V;
    /**
     * Stores the diagonal values of the S matrix, and contains the bidiagonal values of A during initial steps. 
     */
    private double[] s;
    
    /**
     * Creates a new SVD of the matrix {@code A} such that A = U &Sigma; V<sup>T</sup>.The matrix 
     * {@code  A} will be modified and used as temp space when computing the SVD. 
     * @param A the matrix to create the SVD of
     * @param k
     */
    public TruncatedSVD(Matrix A, int k)
    {
        DenseVector invertS = new DenseVector(k);
        if(A.rows() < A.cols())//U will be smaller than V, so lets compute U with Eigen, and reconstruc V
        {
            Lanczos u_lanc = new Lanczos(A, k, true, false);
            U = u_lanc.getEigenVectors();
            s = u_lanc.d;
            for(int i = 0; i < k; i++)
            {
                s[i] = Math.sqrt(Math.max(s[i], 0.0));
                if(s[i] == 0)//numerical issue
                    invertS.set(i, 0.0);
                else
                    invertS.set(i, 1/s[i]);
            }
            //V = (A^T * u * (diag(1/s)))
            V = A.transposeMultiply(U);
            Matrix.diagMult(V, invertS);
            V = V.transpose();
        
        }
        else
        {
            Lanczos v_lanc = new Lanczos(A, k, false, false);
            V = v_lanc.getEigenVectors().transpose();
            s = v_lanc.d;
            for(int i = 0; i < k; i++)
            {
                s[i] = Math.sqrt(Math.max(s[i], 0.0));
                if(s[i] == 0)//numerical issue
                    invertS.set(i, 0.0);
                else
                    invertS.set(i, 1/s[i]);
            }
            //U = (X * diag(1/s) *v)^T
            //TODO this is inefficent, need to add new function to replace
            Matrix tmp = V.clone();
            Matrix.diagMult(invertS, tmp);
            U = A.multiplyTranspose(tmp);
        }
        
    }
    
    private int sLength()
    {
        return min(U.rows(), V.rows());
    }
    
    /**
     * Returns the backing matrix U of the SVD. Do not alter this matrix. 
     * @return the matrix U of the SVD
     */
    public Matrix getU()
    {
        return U;
    }

    /**
     * Returns the backing matrix V of the SVD. Do not later this matrix.
     * @return the matrix V of the SVD
     */
    public Matrix getV()
    {
        return V;
    }

    /**
     * Returns a copy of the sorted array of the singular values, include the near zero ones. 
     * @return a copy of the sorted array of the singular values, including the near zero ones. 
     */
    public double[] getSingularValues()
    {
        return Arrays.copyOf(s, sLength());
    }
    
    /**
     * Returns the diagonal matrix S such that the SVD product results in the original matrix. The diagonal contains the singular values. 
     * @return a dense diagonal matrix containing the singular values
     */
    public Matrix getS()
    {
        Matrix DS = new DenseMatrix(U.rows(), V.rows());
        for(int i = 0; i < sLength(); i++)
            DS.set(i, i, s[i]);
        return DS;
    }
    
    /**
     * Returns the 2 norm of the matrix, which is the maximal singular value. 
     * @return the 2 norm of the matrix
     */
    public double getNorm2()
    {
        return s[0];
    }
    
    /**
     * Returns the condition number of the matrix. The condition number is a positive measure of the numerical 
     * instability of the matrix. The larger the value, the less stable the matrix. For singular matrices, 
     * the result is {@link Double#POSITIVE_INFINITY}. 
     * @return the condition number of the matrix
     */
    public double getCondition()
    {
        return getNorm2()/s[sLength()-1];
    }
    
    private double getDefaultTolerance()
    {
        return max(U.rows(), V.rows())*(nextUp(getNorm2())-getNorm2());
    }
    
    /**
     * Returns the numerical rank of the matrix. Near zero values will be ignored. 
     * @return the rank of the matrix
     */
    public int getRank()
    {
        return getRank(getDefaultTolerance());
    }
    
    /**
     * Indicates whether or not the input matrix was of full rank, full 
     * rank matrices are more numerically stable. 
     * 
     * @return <tt>true</tt> if the matrix was of full tank
     */
    public boolean isFullRank()
    {
        return getRank() == sLength();
    }
    
    /**
     * Returns the numerical rank of the matrix. Values &lt;= than <tt>tol</tt> will be ignored. 
     * @param tol the cut of for singular values
     * @return the rank of the matrix
     */
    public int getRank(double tol)
    {
        for(int i = 0; i < sLength(); i++)
            if(s[i] <= tol)
                return i;
        return sLength();
    }
    
    /**
     * Returns an array containing the inverse singular values. Near zero values are converted to zero. 
     * @return an array containing the inverse singular values
     */
    public double[] getInverseSingularValues()
    {
        return getInverseSingularValues(getDefaultTolerance());
    }
    /**
     * Returns an array containing the inverse singular values. Values that are &lt;= <tt>tol</tt> are converted to zero. 
     * @param tol the cut of for singular values
     * @return an array containing the inverse singular values
     */
    public double[] getInverseSingularValues(double tol)
    {
        double[] sInv = Arrays.copyOf(s, sLength());
        for(int i = 0; i < sInv.length; i++)
            if(sInv[i] > tol)
                sInv[i] = 1.0/sInv[i];
            else
                sInv[i] = 0;
        return sInv;
    }

}
