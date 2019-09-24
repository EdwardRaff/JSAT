/*
 * This implementation has been contributed under the Public Domain. 
 */
package jsat.linear;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import jsat.utils.random.RandomUtil;

/**
 * Computes the top <i>k</i> Eigen Values and Eigen Vectors of a symmetric
 * matrix <b>A<sup>n,n</sup></b>. <b>A</b> may be Sparse or Dense, and the results will be
 * computed faster than using the more general purpose
 * {@link EigenValueDecomposition}.<br>
 * <br>
 * If a non symmetric matrix <b>X<sup>n,m</sup></b> is given, this can implicit
 * compute the top-<i>k</i> eigen values and vectors for the matrix
 * <b>A=X<sup>T</sup> X</b> or <b>A=X X<sup>T</sup></b>, without having to
 * explicitly construct the potentially larger matrix A.
 *
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class Lanczos implements Serializable
{
    /**
     * The eigen values of the resulting decomposition
     */
    public double[] d;
    public Matrix eigenVectors;
    
    public Lanczos(Matrix A, int k, boolean A_AT, boolean is_symmetric) 
    {
        Random rand = RandomUtil.getRandom();
        
        int dims = A_AT ? A.rows() : A.cols();
        
        /**
         * The rank that we will perform computations too
         */
        int k_work = Math.min(k*2+1, dims);
        int extra_ranks = k_work-k;
        
        Vec v_prev = new ConstantVector(0.0, dims);
        //1. Let v_{1} be an arbitrary vector with Euclidean norm 1.
        Vec v_next = new DenseVector(dims);//init to 1/sqrt(n)
        v_next.add(1.0/Math.sqrt(dims));
        
        double[] alpha = new double[k_work];
        double[] beta = new double[k_work];
        
        DenseMatrix V = new DenseMatrix(k_work, dims);
        
        /**
         * Working variable
         */
        DenseVector w_j = new DenseVector(dims);
        DenseVector tmp = new DenseVector(A_AT ? A.cols() : A.rows());
        
        for(int j = 0; j < k_work; j++)
        {
            w_j.zeroOut();
            //3. Let w'_j = A v_j
            if(is_symmetric)
                //w'_j
                A.multiply(v_next, 1.0, w_j);
            else//We are doing A * A' * v_i
            {
                tmp.zeroOut();
                if(A_AT)//We are doing A * A' * v_i
                {
                    A.transposeMultiply(1.0, v_next, tmp);
                    A.multiply(tmp, 1.0, w_j);
                }
                else//We are doing A' * A * v_i
                {
                    A.multiply(v_next, 1.0, tmp);
                    A.transposeMultiply(1.0, tmp, w_j);
                }
            }
            
            //4. Let α_j =w'_j^T v_j.
            alpha[j] = w_j.dot(v_next);
            
            //5. Let w_j =w'_j- α_j v_j - β_j v_{j-1}}.
            w_j.mutableAdd(-alpha[j], v_next); 
            w_j.mutableAdd(-beta[j], v_prev); 
            
            //TODO, do not do full-orthogonalization! Thats too much
            orthogonalize(j, V, w_j);
            
            
            //Save off the row of V we just computed 
            v_prev = V.getRowView(j);
            v_next.copyTo(v_prev);
            
            //For simplicity, we do the first "two" steps at the end
            if(j+1 < k_work)
            {
                //1. β_{j+1}=||w_j||
                beta[j+1] = w_j.pNorm(2);
                //2a. If β_{j+1} == 0, pick as v_{j+1} an arbitrary vector with Euclidean norm 1 that is orthogonal to all of  v_{1},... ,v_{j-1}}.
                if(Math.abs(beta[j+1]) < 1e-15)
                {
                    //We need to pick a new value for w_j, which will become v_{j+1}
                    
                    //fill will random values
                    w_j.applyFunction(x->rand.nextDouble()*2-1);
                    orthogonalize(j+1, V, w_j);
                    
                    w_j.mutableDivide(w_j.pNorm(2)+1e-15);
                    beta[j+1] = 1;
                }
                //2b. v_{j+1}=w_j/β_{j+1}}
                w_j.copyTo(v_next);
                v_next.mutableDivide(beta[j+1]);
            }
        }
        
        //Inefficient computation of eigen values & vectors of diagonal matrix. 
        //TODO is to replace with smart implementaiton
        DenseMatrix triDaig = new DenseMatrix(k_work, k_work);
        for(int i = 0; i < k_work; i++)
        {
            triDaig.set(i, i, alpha[i]);
            if(i+1 < k_work)
            {
                triDaig.set(i, i+1, beta[i+1]);
                triDaig.set(i+1, i, beta[i+1]);
            }
        }
        EigenValueDecomposition evd = new EigenValueDecomposition(triDaig);
        //Sorty by largest magnitude eigen values first
        evd.sortByEigenValue((a,b) -> -Double.compare(Math.abs(a), Math.abs(b)));
        d = Arrays.copyOf(evd.getRealEigenvalues(), k);
//        d = evd.getRealEigenvalues();
        
        eigenVectors = V.transposeMultiply(evd.getV());
        eigenVectors.changeSize(dims, k);
        
    }
    
    /**
     * Returns a Vector of length <t>k</t> with the eigen values of the matrix
     * @return a vector of the eigen values
     */
    public Vec getEigenValues()
    {
        return new DenseVector(d);
    }

    /**
     * Returns a <t>n,k</t> matrix of the eigen vectors computed, where <n> is
     * the dimension of the original input.
     *
     * @return a <t>n,k</t> matrix of the eigen vectors
     */
    public Matrix getEigenVectors() 
    {
        return eigenVectors;
    }
    
    

    /**
     * Helper function for orthogonalizing vectors against existing vectors. One
     * call to this method performs partial orthogonalization, two sequential
     * calls does full orthogonalization
     *
     * @param j the current limit of vectors (rows) in V to orthogonalize
     * against
     * @param V the matrix of Vectors to be orthogonalized
     * @param w_j the vector to make orthogonal compared to the previous j vectors
     */
    private void orthogonalize(int j, DenseMatrix V, Vec w_j) 
    {
        //Orthogonalize step that is needed, but not include din Wiki
        for(int i = 0; i < j; i++)
        {
            Vec V_i = V.getRowView(i);
            double tmp_dot = w_j.dot(V_i);
            
            if(Math.abs(tmp_dot) < 1e-15)//essentially zero, nothing to orthogonalize
                continue;
            
            w_j.mutableAdd(-tmp_dot, V_i);
        }
        
    }
    
}
