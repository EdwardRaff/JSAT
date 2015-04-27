package jsat.linear;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import static jsat.linear.Matrix.*;

/**
 *
 * @author Edward Raff
 */
public class HessenbergForm implements Serializable
{

	private static final long serialVersionUID = 1411467026933172901L;

	public static void hess(Matrix A)
    {
        hess(A, null);
    }
    
    /**
     * Alters the matrix A such that it is in upper Hessenberg form. 
     * @param A the matrix to transform into upper Hessenberg form
     */
    public static void hess(Matrix A, ExecutorService threadpool)
    {
        if(!A.isSquare())
            throw new ArithmeticException("Only square matrices can be converted to Upper Hessenberg form");
        int m = A.rows();
        /**
         * Space used to store the vector for updating the columns of A
         */
        DenseVector columnUpdateTmp = new DenseVector(m);
        double[] vk = new double[m];
        /**
         * Space used for updating the sub matrix at step i
         */
        double[] subMatrixUpdateTmp = new double[m];
        double tmp;//Used for temp values
        for(int i = 0; i < m-2; i++)
        {
            //Holds the norm, sqrt{a_i^2 + ... + a_m^2}
            double s = 0.0;
            //First step of the loop done outside to do extra bit
            double sigh = A.get(i+1, i);//Holds the multiplication factor
            vk[i+1] = sigh;
            s += sigh*sigh;
            sigh = sigh > 0 ? 1 : -1;//Sign dosnt change the squaring, so we do it first
            
            
            
            for(int j = i+2; j < m; j++)
            {
                tmp = A.get(j, i);
                vk[j] = tmp;
                s += tmp*tmp;
            }
            
            double s1 = -sigh*Math.sqrt(s);
            //Now re use s to quickly get the norm of vk, since it will be almost the same vector
            s -= vk[i+1]*vk[i+1];
            vk[i+1] -= s1;
            s += vk[i+1]*vk[i+1];
            double s1Inv = 1.0/Math.sqrt(s);//Re use to store the norm of vk. Do the inverse to multiply quickly instead of divide
            for(int j = i+1; j < m; j++)
                vk[j] *= s1Inv;
            
            //Update sub sub matrix A[i+1:m, i:m]
            //NOTE: The first column that will be altered can be done ourslves, since we know the value set (s1) and that all below it will ber zero
            Matrix subA = new SubMatrix(A, i+1, i, m, m);
            DenseVector vVec = new DenseVector(vk, i+1, m);
            Vec tmpV = new DenseVector(subMatrixUpdateTmp, i, m);
            tmpV.zeroOut();
            vVec.multiply(subA, tmpV);
            
            if(threadpool == null)
                OuterProductUpdate(subA, vVec, tmpV, -2.0);
            else
                OuterProductUpdate(subA, vVec, tmpV, -2.0, threadpool);
            //Zero out ourselves after.
            //TODO implement so we dont compute the first row
            A.set(i+1, i, s1);
            for(int j = i+2; j < m; j++)
                A.set(j, i, 0.0);
            
            
            //Update the columns of A[0:m, i+1:m] 
            subA = new SubMatrix(A, 0, i+1, m, m);
            columnUpdateTmp.zeroOut();
            subA.multiply(vVec, 1.0, columnUpdateTmp);
            if(threadpool == null)
                OuterProductUpdate(subA, columnUpdateTmp, vVec, -2.0);
            else
                OuterProductUpdate(subA, columnUpdateTmp, vVec, -2.0, threadpool);
        }
    }
}
