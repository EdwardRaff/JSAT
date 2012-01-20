
package jsat.datatransform;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * Principle Component Analysis is a method that attempts to create a basis of 
 * the given space that maintains the variance in the data set while eliminating
 * correlation of the variables. 
 * <br>
 * When a full basis is formed, the dimensionality will remain the same, 
 * but the data will be transformed to a new space. <br>
 * PCA is particularly useful when a small number of basis can explain 
 * most of the variance in the data set that is not related to noise, 
 * maintaining information while reducing the dimensionality of the data. 
 * <br><br>
 * PCA works only on the numerical attributes of a data set. <br>
 * For PCA to work correctly, a {@link ZeroMeanTransform} should 
 * be applied to the data set first. If not done, the first 
 * dimension of PCA may contain noise and become uninformative,
 * possibly throwing off the computation of the other PCs
 * 
 * @author Edward Raff
 * @see ZeroMeanTransform
 */
public class PCA implements DataTransform
{
    /**
     * The transposed matrix of the Principal Components
     */
    private Matrix P;

    /**
     * Performs PCA analysis using the given data set, so that transformations may be performed on future data points.  <br>
     * <br>
     * NOTE: The maximum number of PCs will be learned until a convergence threshold is meet. It is possible that the 
     * number of PCs computed will be equal to the number of dimensions, meaning no dimensionality reduction has 
     * occurred, but a transformation of the dimensions into a new space. 
     * 
     * @param dataSet the data set to learn from
     */
    public PCA(DataSet dataSet)
    {
        this(dataSet, Integer.MAX_VALUE);
    }
    
    /**
     * Performs PCA analysis using the given data set, so that transformations may be performed on future data points.  
     * 
     * @param dataSet the data set to learn from
     * @param maxPCs the maximum number of Principal Components to let the algorithm learn. The algorithm may stop
     * earlier if all the variance has been explained, or the convergence threshold has been met. 
     * Note, the computable maximum number of PCs is limited to the minimum of the number of samples and the
     * number of dimensions. 
     */
    public PCA(DataSet dataSet, int maxPCs)
    {
        this(dataSet, maxPCs, 1e-4);
    }
    
    /**
     * Performs PCA analysis using the given data set, so that transformations may be performed on future data points.  
     * 
     * @param dataSet the data set to learn from
     * @param maxPCs the maximum number of Principal Components to let the algorithm learn. The algorithm may stop
     * earlier if all the variance has been explained, or the convergence threshold has been met. 
     * Note, the computable maximum number of PCs is limited to the minimum of the number of samples and the
     * number of dimensions. 
     * @param threshold a convergence threshold, any small value will work. Smaller values will 
     * not produce more accurate results, but may make the algorithm take longer if it would 
     * have terminated before <tt>maxPCs></tt> was reached.  
     */
    public PCA(DataSet dataSet, int maxPCs, double threshold)
    {
        List<Vec> scores = new ArrayList<Vec>();
        List<Vec> loadings = new ArrayList<Vec>();
        //E(0) = X The E-matrix for the zero-th PC

        //Contains the unexplained variance in the data at each step. 
        Matrix E = dataSet.getDataMatrix();
        
        //This is the MAX number of possible Principlal Components
        int PCs = Math.min(dataSet.getSampleSize(), dataSet.getNumNumericalVars());
        PCs = Math.min(maxPCs, PCs);
        Vec t = getColumn(E);
        
        
        double tauOld = t.dot(t);
        for(int i = 1; i <= PCs; i++)
        {
            //1. Project X onto t to and the corresponding loading p
            //p = (E[i-1]' * t) / (t'*t) 
            Vec p = E.transposeMultiply(1.0, t);
            p.mutableDivide(tauOld);
            
            //2. Normalise loading vector p to length 1
            //p = p * (p'*p)^-0.5 
            p.mutableMultiply(Math.pow(p.dot(p), -0.5));
            
            //3. Project X onto p to find corresponding score vector t
            //t = (E[i-1] p)/(p'*p)
            t = E.multiply(p);
            t.mutableDivide(p.dot(p));
            
            scores.add(t);///t is a new vector each time from step 3, and does not get altered after this. So no copy needed
            loadings.add(p);//p is a new vecor each time created at step 1, and does not get altered after this. So no copy needed
            //4. Check for convergence.
            double tauNew = t.dot(t);
            if(Math.abs(tauNew-tauOld) <= threshold*tauNew)
                return;
            tauOld =  tauNew;
            
            //5. Remove the estimated PC component from E[i-1]
            Matrix.OuterProductUpdate(E, t, p, -1.0);
        }
        
        P  = new DenseMatrix(loadings.size(), loadings.get(0).length());
        for(int i = 0; i < loadings.size(); i++)
        {
            Vec pi = loadings.get(i);
            for(int j = 0; j < pi.length(); j++)
                P.set(i, j, pi.get(j));
        }
    }
    
    /**
     * Returns the first non zero column
     * @param x the matrix to get a column from
     * @return the first non zero column
     */
    private static Vec getColumn(Matrix x)
    {
        Vec t;
        
        for(int i = 0; i < x.cols(); i++)
        {
            t = x.getColumn(i);
            if(t.dot(t) > 0 )
                return t;
        }
        
        throw new ArithmeticException("Matrix is essentially zero");
    }

    public DataPoint transform(DataPoint dp)
    {
        DataPoint newDP = new DataPoint(
                P.multiply(dp.getNumericalValues()), 
                Arrays.copyOf(dp.getCategoricalValues(), dp.numCategoricalValues()), 
                CategoricalData.copyOf(dp.getCategoricalData()),
                dp.getWeight());
        return newDP;
    }
    
}
