
package jsat.datatransform;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
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

    private static final long serialVersionUID = 8736609877239941617L;
    /**
     * The transposed matrix of the Principal Components
     */
    private Matrix P;
    private int maxPCs;
    private double threshold;
    
    /**
     * Creates a new object for performing PCA that stops at 50 principal components. This may not be optimal for any particular dataset
     *
     */
    public PCA()
    {
        this(50);
    }

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
     * Creates a new object for performing PCA
     *
     * @param maxPCs the maximum number of Principal Components to let the
     * algorithm learn. The algorithm may stop earlier if all the variance has
     * been explained, or the convergence threshold has been met. Note, the
     * computable maximum number of PCs is limited to the minimum of the number
     * of samples and the number of dimensions.
     */
    public PCA(int maxPCs)
    {
        this(maxPCs, 1e-4);
    }

    /**
     * Creates a new object for performing PCA
     * 
     * @param maxPCs the maximum number of Principal Components to let the algorithm learn. The algorithm may stop
     * earlier if all the variance has been explained, or the convergence threshold has been met. 
     * Note, the computable maximum number of PCs is limited to the minimum of the number of samples and the
     * number of dimensions. 
     * @param threshold a convergence threshold, any small value will work. Smaller values will 
     * not produce more accurate results, but may make the algorithm take longer if it would 
     * have terminated before <tt>maxPCs</tt> was reached.  
     */
    public PCA(int maxPCs, double threshold)
    {
        setMaxPCs(maxPCs);
        setThreshold(threshold);
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
     * have terminated before <tt>maxPCs</tt> was reached.  
     */
    public PCA(DataSet dataSet, int maxPCs, double threshold)
    {
        this(maxPCs, threshold);
        fit(dataSet);
    }

    @Override
    public void fit(DataSet dataSet)
    {
        //Edwad, don't forget. This is: Nonlinear Iterative PArtial Least Squares (NIPALS) algo
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
        Vec p = new DenseVector(E.cols());
        for(int i = 1; i <= PCs; i++)
        {
            for(int iter = 0; iter < 100; iter++)
            {
                //1. Project X onto t to and the corresponding loading p
                //p = (E[i-1]' * t) / (t'*t)
                p.zeroOut();
                E.transposeMultiply(1.0, t, p);
                p.mutableDivide(tauOld);

                //2. Normalise loading vector p to length 1
                //p = p * (p'*p)^-0.5 
                p.mutableMultiply(Math.pow(p.dot(p), -0.5));

                //3. Project X onto p to find corresponding score vector t
                //t = (E[i-1] p)/(p'*p)
                t = E.multiply(p);
                t.mutableDivide(p.dot(p));

                //4. Check for convergence.
                double tauNew = t.dot(t);
                if(iter > 0 && Math.abs(tauNew-tauOld) <= threshold*tauNew || iter == 99)//go at least one round
                {
                    scores.add(new DenseVector(t));
                    loadings.add(new DenseVector(p));
                    break;
                }
                tauOld =  tauNew;                
            }
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
     * Copy constructor
     * @param other the transform to copy
     */
    private PCA(PCA other)
    {
        if(other.P != null)
            this.P = other.P.clone();
        this.maxPCs = other.maxPCs;
        this.threshold = other.threshold;
    }

    /**
     * sets the maximum number of principal components to learn
     * @param maxPCs the maximum number of principal components to learn
     */
    public void setMaxPCs(int maxPCs)
    {
        if(maxPCs <= 0)
            throw new IllegalArgumentException("number of principal components must be a positive number, not " + maxPCs);
        this.maxPCs = maxPCs;
    }

    /**
     * 
     * @return maximum number of principal components to learn
     */
    public int getMaxPCs()
    {
        return maxPCs;
    }

    /**
     * 
     * @param threshold the threshold for convergence of the algorithm
     */
    public void setThreshold(double threshold)
    {
        if(threshold <= 0 || Double.isInfinite(threshold) || Double.isNaN(threshold))
            throw new IllegalArgumentException("threshold must be in the range (0, Inf), not " + threshold);
        this.threshold = threshold;
    }

    public double getThreshold()
    {
        return threshold;
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

    @Override
    public DataPoint transform(DataPoint dp)
    {
        DataPoint newDP = new DataPoint(
                P.multiply(dp.getNumericalValues()), 
                Arrays.copyOf(dp.getCategoricalValues(), dp.numCategoricalValues()), 
                CategoricalData.copyOf(dp.getCategoricalData()),
                dp.getWeight());
        return newDP;
    }

    @Override
    public DataTransform clone()
    {
        return new PCA(this);
    }
    
}
