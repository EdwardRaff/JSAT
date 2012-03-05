
package jsat.linear;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 * This class provides methods useful for statistical operations that involve matrices and vectors. 
 * 
 * @author Edward Raff
 */
public class MatrixStatistics
{
    
    public static <V extends Vec> Vec MeanVector(List<V> dataSet)
    {
        if(dataSet.isEmpty())
            throw new ArithmeticException("Can not compute the mean of zero data points");
        
        Vec mean = new DenseVector(dataSet.get(0).length());
        MeanVector(mean, dataSet);
        return mean;
    }
    
    /**
     * Computes the mean of the given data set. 
     * 
     * @param mean the zeroed out vector to store the mean in. It's contents will be altered
     * @param dataSet the set of data points to compute the mean from
     */
    public static <V extends Vec> void MeanVector(Vec mean, List<V> dataSet)
    {
        if(dataSet.isEmpty())
            throw new ArithmeticException("Can not compute the mean of zero data points");
        else if(dataSet.get(0).length() != mean.length())
            throw new ArithmeticException("Vector dimensions do not agree");

        for (Vec x : dataSet)
            mean.mutableAdd(x);
        mean.mutableDivide(dataSet.size());
    }
    
    public static <V extends Vec> Matrix CovarianceMatrix(Vec mean, List<V> dataSet)
    {
        Matrix coMatrix = new DenseMatrix(mean.length(), mean.length());
        CovarianceMatrix(mean, coMatrix, dataSet);
        return coMatrix;
    }
    
    public static <V extends Vec> void CovarianceMatrix(Vec mean, Matrix covariance, List<V> dataSet)
    {
        if(!covariance.isSquare())
            throw new ArithmeticException("Storage for covariance matrix must be square");
        else if(covariance.rows() != mean.length())
            throw new ArithmeticException("Covariance Matrix size and mean size do not agree");
        else if(dataSet.isEmpty())
            throw new ArithmeticException("No data points to compute covariance from");
        else if(mean.length() != dataSet.get(0).length())
            throw new ArithmeticException("Data vectors do not agree with mean and covariance matrix");
        /**
         * Covariance definition
         * 
         *   n
         * =====                    T 
         * \     /     _\  /     _\
         *  >    |x  - x|  |x  - x|
         * /     \ i    /  \ i    /
         * =====
         * i = 1
         * 
         */
        Vec scratch = new DenseVector(mean.length());
        for (Vec x : dataSet)
        {
            x.copyTo(scratch);
            scratch.mutableSubtract(mean);
            Matrix.OuterProductUpdate(covariance, scratch, scratch, 1.0);
        }
        covariance.mutableMultiply(1.0 / (dataSet.size() - 1.0));
    }
    
    /**
     * Computes the weighted result for the covariance matrix of the given data set. 
     * If all weights have the same value, the result will come out equivalent to 
     * {@link #CovarianceMatrix(jsat.linear.Vec, java.util.List) }
     * 
     * @param mean the mean of the distribution. 
     * @param dataSet the set of data points that contain vectors 
     * @param covariance the zeroed matrix to store the result in. Its values will be altered.
     */
    public static void CovarianceMatrix(Vec mean, List<DataPoint> dataSet, Matrix covariance)
    {
        double sumOfWeights = 0.0, sumOfSquaredWeights = 0.0;
        
        for(DataPoint dp : dataSet)
        {
            sumOfWeights += dp.getWeight();
            sumOfSquaredWeights += Math.pow(dp.getWeight(), 2);
        }
        
        CovarianceMatrix(mean, dataSet, covariance, sumOfWeights, sumOfSquaredWeights);
    }
    
    /**
     * Computes the weighted result for the covariance matrix of the given data set. 
     * If all weights have the same value, the result will come out equivalent to 
     * {@link #CovarianceMatrix(jsat.linear.Vec, java.util.List) }
     * 
     * @param mean the mean of the distribution. 
     * @param dataSet the set of data points that contain vectors 
     * @param covariance the zeroed matrix to store the result in. Its values will be altered.
     * @param sumOfWeights the sum of each weight in <tt>dataSet</tt>
     * @param sumOfSquaredWeights  the sum of the squared weights in <tt>dataSet</tt>
     */
    public static void CovarianceMatrix(Vec mean, List<DataPoint> dataSet, Matrix covariance, double sumOfWeights, double sumOfSquaredWeights)
    {
        if (!covariance.isSquare())
            throw new ArithmeticException("Storage for covariance matrix must be square");
        else if (covariance.rows() != mean.length())
            throw new ArithmeticException("Covariance Matrix size and mean size do not agree");
        else if (dataSet.isEmpty())
            throw new ArithmeticException("No data points to compute covariance from");
        else if (mean.length() != dataSet.get(0).getNumericalValues().length())
            throw new ArithmeticException("Data vectors do not agree with mean and covariance matrix");

        /**
         * Weighted definition of the covariance matrix 
         * 
         *          n
         *        =====
         *        \
         *         >    w
         *        /      i          n
         *        =====           =====                      T
         *        i = 1           \        /     _\  /     _\
         * ----------------------  >    w  |x  - x|  |x  - x|
         *           2            /      i \ i    /  \ i    /
         * /  n     \      n      =====
         * |=====   |    =====    i = 1
         * |\       |    \      2
         * | >    w |  -  >    w
         * |/      i|    /      i
         * |=====   |    =====
         * \i = 1   /    i = 1
         */

        Vec scratch = new DenseVector(mean.length());

        for (int i = 0; i < dataSet.size(); i++)
        {
            DataPoint dp = dataSet.get(i);
            Vec x = dp.getNumericalValues();
            x.copyTo(scratch);
            scratch.mutableSubtract(mean);
            Matrix.OuterProductUpdate(covariance, scratch, scratch, dp.getWeight());
        }
        covariance.mutableMultiply(sumOfWeights / (Math.pow(sumOfWeights, 2) - sumOfSquaredWeights));
    }
    
    public static Matrix CovarianceMatrix(Vec mean, DataSet dataSet)
    {
        Matrix covariance = new DenseMatrix(mean.length(), mean.length());
        CovarianceMatrix(mean, dataSet, covariance);
        return covariance;
    }
    
    public static void CovarianceMatrix(Vec mean, DataSet dataSet, Matrix covariance)
    {
        double sumOfWeights = 0.0, sumOfSquaredWeights = 0.0;
        
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            DataPoint dp = dataSet.getDataPoint(i);
            sumOfWeights += dp.getWeight();
            sumOfSquaredWeights += Math.pow(dp.getWeight(), 2);
        }
        
        CovarianceMatrix(mean, dataSet, covariance, sumOfWeights, sumOfSquaredWeights);
    }

    
    public static void CovarianceMatrix(Vec mean, DataSet dataSet, Matrix covariance, double sumOfWeights, double sumOfSquaredWeights)
    {
        if (!covariance.isSquare())
            throw new ArithmeticException("Storage for covariance matrix must be square");
        else if (covariance.rows() != mean.length())
            throw new ArithmeticException("Covariance Matrix size and mean size do not agree");
        else if (dataSet.getSampleSize() == 0)
            throw new ArithmeticException("No data points to compute covariance from");
        else if (mean.length() != dataSet.getNumNumericalVars())
            throw new ArithmeticException("Data vectors do not agree with mean and covariance matrix");

        /**
         * Weighted definition of the covariance matrix 
         * 
         *          n
         *        =====
         *        \
         *         >    w
         *        /      i          n
         *        =====           =====                      T
         *        i = 1           \        /     _\  /     _\
         * ----------------------  >    w  |x  - x|  |x  - x|
         *           2            /      i \ i    /  \ i    /
         * /  n     \      n      =====
         * |=====   |    =====    i = 1
         * |\       |    \      2
         * | >    w |  -  >    w
         * |/      i|    /      i
         * |=====   |    =====
         * \i = 1   /    i = 1
         */

        Vec scratch = new DenseVector(mean.length());

        for (int i = 0; i < dataSet.getSampleSize(); i++)
        {
            DataPoint dp = dataSet.getDataPoint(i);
            Vec x = dp.getNumericalValues();
            x.copyTo(scratch);
            scratch.mutableSubtract(mean);
            Matrix.OuterProductUpdate(covariance, scratch, scratch, dp.getWeight());
        }
        covariance.mutableMultiply(sumOfWeights / (Math.pow(sumOfWeights, 2) - sumOfSquaredWeights));
    }
}
