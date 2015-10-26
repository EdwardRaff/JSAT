
package jsat.linear;

import static java.lang.Math.pow;
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
    private MatrixStatistics()
    {
        
    }
    
    /**
     * Computes the mean of the given data set.  
     * @param <V> the vector type
     * @param dataSet the list of vectors to compute the mean of
     * @return the mean of the vectors 
     */
    public static <V extends Vec> Vec meanVector(final List<V> dataSet)
    {
        if(dataSet.isEmpty()) {
          throw new ArithmeticException("Can not compute the mean of zero data points");
        }
        
        final Vec mean = new DenseVector(dataSet.get(0).length());
        meanVector(mean, dataSet);
        return mean;
    }
    
    /**
     * Computes the weighted mean of the given data set. 
     * 
     * @param dataSet the dataset to compute the mean from
     * @return the mean of the numeric vectors in the data set
     */
    public static Vec meanVector(final DataSet dataSet)
    {
        final DenseVector dv = new DenseVector(dataSet.getNumNumericalVars());
        meanVector(dv, dataSet);
        return dv;
    }
    
    /**
     * Computes the mean of the given data set. 
     * 
     * @param mean the zeroed out vector to store the mean in. Its contents will be altered
     * @param dataSet the set of data points to compute the mean from
     */
    public static <V extends Vec> void meanVector(final Vec mean, final List<V> dataSet)
    {
        if(dataSet.isEmpty()) {
          throw new ArithmeticException("Can not compute the mean of zero data points");
        } else if(dataSet.get(0).length() != mean.length()) {
          throw new ArithmeticException("Vector dimensions do not agree");
        }

        for (final Vec x : dataSet) {
          mean.mutableAdd(x);
        }
        mean.mutableDivide(dataSet.size());
    }
    
    /**
     * Computes the weighted mean of the data set
     * @param mean the zeroed out vector to store the mean in. Its contents will be altered
     * @param dataSet the set of data points to compute the mean from
     */
    public static void meanVector(final Vec mean, final DataSet dataSet)
    {
        if(dataSet.getSampleSize() == 0) {
          throw new ArithmeticException("Can not compute the mean of zero data points");
        }
        double sumOfWeights = 0;
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            final DataPoint dp = dataSet.getDataPoint(i);
            final double w = dp.getWeight();
            sumOfWeights += w;
            mean.mutableAdd(w, dp.getNumericalValues());
        }
        mean.mutableDivide(sumOfWeights);
    }
    
    public static <V extends Vec> Matrix covarianceMatrix(final Vec mean, final List<V> dataSet)
    {
        final Matrix coMatrix = new DenseMatrix(mean.length(), mean.length());
        covarianceMatrix(mean, coMatrix, dataSet);
        return coMatrix;
    }
    
    public static <V extends Vec> void covarianceMatrix(final Vec mean, final Matrix covariance, final List<V> dataSet)
    {
        if(!covariance.isSquare()) {
          throw new ArithmeticException("Storage for covariance matrix must be square");
        } else if(covariance.rows() != mean.length()) {
          throw new ArithmeticException("Covariance Matrix size and mean size do not agree");
        } else if(dataSet.isEmpty()) {
          throw new ArithmeticException("No data points to compute covariance from");
        } else if(mean.length() != dataSet.get(0).length()) {
          throw new ArithmeticException("Data vectors do not agree with mean and covariance matrix");
        }
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
        final Vec scratch = new DenseVector(mean.length());
        for (final Vec x : dataSet)
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
     * {@link #covarianceMatrix(jsat.linear.Vec, java.util.List) }
     * 
     * @param mean the mean of the distribution. 
     * @param dataSet the set of data points that contain vectors 
     * @param covariance the zeroed matrix to store the result in. Its values will be altered.
     */
    public static void covarianceMatrix(final Vec mean, final List<DataPoint> dataSet, final Matrix covariance)
    {
        double sumOfWeights = 0.0, sumOfSquaredWeights = 0.0;
        
        for(final DataPoint dp : dataSet)
        {
            sumOfWeights += dp.getWeight();
            sumOfSquaredWeights += Math.pow(dp.getWeight(), 2);
        }
        
        covarianceMatrix(mean, dataSet, covariance, sumOfWeights, sumOfSquaredWeights);
    }
    
    /**
     * Computes the weighted result for the covariance matrix of the given data set. 
     * If all weights have the same value, the result will come out equivalent to 
     * {@link #covarianceMatrix(jsat.linear.Vec, java.util.List) }
     * 
     * @param mean the mean of the distribution. 
     * @param dataSet the set of data points that contain vectors 
     * @param covariance the zeroed matrix to store the result in. Its values will be altered.
     * @param sumOfWeights the sum of each weight in <tt>dataSet</tt>
     * @param sumOfSquaredWeights  the sum of the squared weights in <tt>dataSet</tt>
     */
    public static void covarianceMatrix(final Vec mean, final List<DataPoint> dataSet, final Matrix covariance, final double sumOfWeights, final double sumOfSquaredWeights)
    {
        if (!covariance.isSquare()) {
          throw new ArithmeticException("Storage for covariance matrix must be square");
        } else if (covariance.rows() != mean.length()) {
          throw new ArithmeticException("Covariance Matrix size and mean size do not agree");
        } else if (dataSet.isEmpty()) {
          throw new ArithmeticException("No data points to compute covariance from");
        } else if (mean.length() != dataSet.get(0).getNumericalValues().length()) {
          throw new ArithmeticException("Data vectors do not agree with mean and covariance matrix");
        }

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

        final Vec scratch = new DenseVector(mean.length());

        for (int i = 0; i < dataSet.size(); i++)
        {
            final DataPoint dp = dataSet.get(i);
            final Vec x = dp.getNumericalValues();
            x.copyTo(scratch);
            scratch.mutableSubtract(mean);
            Matrix.OuterProductUpdate(covariance, scratch, scratch, dp.getWeight());
        }
        covariance.mutableMultiply(sumOfWeights / (Math.pow(sumOfWeights, 2) - sumOfSquaredWeights));
    }
    
    /**
     * Computes the weighted covariance matrix of the data set
     * @param mean the mean of the data set
     * @param dataSet the dataset to compute the covariance of
     * @return the covariance matrix of the data set
     */
    public static Matrix covarianceMatrix(final Vec mean, final DataSet dataSet)
    {
        final Matrix covariance = new DenseMatrix(mean.length(), mean.length());
        covarianceMatrix(mean, dataSet, covariance);
        return covariance;
    }
    
    /**
     * Computes the weighted covariance matrix of the given data set. 
     * @param mean the mean of the data set
     * @param dataSet the dataset to compute the covariance of
     * @param covariance the zeroed out matrix to store the result into 
     */
    public static void covarianceMatrix(final Vec mean, final DataSet dataSet, final Matrix covariance)
    {
        double sumOfWeights = 0.0, sumOfSquaredWeights = 0.0;
        
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            final DataPoint dp = dataSet.getDataPoint(i);
            sumOfWeights += dp.getWeight();
            sumOfSquaredWeights += Math.pow(dp.getWeight(), 2);
        }
        
        covarianceMatrix(mean, dataSet, covariance, sumOfWeights, sumOfSquaredWeights);
    }

    /**
     * Computes the weighted covariance matrix of the given data set. Superfluous calculations are avoided by having the call provide information. 
     * @param mean the mean of the data set
     * @param dataSet the dataset to compute the covariance of
     * @param covariance the zeroed out matrix to store the result into 
     * @param sumOfWeights the sum of the weights for each data point in the dataset
     * @param sumOfSquaredWeights the sum of the squared weights for each data point in the data set
     */
    public static void covarianceMatrix(final Vec mean, final DataSet dataSet, final Matrix covariance, final double sumOfWeights, final double sumOfSquaredWeights)
    {
        if (!covariance.isSquare()) {
          throw new ArithmeticException("Storage for covariance matrix must be square");
        } else if (covariance.rows() != mean.length()) {
          throw new ArithmeticException("Covariance Matrix size and mean size do not agree");
        } else if (dataSet.getSampleSize() == 0) {
          throw new ArithmeticException("No data points to compute covariance from");
        } else if (mean.length() != dataSet.getNumNumericalVars()) {
          throw new ArithmeticException("Data vectors do not agree with mean and covariance matrix");
        }

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

        final Vec scratch = new DenseVector(mean.length());

        for (int i = 0; i < dataSet.getSampleSize(); i++)
        {
            final DataPoint dp = dataSet.getDataPoint(i);
            final Vec x = dp.getNumericalValues();
            x.copyTo(scratch);
            scratch.mutableSubtract(mean);
            Matrix.OuterProductUpdate(covariance, scratch, scratch, dp.getWeight());
        }
        covariance.mutableMultiply(sumOfWeights / (Math.pow(sumOfWeights, 2) - sumOfSquaredWeights));
    }
    
    /**
     * Computes the weighted diagonal of the covariance matrix, which is the 
     * standard deviations of the columns of all values. 
     * 
     * @param means the already computed mean of the data set
     * @param diag the zeroed out vector to store the diagonal in. Its contents 
     * will be altered
     * @param dataset the data set to compute the covariance diagonal from
     */
    public static void covarianceDiag(final Vec means, final Vec diag, final DataSet dataset)
    {
        final int n = dataset.getSampleSize();
        final int d = dataset.getNumNumericalVars();
        
        final int[] nnzCounts = new int[d];
        double sumOfWeights = 0;
        for(int i = 0; i < n; i++)
        {
            final DataPoint dp = dataset.getDataPoint(i);
            final double w = dp.getWeight();
            sumOfWeights += w;
            final Vec x = dataset.getDataPoint(i).getNumericalValues();
            for(final IndexValue iv : x)
            {
                final int indx = iv.getIndex();
                nnzCounts[indx]++;
                diag.increment(indx, w*pow(iv.getValue()-means.get(indx), 2));
            }
        }
        
        //add zero observations
        for(int i = 0; i < nnzCounts.length; i++) {
          diag.increment(i, pow(means.get(i), 2)*(n-nnzCounts[i]) );
        }
        diag.mutableDivide(sumOfWeights);
    }
    
    /**
     * Computes the weighted diagonal of the covariance matrix, which is the 
     * standard deviations of the columns of all values. 
     * 
     * @param means the already computed mean of the data set
     * @param dataset the data set to compute the covariance diagonal from
     * @return the diagonal of the covariance matrix for the given data 
     */
    public static Vec covarianceDiag(final Vec means, final DataSet dataset)
    {
        final DenseVector diag = new DenseVector(dataset.getNumNumericalVars());
        covarianceDiag(means, diag, dataset);
        return diag;
    }
    
    /**
     * Computes the diagonal of the covariance matrix, which is the standard 
     * deviations of the columns of all values. 
     * 
     * @param <V> the type of the vector
     * @param means the already computed mean of the data set
     * @param diag the zeroed out vector to store the diagonal in. Its contents 
     * will be altered
     * @param dataset the data set to compute the covariance diagonal from
     */
    public static <V extends Vec> void covarianceDiag(final Vec means, final Vec diag, final List<V> dataset)
    {
        final int n = dataset.size();
        final int d = dataset.get(0).length();
        
        final int[] nnzCounts = new int[d];
        for(int i = 0; i < n; i++)
        {
            final Vec x = dataset.get(i);
            for(final IndexValue iv : x)
            {
                final int indx = iv.getIndex();
                nnzCounts[indx]++;
                diag.increment(indx, pow(iv.getValue()-means.get(indx), 2));
            }
        }
        
        //add zero observations
        for(int i = 0; i < nnzCounts.length; i++) {
          diag.increment(i, pow(means.get(i), 2)*(n-nnzCounts[i]) );
        }
        diag.mutableDivide(n);
    }
    
    /**
     * Computes the diagonal of the covariance matrix, which is the standard 
     * deviations of the columns of all values. 
     * 
     * @param <V>
     * @param means the already computed mean of the data set
     * @param dataset the data set to compute the covariance diagonal from
     * @return the diagonal of the covariance matrix for the given data 
     */
    public static <V extends Vec> Vec covarianceDiag(final Vec means, final List<V> dataset)
    {
        final int d = dataset.get(0).length();
        final DenseVector diag = new DenseVector(d);
        covarianceDiag(means, diag, dataset);;
        return diag;
    }
}
