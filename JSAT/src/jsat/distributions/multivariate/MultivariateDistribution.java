
package jsat.distributions.multivariate;

import java.io.Serializable;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;

/**
 * This interface represents the contract that any continuous multivariate distribution must implement
 * 
 * @author Edward Raff
 */
public interface MultivariateDistribution extends Cloneable, Serializable
{
    /**
     * Computes the log of the probability density function. If the 
     * probability of the input is zero, the log of zero would be 
     * {@link Double#NEGATIVE_INFINITY}. Instead, -{@link Double#MAX_VALUE} is returned. 
     * 
     * @param x the array for the vector the get the log probability of
     * @return the log of the probability. 
     * @throws ArithmeticException if the vector is not the correct length, or the distribution has not yet been set
     */
    public double logPdf(double... x);
    
    /**
     * Computes the log of the probability density function. If the 
     * probability of the input is zero, the log of zero would be 
     * {@link Double#NEGATIVE_INFINITY}. Instead, -{@link Double#MAX_VALUE} is returned. 
     * 
     * @param x the vector the get the log probability of
     * @return the log of the probability. 
     * @throws ArithmeticException if the vector is not the correct length, or the distribution has not yet been set
     */
    public double logPdf(Vec x);
    
    /**
     * Returns the probability of a given vector from this distribution. By definition, 
     * the probability will always be in the range [0, 1]. 
     * 
     * @param x the array of the vector the get the log probability of
     * @return the probability 
     * @throws ArithmeticException if the vector is not the correct length, or the distribution has not yet been set
     */
    public double pdf(double... x);
    
    /**
     * Returns the probability of a given vector from this distribution. By definition, 
     * the probability will always be in the range [0, 1]. 
     * 
     * @param x the vector the get the log probability of
     * @return the probability 
     * @throws ArithmeticException if the vector is not the correct length, or the distribution has not yet been set
     */
    public double pdf(Vec x);
    
    /**
     * Sets the parameters of the distribution to attempt to fit the given list of vectors.
     * All vectors are assumed to have the same weight. 
     * @param <V> the vector type
     * @param dataSet the list of data points
     * @return <tt>true</tt> if the distribution was fit to the data, or <tt>false</tt> 
     * if the distribution could not be fit to the data set. 
     */
    public <V extends Vec> boolean setUsingData(List<V> dataSet);
    
    /**
     * Sets the parameters of the distribution to attempt to fit the given list of vectors.
     * All vectors are assumed to have the same weight. 
     * @param <V> the vector type
     * @param dataSet the list of data points
     * @param threadpool the source of threads for computation
     * @return <tt>true</tt> if the distribution was fit to the data, or <tt>false</tt> 
     * if the distribution could not be fit to the data set. 
     */
    public <V extends Vec> boolean setUsingData(List<V> dataSet, ExecutorService threadpool);
    
    /**
     * Sets the parameters of the distribution to attempt to fit the given list of data points. 
     * The {@link DataPoint#getWeight()  weights} of the data points will be used.
     * 
     * @param dataPoints the list of data points to use
     * @return <tt>true</tt> if the distribution was fit to the data, or <tt>false</tt> 
     * if the distribution could not be fit to the data set. 
     */
    public boolean setUsingDataList(List<DataPoint> dataPoints);
    
    /**
     * Sets the parameters of the distribution to attempt to fit the given list of data points. 
     * The {@link DataPoint#getWeight()  weights} of the data points will be used.
     * 
     * @param dataPoints the list of data points to use
     * @param threadpool the source of threads for computation
     * @return <tt>true</tt> if the distribution was fit to the data, or <tt>false</tt> 
     * if the distribution could not be fit to the data set. 
     */
    public boolean setUsingDataList(List<DataPoint> dataPoints, ExecutorService threadpool);
    
    /**
     * Sets the parameters of the distribution to attempt to fit the given list of data points. 
     * The {@link DataPoint#getWeight()  weights} of the data points will be used.
     * 
     * @param dataSet the data set to use
     * @return <tt>true</tt> if the distribution was fit to the data, or <tt>false</tt> 
     * if the distribution could not be fit to the data set. 
     */
    public boolean setUsingData(DataSet dataSet);
    
    /**
     * Sets the parameters of the distribution to attempt to fit the given list of data points. 
     * The {@link DataPoint#getWeight()  weights} of the data points will be used.
     * 
     * @param dataSet the data set to use
     * @param threadpool the source of threads for computation
     * @return <tt>true</tt> if the distribution was fit to the data, or <tt>false</tt> 
     * if the distribution could not be fit to the data set. 
     */
    public boolean setUsingData(DataSet dataSet, ExecutorService threadpool);

    public MultivariateDistribution clone();
    
    /**
     * Performs sampling on the current distribution. 
     * @param count the number of iid samples to draw
     * @param rand the source of randomness 
     * @return a list of sample vectors from this distribution 
     */
    public List<Vec> sample(int count, Random rand);
}
