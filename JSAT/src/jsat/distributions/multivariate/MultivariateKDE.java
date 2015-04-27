
package jsat.distributions.multivariate;

import java.util.List;
import jsat.distributions.empirical.KernelDensityEstimator;
import jsat.distributions.empirical.kernelfunc.KernelFunction;
import jsat.linear.Vec;
import jsat.linear.VecPaired;

/**
 * There are several methods of generalizing the {@link KernelDensityEstimator} to the multivariate case. 
 * This class provides a contract for implementations that provide a generalization of the KDE.  
 * 
 * @author Edward Raff
 */
abstract public class MultivariateKDE extends MultivariateDistributionSkeleton
{

	private static final long serialVersionUID = 614136649331326270L;

	/**
     * Returns the list of vectors that have a non zero contribution to the density of the query point <tt>x</tt>. 
     * Each vector is paired with its integer index from the original constructing list vectors, and a double 
     * indicating its weight given the kernel function in use. 
     * 
     * @param x the query point
     * @return the list of near by vectors and their weights
     */
    abstract public List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> getNearby(Vec x);
    
    /**
     * Returns the list of vectors that have a non zero contribution to the density of the query point <tt>x</tt>. 
     * Each vector is paired with its integer index from the original constructing list vectors, and a double 
     * indicating its distance from the query point divided by the bandwidth of the point. 
     * 
     * @param x the query point
     * @return the list of near by vectors and their weights
     */
    abstract public List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> getNearbyRaw(Vec x);
    
    /**
     * 
     * @return the kernel function used
     */
    abstract public KernelFunction getKernelFunction();
    
    /**
     * A caller may want to increase or decrease the bandwidth after training 
     * has been completed to get smoother model, or decrease it to observe 
     * behavior. This method will scaled the bandwidth of each data point by the given factor
     * @param scale the value to scale the bandwidth used 
     */
    abstract public void scaleBandwidth(double scale);

    @Override
    abstract public MultivariateKDE clone();
}
