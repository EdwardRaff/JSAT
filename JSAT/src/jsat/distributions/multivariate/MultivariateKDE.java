
package jsat.distributions.multivariate;

import java.util.List;
import jsat.distributions.empirical.KernelDensityEstimator;
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
    /**
     * Returns the list of vectors that have a non zero contribution to the density of the query point <tt>x</tt>. 
     * Each vector is paired with its integer index from the original constructing list vectors, and a double 
     * indicating its weight given the kernel function in use. 
     * 
     * @param x the query point
     * @return the list of near by vectors and their weights
     */
    abstract public List<VecPaired<Double, VecPaired<Integer, Vec>>> getNearby(Vec x);
}
