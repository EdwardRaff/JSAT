
package jsat.linear.distancemetrics;

import jsat.classifiers.NearestNeighbour;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.PolynomialKernel;
import jsat.distributions.kernels.RBFKernel;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class KernelDistance implements DistanceMetric
{
    KernelTrick kf;

    public KernelDistance(KernelTrick kf)
    {
        this.kf = kf;
    }
    
    

    /**
     * Returns the square of the distance function expanded as kernel methods. 
     * <br>
     * d<sup>2</sup>(x,y) = K(x,x) - 2*K(x,y) + K(y,y)
     * 
     * <br><br>
     * Special Notes:<br>
     * The use of {@link RBFKernel} or {@link PolynomialKernel} of degree 1 
     * in the {@link NearestNeighbour} classifier will degenerate into the
     * normal nearest neighbor algorithm. 
     * 
     * @param a the first vector
     * @param b the second vector
     * @return the distance metric based on a kernel function
     */
    public double dist(Vec a, Vec b)
    {
        return kf.eval(a, a) - 2*kf.eval(a, b) + kf.eval(b, b);
    }
    
}
