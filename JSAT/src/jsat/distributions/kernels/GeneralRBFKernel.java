package jsat.distributions.kernels;

import java.util.List;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * This class provides a generalization of the {@link RBFKernel} to arbitrary 
 * {@link DistanceMetric distance metrics}, and is of the form 
 * <i>exp(-d(x, y)<sup>2</sup>/(2 {@link #setSigma(double) &sigma;}<sup>2</sup>
 * ))</i>. So long as the distance metric is valid, the resulting kernel trick
 * will be a valid kernel. <br>
 * <br>
 * If the {@link EuclideanDistance} is used, then this becomes equivalent to the
 * {@link RBFKernel}. <br> 
 * <br>
 * Note, that since the {@link KernelTrick} has no concept of training - the 
 * distance metric can not require training either. A pre-trained metric can 
 * be admissible thought. 
 * 
 * @author Edward Raff
 */
public class GeneralRBFKernel extends DistanceMetricBasedKernel
{
    private double sigma;
    private double sigmaSqrd2Inv;

    /**
     * Creates a new Generic RBF Kernel
     * @param d the distance metric to use
     * @param sigma the standard deviation to use
     */
    public GeneralRBFKernel(DistanceMetric d, double sigma)
    {
        super(d);
        setSigma(sigma);
    }
    
    /**
     * Sets the kernel width parameter, which must be a positive value. Larger 
     * values indicate a larger width
     * 
     * @param sigma the sigma value
     */
    public void setSigma(double sigma)
    {
        if(sigma <= 0 || Double.isNaN(sigma) || Double.isInfinite(sigma))
            throw new IllegalArgumentException("Sigma must be a positive constant, not " + sigma);
        this.sigma = sigma;
        this.sigmaSqrd2Inv = 0.5/(sigma*sigma);
    }

    /**
     * 
     * @return the width parameter to use for the kernel 
     */
    public double getSigma()
    {
        return sigma;
    }
    
    @Override
    public KernelTrick clone()
    {
        return new GeneralRBFKernel(d.clone(), sigma);
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        double dist = d.dist(a, b);
        return Math.exp(-dist*dist * sigmaSqrd2Inv);
    }

    @Override
    public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        double dist = d.dist(a, b, qi, vecs, cache);
        return Math.exp(-dist*dist * sigmaSqrd2Inv);
        
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        double dist = d.dist(a, b, vecs, cache);
        return Math.exp(-dist*dist * sigmaSqrd2Inv);
    }
}
