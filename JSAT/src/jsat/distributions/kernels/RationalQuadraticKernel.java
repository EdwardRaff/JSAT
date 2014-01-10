
package jsat.distributions.kernels;

import java.util.List;
import jsat.linear.Vec;

/**
 * Provides an implementation of the Rational Quadratic Kernel, which is of the 
 * form: <br>
 * k(x, y) = 1 - ||x-y||<sup>2</sup> / (||x-y||<sup>2</sup> + c)
 * 
 * @author Edward Raff
 */
public class RationalQuadraticKernel extends BaseL2Kernel
{
    private double c;

    /**
     * Creates a new RQ Kernel
     * @param c the positive additive coefficient 
     */
    public RationalQuadraticKernel(double c)
    {
        this.c = c;
    }

    /**
     * Sets the positive additive coefficient
     * @param c the positive additive coefficient 
     */
    public void setC(double c)
    {
        if(c <= 0 || Double.isNaN(c) || Double.isInfinite(c))
            throw new IllegalArgumentException("coefficient must be in (0, Inf), not " + c);
        this.c = c;
    }

    /**
     * Returns the positive additive coefficient
     * @return the positive additive coefficient
     */
    public double getC()
    {
        return c;
    }
    
    @Override
    public double eval(Vec a, Vec b)
    {
        double dist = Math.pow(a.pNormDist(2, b), 2);
        return 1-dist/(dist+c);
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> trainingSet, List<Double> cache)
    {
        double dist = getSqrdNorm(a, b, trainingSet, cache);
        return 1-dist/(dist+c);
    }
        
    @Override
    public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        double dist = getSqrdNorm(a, b, qi, vecs, cache);
        return 1-dist/(dist+c);
    }
    
    @Override
    public RationalQuadraticKernel clone()
    {
        return new RationalQuadraticKernel(c);
    }
}
