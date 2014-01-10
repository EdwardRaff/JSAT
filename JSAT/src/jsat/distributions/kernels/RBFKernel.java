
package jsat.distributions.kernels;

import java.util.List;
import jsat.linear.Vec;
import jsat.text.GreekLetters;

/**
 * Provides a kernel for the Radial Basis Function, which is of the form
 * <br>
 * k(x, y) = exp(-||x-y||<sup>2</sup>/(2*&sigma;<sup>2</sup>))
 * 
 * @author Edward Raff
 */
public class RBFKernel extends BaseL2Kernel
{
    private double sigma;
    private double sigmaSqrd2Inv;

    /**
     * Creates a new RBF kernel
     * @param sigma the sigma parameter
     */
    public RBFKernel(double sigma)
    {
        setSigma(sigma);
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        if(a == b)//Same refrence means dist of 0, exp(0) = 1
            return 1;
        return Math.exp(-Math.pow(a.pNormDist(2, b),2) * sigmaSqrd2Inv);
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> trainingSet, List<Double> cache)
    {
        if(a == b)
            return 1; 
        return Math.exp(-getSqrdNorm(a, b, trainingSet, cache)* sigmaSqrd2Inv);
    }
    
    @Override
    public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        return Math.exp(-getSqrdNorm(a, b, qi, vecs, cache)* sigmaSqrd2Inv);
    }

    /**
     * Sets the sigma parameter, which must be a positive value
     * @param sigma the sigma value
     */
    public void setSigma(double sigma)
    {
        if(sigma <= 0)
            throw new IllegalArgumentException("Sigma must be a positive constant, not " + sigma);
        this.sigma = sigma;
        this.sigmaSqrd2Inv = 0.5/(sigma*sigma);
    }

    public double getSigma()
    {
        return sigma;
    }

    @Override
    public String toString()
    {
        return "RBF Kernel( " + GreekLetters.sigma +" = " + sigma +")";
    }

    @Override
    public KernelTrick clone()
    {
        return new RBFKernel(sigma);
    }
}
