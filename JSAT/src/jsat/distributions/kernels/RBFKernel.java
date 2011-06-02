
package jsat.distributions.kernels;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class RBFKernel implements KernelFunction
{
    private double sigma;

    public RBFKernel(double sigma)
    {
        this.sigma = sigma;
    }

    public double eval(Vec a, Vec b)
    {
        return -Math.exp(a.pNormDist(2, b) / (2*sigma*sigma));
    }

    @Override
    public String toString()
    {
        return "RBF Kernel";
    }
    
    

}
