
package jsat.distributions.kernels;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class PolynomialKernel implements KernelFunction 
{
    private double d;

    public PolynomialKernel(double d)
    {
        this.d = d;
    }

    public double eval(Vec a, Vec b)
    {
        return Math.pow(1+a.dot(b), d);
    }

    @Override
    public String toString()
    {
        return d + "-degree Polynomial";
    }
    
    
}
