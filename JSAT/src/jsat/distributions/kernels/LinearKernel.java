
package jsat.distributions.kernels;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class LinearKernel implements KernelFunction
{

    public double eval(Vec a, Vec b)
    {
        return a.dot(b);
    }

    @Override
    public String toString()
    {
        return "Linear";
    }
    
    
    
}
