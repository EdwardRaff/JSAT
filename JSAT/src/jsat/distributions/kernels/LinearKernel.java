
package jsat.distributions.kernels;

import jsat.linear.Vec;

/**
 *
 * k(x,y) = x.y + x
 * 
 * @author Edward Raff
 */
public class LinearKernel implements KernelTrick
{

    private double c;

    public LinearKernel(double c)
    {
        this.c = c;
    }

    /**
     * Defaults c = 0 
     */
    public LinearKernel()
    {
        this(0);
    }

    public void setC(double c)
    {
        this.c = c;
    }

    public double getC()
    {
        return c;
    }
    
    
    public double eval(Vec a, Vec b)
    {
        return a.dot(b) + c;
    }

    @Override
    public String toString()
    {
        return "Linear";
    }
    
    
    
}
