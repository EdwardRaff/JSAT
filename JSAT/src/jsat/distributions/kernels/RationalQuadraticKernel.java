
package jsat.distributions.kernels;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class RationalQuadraticKernel implements KernelFunction
{
    private double c;

    public RationalQuadraticKernel(double c)
    {
        this.c = c;
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
        double dist = Math.pow(a.pNormDist(2, b), 2);
        return 1-dist/(dist+c);
    }
    
}
