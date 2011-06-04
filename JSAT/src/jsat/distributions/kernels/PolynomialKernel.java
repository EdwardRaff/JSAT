
package jsat.distributions.kernels;

import jsat.linear.Vec;

/**
 * k(x,y) = (alpha * x.y + c)^d
 * @author Edward Raff
 */
public class PolynomialKernel implements KernelFunction 
{
    private double d;
    private double alpha;
    private double c;

    public PolynomialKernel(double d, double alpha, double c)
    {
        this.d = d;
        this.alpha = alpha;
        this.c = c;
    }

    /**
     * Defaults alpha = 1 and c = 1
     * @param d 
     */
    public PolynomialKernel(double d)
    {
        this(d, 1, 1);
    }

    public void setAlpha(double alpha)
    {
        this.alpha = alpha;
    }

    public void setC(double c)
    {
        this.c = c;
    }

    public void setD(double d)
    {
        this.d = d;
    }

    public double getAlpha()
    {
        return alpha;
    }

    public double getC()
    {
        return c;
    }

    public double getD()
    {
        return d;
    }

    public double eval(Vec a, Vec b)
    {
        return Math.pow(c+a.dot(b)*alpha, d);
    }

    @Override
    public String toString()
    {
        return d + "-degree Polynomial";
    }
    
    
}
