
package jsat.distributions.kernels;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class SigmoidKernel implements KernelFunction
{
    double alpha;
    double c;

    public SigmoidKernel(double alpha, double C)
    {
        this.alpha = alpha;
        this.c = C;
    }

    public void setAlpha(double alpha)
    {
        this.alpha = alpha;
    }

    public double getAlpha()
    {
        return alpha;
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
        return Math.tanh(alpha*a.dot(b)+c);
    }
    
}
