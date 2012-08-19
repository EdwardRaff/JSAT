
package jsat.distributions.kernels;

import java.util.Collections;
import java.util.List;
import jsat.linear.Vec;
import jsat.parameters.Parameter;

/**
 * Provides a linear kernel function, which computes the normal dot product. 
 * k(x,y) = x.y + c
 * 
 * @author Edward Raff
 */
public class LinearKernel implements KernelTrick
{

    private double c;

    /**
     * Creates a new Linear Kernel that computes the dot product and offsets it by a specified valie
     * @param c the bias term for the dot product
     */
    public LinearKernel(double c)
    {
        this.c = c;
    }

    /**
     * Creates a new Linear Kernel with no bias
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
    
    
    @Override
    public double eval(Vec a, Vec b)
    {
        return a.dot(b) + c;
    }

    @Override
    public String toString()
    {
        return "Linear";
    }

    @Override
    public List<Parameter> getParameters()
    {
        return Collections.emptyList();
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return null;
    }

    @Override
    public LinearKernel clone()
    {
        return new LinearKernel(c);
    }
}
