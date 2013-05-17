
package jsat.distributions.kernels;

import java.util.*;
import jsat.linear.Vec;
import jsat.parameters.DoubleParameter;
import jsat.parameters.Parameter;

/**
 * Provides an implementation of the Sigmoid (Hyperbolic Tangent) Kernel, which 
 * is of the form:<br> k(x, y) = tanh(alpha * &lt; x, y &gt; +c)<br>
 * Technically, this kernel is not positive definite. 
 * 
 * @author Edward Raff
 */
public class SigmoidKernel implements KernelTrick
{
    private double alpha;
    private double c;

    /**
     * Creates a new Sigmoid Kernel
     * @param alpha the scaling factor for the dot product
     * @param C the additive constant
     */
    public SigmoidKernel(double alpha, double C)
    {
        this.alpha = alpha;
        this.c = C;
    }

    /**
     * Creates a new Sigmoid Kernel with a bias term of 1
     * @param alpha the scaling factor for the dot product
     */
    public SigmoidKernel(double alpha)
    {
        this(alpha, 1);
    }

    /**
     * Sets the scaling factor for the dot product, this is equivalent to 
     * multiplying each value in the data set by a constant factor 
     * @param alpha the scaling factor
     */
    public void setAlpha(double alpha)
    {
        if(Double.isInfinite(alpha) || Double.isNaN(alpha) || alpha == 0)
            throw new IllegalArgumentException("alpha must be a real non zero value, not " + alpha);
        this.alpha = alpha;
    }

    /**
     * Returns the scaling parameter
     * @return the scaling parameter
     */
    public double getAlpha()
    {
        return alpha;
    }

    /**
     * Sets the additive term, when set to one this is equivalent to adding a 
     * bias term of 1 to each vector. This is done after the scaling by 
     * {@link #setAlpha(double) alpha}. 
     * @param c the non negative additive term
     */
    public void setC(double c)
    {
        if(c < 0 || Double.isNaN(c) || Double.isInfinite(c))
            throw new IllegalArgumentException("C must be non negative, not " + c);
        this.c = c;
    }

    /**
     * Returns the additive constant
     * @return the additive constant
     */
    public double getC()
    {
        return c;
    }
    
    @Override
    public double eval(Vec a, Vec b)
    {
        return Math.tanh(alpha*a.dot(b)+c);
    }
    
    private List<Parameter> params = Collections.unmodifiableList(new ArrayList<Parameter>(2)
    {{
        add(new DoubleParameter() {

                @Override
                public double getValue()
                {
                    return getAlpha();
                }

                @Override
                public boolean setValue(double val)
                {
                    try
                    {
                        setAlpha(val);
                        return true;
                    }
                    catch (Exception ex)
                    {
                        return false;
                    }
                }

                @Override
                public String getASCIIName()
                {
                    return "SigmoidKernel_Alpha";
                }
            });
        add(new DoubleParameter() {

                @Override
                public double getValue()
                {
                    return getC();
                }

                @Override
                public boolean setValue(double val)
                {
                    try
                    {
                        setC(val);
                        return true;
                    }
                    catch (Exception ex)
                    {
                        return false;
                    }
                }

                @Override
                public String getASCIIName()
                {
                    return "SigmoidKernel_C";
                }
            });
    }});
    
    private Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

    @Override
    public List<Parameter> getParameters()
    {
        return params;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return paramMap.get(paramName);
    }

    @Override
    public SigmoidKernel clone()
    {
        return new SigmoidKernel(alpha, c);
    }
}
