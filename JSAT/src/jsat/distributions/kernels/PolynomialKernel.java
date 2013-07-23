
package jsat.distributions.kernels;

import java.util.*;
import jsat.linear.Vec;
import jsat.parameters.DoubleParameter;
import jsat.parameters.Parameter;

/**
 * Provides a Polynomial Kernel of the form <br>
 * k(x,y) = (alpha * x.y + c)^d
 * @author Edward Raff
 */
public class PolynomialKernel extends BaseKernelTrick 
{
    private double degree;
    private double alpha;
    private double c;

    /**
     * Creates a new polynomial kernel
     * @param degree the degree of the polynomial 
     * @param alpha the term to scale the dot product by
     * @param c the additive term
     */
    public PolynomialKernel(double degree, double alpha, double c)
    {
        this.degree = degree;
        this.alpha = alpha;
        this.c = c;
    }

    /**
     * Defaults alpha = 1 and c = 1
     * @param degree the degree of the polynomial 
     */
    public PolynomialKernel(double degree)
    {
        this(degree, 1, 1);
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
     * Sets the degree of the polynomial 
     * @param d the degree of the polynomial 
     */
    public void setDegree(double d)
    {
        this.degree = d;
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
     * Returns the additive constant
     * @return the additive constant
     */
    public double getC()
    {
        return c;
    }

    /**
     * Returns the degree of the polynomial 
     * @return the degree of the polynomial 
     */
    public double getDegree()
    {
        return degree;
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        return Math.pow(c+a.dot(b)*alpha, degree);
    }

    @Override
    public String toString()
    {
        return "Polynomial Kernel ( degree="+degree + ", c=" + c + ", alpha=" + alpha + ")";
    }

    private List<Parameter> params = Collections.unmodifiableList(new ArrayList<Parameter>(3)
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
                    setAlpha(val);
                    return true;
                }

                @Override
                public String getASCIIName()
                {
                    return "PolynomialKernel_Alpha";
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
                    setC(val);
                    return true;
                }

                @Override
                public String getASCIIName()
                {
                    return "PolynomialKernel_C";
                }
            });
        add(new DoubleParameter() {

                @Override
                public double getValue()
                {
                    return getDegree();
                }

                @Override
                public boolean setValue(double val)
                {
                    setDegree(val);
                    return true;
                }

                @Override
                public String getASCIIName()
                {
                    return "PolynomialKernel_Degree";
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
    public PolynomialKernel clone()
    {
        return new PolynomialKernel(degree, alpha, c);
    }
}
