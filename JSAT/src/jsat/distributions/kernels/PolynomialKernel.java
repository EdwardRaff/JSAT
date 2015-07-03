
package jsat.distributions.kernels;

import java.util.*;
import jsat.DataSet;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.discrete.UniformDiscrete;
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

    private static final long serialVersionUID = 9123109691782934745L;
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
    
    /**
     * Guesses the distribution to use for the degree parameter
     *
     * @param d the dataset to get the guess for
     * @return the guess for the degree parameter
     * @see #setDegree(double) 
     */
    public static Distribution guessDegree(DataSet d)
    {
        return new UniformDiscrete(2, 9);
    }
    
    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }

    @Override
    public PolynomialKernel clone()
    {
        return new PolynomialKernel(degree, alpha, c);
    }
}
