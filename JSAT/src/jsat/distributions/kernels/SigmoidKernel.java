
package jsat.distributions.kernels;

import java.util.*;
import jsat.DataSet;
import jsat.distributions.Distribution;
import jsat.distributions.LogUniform;
import jsat.distributions.Uniform;
import jsat.linear.Vec;
import jsat.parameters.Parameter;

/**
 * Provides an implementation of the Sigmoid (Hyperbolic Tangent) Kernel, which 
 * is of the form:<br> k(x, y) = tanh(alpha * &lt; x, y &gt; +c)<br>
 * Technically, this kernel is not positive definite. 
 * 
 * @author Edward Raff
 */
public class SigmoidKernel extends BaseKernelTrick
{

    private static final long serialVersionUID = 8066799016611439349L;
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
    
    /**
     * Guesses a distribution for the &alpha; parameter
     *
     * @param d the data to get the guess for
     * @return a distribution for the &alpha; parameter
     */
    public static Distribution guessAlpha(DataSet d)
    {
        return new LogUniform(1e-12, 1e3);//from A Study on Sigmoid Kernels for SVM and the Training of non-PSD Kernels by SMO-type Methods
    }
    
    /**
     * Guesses a distribution for the &alpha; parameter
     *
     * @param d the data to get the guess for
     * @return a distribution for the &alpha; parameter
     */
    public static Distribution guessC(DataSet d)
    {
        return new Uniform(-2.4, 2.4);//from A Study on Sigmoid Kernels for SVM and the Training of non-PSD Kernels by SMO-type Methods
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
    public SigmoidKernel clone()
    {
        return new SigmoidKernel(alpha, c);
    }
}
