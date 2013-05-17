
package jsat.distributions.kernels;

import java.util.Arrays;
import java.util.List;
import jsat.linear.Vec;
import jsat.parameters.DoubleParameter;
import jsat.parameters.Parameter;
import jsat.text.GreekLetters;

/**
 * Provides a kernel for the Radial Basis Function, which is of the form
 * <br>
 * k(x, y) = exp(-||x-y||<sup>2</sup>/(2*&sigma;<sup>2</sup>))
 * 
 * @author Edward Raff
 */
public class RBFKernel implements KernelTrick
{
    private double sigma;
    private double sigmaSqrd2Inv;

    /**
     * Creates a new RBF kernel
     * @param sigma the sigma parameter
     */
    public RBFKernel(double sigma)
    {
        setSigma(sigma);
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        if(a == b)//Same refrence means dist of 0, exp(0) = 1
            return 1;
        return Math.exp(-Math.pow(a.pNormDist(2, b),2) * sigmaSqrd2Inv);
    }

    /**
     * Sets the sigma parameter, which must be a positive value
     * @param sigma the sigma value
     */
    public void setSigma(double sigma)
    {
        if(sigma <= 0)
            throw new IllegalArgumentException("Sigma must be a positive constant, not " + sigma);
        this.sigma = sigma;
        this.sigmaSqrd2Inv = 0.5/(sigma*sigma);
    }

    public double getSigma()
    {
        return Math.sqrt(sigma/2);
    }
    
    

    @Override
    public String toString()
    {
        return "RBF Kernel( " + GreekLetters.sigma +" = " + sigma +")";
    }

    private Parameter param = new DoubleParameter() 
    {

        @Override
        public double getValue()
        {
            return getSigma();
        }

        @Override
        public boolean setValue(double val)
        {
            if(val <= 0 || Double.isInfinite(val))
                return false;
            setSigma(val);
            return true;
        }

        @Override
        public String getASCIIName()
        {
            return "RBFKernel_sigma";
        }
    };
    
    @Override
    public List<Parameter> getParameters()
    {
        return Arrays.asList(param);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        if(paramName.equals(param.getASCIIName()))
            return param;
        return null;
    }

    @Override
    public KernelTrick clone()
    {
        return new RBFKernel(sigma);
    }
}
