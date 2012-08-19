
package jsat.distributions.kernels;

import java.util.Arrays;
import java.util.List;
import jsat.linear.Vec;
import jsat.parameters.DoubleParameter;
import jsat.parameters.Parameter;

/**
 * Provides a kernel for the Radial Basis Function. 
 * @author Edward Raff
 */
public class RBFKernel implements KernelTrick
{
    private double sigma;

    public RBFKernel(double sigma)
    {
        this.sigma = sigma;
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        return -Math.exp(a.pNormDist(2, b) / (2*sigma*sigma));
    }

    @Override
    public String toString()
    {
        return "RBF Kernel";
    }

    private Parameter param = new DoubleParameter() {

        @Override
        public double getValue()
        {
            return sigma;
        }

        @Override
        public boolean setValue(double val)
        {
            sigma = val;
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
