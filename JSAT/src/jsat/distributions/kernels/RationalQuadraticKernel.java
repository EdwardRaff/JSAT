
package jsat.distributions.kernels;

import java.util.Arrays;
import java.util.List;
import jsat.linear.Vec;
import jsat.parameters.DoubleParameter;
import jsat.parameters.Parameter;

/**
 * Provides an implementation of the Rational Quadratic Kernel, which is of the 
 * form: <br>
 * k(x, y) = 1 - ||x-y||<sup>2</sup> / (||x-y||<sup>2</sup> + c)
 * 
 * @author Edward Raff
 */
public class RationalQuadraticKernel implements KernelTrick
{
    private double c;

    /**
     * Creates a new RQ Kernel
     * @param c the positive additive coefficient 
     */
    public RationalQuadraticKernel(double c)
    {
        this.c = c;
    }

    /**
     * Sets the positive additive coefficient
     * @param c the positive additive coefficient 
     */
    public void setC(double c)
    {
        if(c <= 0 || Double.isNaN(c) || Double.isInfinite(c))
            throw new IllegalArgumentException("coefficient must be in (0, Inf), not " + c);
        this.c = c;
    }

    /**
     * Returns the positive additive coefficient
     * @return the positive additive coefficient
     */
    public double getC()
    {
        return c;
    }
    
    @Override
    public double eval(Vec a, Vec b)
    {
        if(a == b)//d(x,y) = 0, 0 / c = 0, 1-0 = 1
            return 1;
        double dist = Math.pow(a.pNormDist(2, b), 2);
        return 1-dist/(dist+c);
    }
    
    private Parameter param = new DoubleParameter() 
    {

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
            catch(Exception ex)
            {
                return false;
            }
        }

        @Override
        public String getASCIIName()
        {
            return "RationalQuadraticKernel_C";
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
    public RationalQuadraticKernel clone()
    {
        return new RationalQuadraticKernel(c);
    }
}
