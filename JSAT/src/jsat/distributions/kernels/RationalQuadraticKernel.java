
package jsat.distributions.kernels;

import java.util.Arrays;
import java.util.List;
import jsat.linear.Vec;
import jsat.parameters.DoubleParameter;
import jsat.parameters.Parameter;

/**
 * Provides an implementation of the Rational Quadratic Kernel. 
 * 
 * @author Edward Raff
 */
public class RationalQuadraticKernel implements KernelTrick
{
    private double c;

    public RationalQuadraticKernel(double c)
    {
        this.c = c;
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
        double dist = Math.pow(a.pNormDist(2, b), 2);
        return 1-dist/(dist+c);
    }
    
    Parameter param = new DoubleParameter() {

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
            return "RQKernel_C";
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
