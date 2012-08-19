
package jsat.distributions.kernels;

import java.util.*;
import jsat.linear.Vec;
import jsat.parameters.DoubleParameter;
import jsat.parameters.Parameter;

/**
 * Provides an implementation of the Sigmoid Kernel. 
 * 
 * @author Edward Raff
 */
public class SigmoidKernel implements KernelTrick
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
    
    @Override
    public double eval(Vec a, Vec b)
    {
        return Math.tanh(alpha*a.dot(b)+c);
    }
    
    List<Parameter> params = Collections.unmodifiableList(new ArrayList<Parameter>(2)
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
                    setC(val);
                    return true;
                }

                @Override
                public String getASCIIName()
                {
                    return "SigmoidKernel_C";
                }
            });
    }});
    
    Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

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
