
package jsat.distributions.kernels;

import java.util.*;
import jsat.linear.Vec;
import jsat.parameters.DoubleParameter;
import jsat.parameters.Parameter;

/**
 * Provides a Polynomial Kernel 
 * k(x,y) = (alpha * x.y + c)^d
 * @author Edward Raff
 */
public class PolynomialKernel implements KernelTrick 
{
    private double d;
    private double alpha;
    private double c;

    public PolynomialKernel(double d, double alpha, double c)
    {
        this.d = d;
        this.alpha = alpha;
        this.c = c;
    }

    /**
     * Defaults alpha = 1 and c = 1
     * @param d 
     */
    public PolynomialKernel(double d)
    {
        this(d, 1, 1);
    }

    public void setAlpha(double alpha)
    {
        this.alpha = alpha;
    }

    public void setC(double c)
    {
        this.c = c;
    }

    public void setD(double d)
    {
        this.d = d;
    }

    public double getAlpha()
    {
        return alpha;
    }

    public double getC()
    {
        return c;
    }

    public double getD()
    {
        return d;
    }

    @Override
    public double eval(Vec a, Vec b)
    {
        return Math.pow(c+a.dot(b)*alpha, d);
    }

    @Override
    public String toString()
    {
        return d + "-degree Polynomial";
    }

    List<Parameter> params = Collections.unmodifiableList(new ArrayList<Parameter>(3)
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
                    return "PolyKF_Alpha";
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
                    return "PolyKF_C";
                }
            });
        add(new DoubleParameter() {

                @Override
                public double getValue()
                {
                    return getD();
                }

                @Override
                public boolean setValue(double val)
                {
                    setD(val);
                    return true;
                }

                @Override
                public String getASCIIName()
                {
                    return "PolyKF_D";
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
    public PolynomialKernel clone()
    {
        return new PolynomialKernel(d, alpha, c);
    }
}
