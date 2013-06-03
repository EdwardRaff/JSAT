
package jsat.distributions.kernels;

import java.util.Arrays;
import java.util.List;
import jsat.linear.Vec;
import jsat.parameters.DoubleParameter;
import jsat.parameters.Parameter;
import jsat.text.GreekLetters;
import jsat.utils.DoubleList;

/**
 * Provides a kernel for the Radial Basis Function, which is of the form
 * <br>
 * k(x, y) = exp(-||x-y||<sup>2</sup>/(2*&sigma;<sup>2</sup>))
 * 
 * @author Edward Raff
 */
public class RBFKernel implements CacheAcceleratedKernel
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
    
    @Override
    public DoubleList getCache(List<? extends Vec> trainingSet)
    {
        DoubleList cache = new DoubleList(trainingSet.size());
        for(int i = 0; i < trainingSet.size(); i++)
            cache.add(trainingSet.get(i).dot(trainingSet.get(i)));
        return cache;
    }
    
    
    @Override
    public void addToCache(Vec newVec, List<Double> cache)
    {
        cache.add(newVec.dot(newVec));
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> trainingSet, List<Double> cache)
    {
        if(a == b)
            return 1;
        final double cache_a = cache.get(a);
        final double cache_b = cache.get(b);
        return Math.exp(-(cache_a - 2*trainingSet.get(a).dot(trainingSet.get(b))+cache_b)* sigmaSqrd2Inv);
    }

    @Override
    public double evalSum(List<? extends Vec> finalSet, List<Double> cache, double[] alpha, Vec y, int start, int end)
    {
        final double y_dot = y.dot(y);
        double sum = 0;
        
        for(int i = start; i < end; i++)
            sum += alpha[i] * Math.exp(-(cache.get(i) - 2*finalSet.get(i).dot(y)+y_dot)* sigmaSqrd2Inv);
        
        return sum;
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
        return sigma;
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
