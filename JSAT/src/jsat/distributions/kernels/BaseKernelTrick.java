
package jsat.distributions.kernels;

import java.util.List;
import jsat.linear.Vec;

/**
 * This provides a simple base implementation for the cache related methods in 
 * Kernel Trick. By default they will all call 
 * {@link #eval(jsat.linear.Vec, jsat.linear.Vec) } directly. For this reason 
 * {@link #supportsAcceleration() } defaults to returning false.  If the Kernel 
 * supports cache acceleration, {@link #evalSum(java.util.List, java.util.List, 
 * double[], jsat.linear.Vec, int, int) } will make use of the acceleration. 
 * All other methods must be overloaded. 
 * 
 * @author Edward Raff
 */
public abstract class BaseKernelTrick implements KernelTrick
{
    private static final long serialVersionUID = 7230585838672226751L;

    @Override
    public boolean supportsAcceleration()
    {
        return false;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> trainingSet)
    {
        return null;
    }
    
    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        return null;
    }

    @Override
    public void addToCache(Vec newVec, List<Double> cache)
    {
        
    }

    @Override
    public double eval(int a, int b, List<? extends Vec> trainingSet, List<Double> cache)
    {
        return eval(trainingSet.get(a), trainingSet.get(b));
    }

    @Override
    public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        return eval(vecs.get(a), b);
    }
    
    @Override
    public double evalSum(List<? extends Vec> finalSet, List<Double> cache, double[] alpha, Vec y, int start, int end)
    {
        return evalSum(finalSet, cache, alpha, y, getQueryInfo(y), start, end);
    }

    @Override
    public double evalSum(List<? extends Vec> finalSet, List<Double> cache, double[] alpha, Vec y, List<Double> qi, int start, int end)
    {
        double sum = 0;
        
        for(int i = start; i < end; i++)
            sum += alpha[i] * eval(i, y, qi, finalSet, cache);
        
        return sum;
    }

    @Override
    abstract public KernelTrick clone();
    
    @Override
    public boolean normalized()
    {
        return false;
    }
}
