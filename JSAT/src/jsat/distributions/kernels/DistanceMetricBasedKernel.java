package jsat.distributions.kernels;

import java.util.List;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;

/**
 * This abstract class provides the means of implementing a Kernel based off 
 * some {@link DistanceMetric}. This will pre-implement most of the methods of
 * the KernelTrick interface, including using the distance acceleration of the 
 * metric (if supported) when appropriate. 
 * 
 * @author Edward Raff
 */
public abstract class DistanceMetricBasedKernel implements KernelTrick
{

	private static final long serialVersionUID = 8395066824809874527L;
	/**
     * the distance metric to use for the Kernel
     */
    @ParameterHolder
    protected DistanceMetric d;

    /**
     * Creates a new distance based kerenel
     * @param d the distance metric to use
     */
    public DistanceMetricBasedKernel(DistanceMetric d)
    {
        this.d = d;
    }
    
    @Override
    abstract public KernelTrick clone();

    @Override
    public boolean supportsAcceleration()
    {
        return d.supportsAcceleration();
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> trainingSet)
    {
        return d.getAccelerationCache(trainingSet);
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        return d.getQueryInfo(q);
    }

    @Override
    public void addToCache(Vec newVec, List<Double> cache)
    {
        cache.addAll(d.getQueryInfo(newVec));
    }

    @Override
    public double evalSum(List<? extends Vec> finalSet, List<Double> cache, double[] alpha, Vec y, int start, int end)
    {
        return evalSum(finalSet, cache, alpha, y, d.getQueryInfo(y), start, end);
    }

    @Override
    public double evalSum(List<? extends Vec> finalSet, List<Double> cache, double[] alpha, Vec y, List<Double> qi, int start, int end)
    {
        double sum = 0;

        for (int i = start; i < end; i++)
            if (alpha[i] != 0)
                sum += alpha[i] * eval(i, y, qi, finalSet, cache);

        return sum;
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
    
}
