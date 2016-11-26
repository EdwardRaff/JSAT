package jsat.distributions.kernels;

import java.util.List;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.utils.DoubleList;

/**
 * Many Kernels can be described in terms the L2 norm with some operations
 * performed on it. For example, the {@link RBFKernel} and
 * {@link RationalQuadraticKernel} can both be expressed in terms of the
 * L<sub>2</sub> norm of the two inputs. To simplify the addition of other
 * kernels based on the same norm, this base class exists.<br>
 * <br>
 * All Kernels extending this base kernel support acceleration. This is done making use of the fact that 
 * || x - y ||<sup>2</sup> = x x' +y y' - 2 x y' <br>
 * Thus the cached value for each vector is its self dot product. 
 * @author Edward Raff
 */
public abstract class BaseL2Kernel implements KernelTrick
{

    private static final long serialVersionUID = 2917497058710848085L;

    @Override
    abstract public double eval(Vec a, Vec b);

    @Override
    abstract public KernelTrick clone();

    @Override
    public boolean supportsAcceleration()
    {
        return true;
    }
    
    /**
     * Returns the squared L<sup>2</sup> norm between two points from the cache values. 
     * @param i the first index in the vector list
     * @param j the second index in the vector list
     * @param vecs the list of vectors that make the collection
     * @param cache the cache of values for each vector in the collection
     * @return the squared norm ||x<sub>i</sub>-x<sub>j</sub>||<sup>2</sup>
     */
    protected double getSqrdNorm(int i, int j, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return Math.pow(vecs.get(i).pNormDist(2.0, vecs.get(j)), 2);
        return cache.get(i)+cache.get(j)-2*vecs.get(i).dot(vecs.get(j));
    }
    
    /**
     * Returns the squared L<sup>2</sup> norm of the given point from the cache
     * @param i the index in the vector list to get the squared norm from
     * @param vecs the list of vectors that make the collection
     * @param cache the cache of values for each vector in the collection
     * @return the squared norm ||x<sub>i</sub>||<sup>2</sup>
     */
    protected double getSqrdNorm(int i, List<? extends Vec> vecs, List<Double> cache)
    {
        return cache.get(i);
    }
    
    /**
     * Returns the squared L<sup>2</sup> norm between a point in the cache and one with a provided qi value
     * @param i the index in the vector list
     * @param y the other vector 
     * @param qi the acceleration values for the other vector
     * @param vecs the list of vectors to make the collection
     * @param cache the cache of values for each vector in the collection
     * @return the squared norm ||x<sub>i</sub>-y||<sup>2</sup>
     */
    protected double getSqrdNorm(int i, Vec y, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return Math.pow(vecs.get(i).pNormDist(2.0, y), 2);
        return cache.get(i)+qi.get(0)-2*vecs.get(i).dot(y);
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> trainingSet)
    {
        DoubleList cache = new DoubleList(trainingSet.size());
        for(int i = 0; i < trainingSet.size(); i++)
            cache.add(trainingSet.get(i).dot(trainingSet.get(i)));
        return cache;
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        DoubleList dl = new DoubleList(1);
        dl.add(q.dot(q));
        return dl;
    }

    @Override
    public void addToCache(Vec newVec, List<Double> cache)
    {
        cache.add(newVec.dot(newVec));
    }

    @Override
    abstract public double eval(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache);

    @Override
    abstract public double eval(int a, int b, List<? extends Vec> trainingSet, List<Double> cache);

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
            if(alpha[i] != 0.0)
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
