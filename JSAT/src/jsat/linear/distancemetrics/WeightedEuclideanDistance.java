
package jsat.linear.distancemetrics;

import jsat.linear.Vec;
import jsat.linear.VecOps;
import jsat.math.Function;
import jsat.math.FunctionBase;

/**
 * Implements the weighted Euclidean distance such that d(a, b) =
 * <big>&accumulateSum;</big><sub>&forall; i &isin; |w|</sub> w<sub>i</sub> 
 * (x<sub>i</sub>-y<sub>i</sub>)<sup>2</sup> <br>
 * When used with a weight vector of ones, it degenerates into 
 * the {@link EuclideanDistance}. 
 * 
 * @author Edward Raff
 */
public class WeightedEuclideanDistance implements DistanceMetric
{
    private Vec w;
    private static final Function sqrtFunc = new FunctionBase() 
    {

        @Override
        public double f(Vec x)
        {
            final double xx = x.get(0);
            return xx*xx;
        }
    };

    /**
     * Creates a new weighted Euclidean distance metric using the 
     * given set of weights.
     * @param w the weight to apply to each variable
     */
    public WeightedEuclideanDistance(Vec w)
    {
        setWeight(w);
    }

    /**
     * Returns the weight vector used by this object. Altering the returned 
     * vector is visible to this object, so there is no need to set it again
     * using {@link #setWeight(jsat.linear.Vec) }. If you do not want to 
     * alter it, you will need to clone the returned object and modify that. 
     * @return the weight vector used by this object
     */
    public Vec getWeight()
    {
        return w;
    }

    /**
     * Sets the weight vector to use for the distance function
     * @param w the weight vector to use
     * @throws NullPointerException if {@code w} is null
     */
    public void setWeight(Vec w)
    {
        if(w == null)
            throw new NullPointerException();
        this.w = w;
    }

    @Override
    public double dist(Vec a, Vec b)
    {
        return Math.sqrt(VecOps.accumulateSum(w, a, b, sqrtFunc));
    }

    @Override
    public boolean isSymmetric()
    {
        return true;
    }

    @Override
    public boolean isSubadditive()
    {
        return true;
    }

    @Override
    public boolean isIndiscemible()
    {
        return true;
    }

    @Override
    public double metricBound()
    {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public DistanceMetric clone()
    {
        return new WeightedEuclideanDistance(w.clone());
    }

}
