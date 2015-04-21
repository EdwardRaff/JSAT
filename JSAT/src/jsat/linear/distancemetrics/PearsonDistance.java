
package jsat.linear.distancemetrics;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.linear.IndexValue;
import jsat.linear.Vec;

/**
 * A valid distance metric formed from the Pearson Correlation between two vectors.
 * The distance in the range of [0, 1]. 
 * 
 * @author Edward Raff
 */
public class PearsonDistance implements DistanceMetric
{

	private static final long serialVersionUID = 1090726755301934198L;
	private boolean bothNonZero;
    private boolean absoluteDistance;

    /**
     * Creates a new standard Pearson Distance that does not ignore zero values 
     * and anti-correlated values are considered far away. 
     */
    public PearsonDistance()
    {
        this(false, false);
    }

    /**
     * Creates a new Pearson Distance object
     * @param bothNonZero {@code true} if non zero values should be treated as 
     * "missing" or "no vote", and will not contribute. But this will not 
     * change the mean value used. {@code false} produces the standard Pearson value. 
     * @param absoluteDistance {@code true} to use the absolute correlation, meaning 
     * correlated and anti-correlated values will have the same distance. 
     */
    public PearsonDistance(boolean bothNonZero, boolean absoluteDistance)
    {
        this.bothNonZero = bothNonZero;
        this.absoluteDistance = absoluteDistance;
    }

    @Override
    public double dist(Vec a, Vec b)
    {
        double r = correlation(a, b, bothNonZero);
        if(Double.isNaN(r))
            return Double.MAX_VALUE;
        if(absoluteDistance)
            return Math.sqrt(1-r*r);
        else
            return Math.sqrt((1-r)*0.5);
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
        return 1;
    }

    @Override
    public PearsonDistance clone()
    {
        return new PearsonDistance(bothNonZero, absoluteDistance);
    }
    
    /**
     * Computes the Pearson correlation between two vectors. If one of the vectors is all zeros, the result is undefined. 
     * In cases where both are zero vectors, 1 will be returned to indicate they are the same. In cases where one of the 
     * numerator coefficients is zero, its value will be bumped up to an epsilon to provide a near result. <br>
     * <br>
     * In cases where {@code bothNonZero} is {@code true}, and the vectors have no overlapping non zero values, 0 will
     * be returned. 
     * @param a the first vector
     * @param b the second vector
     * @param bothNonZero {@code false} is the normal Pearson correlation. {@code true} will make the computation ignore 
     * all indexes where one of the values is zero, the mean will be from all non zero values in each vector. 
     * @return the Pearson correlation in [-1, 1]
     */
    public static double correlation(Vec a, Vec b, boolean bothNonZero)
    {
        final double aMean;
        final double bMean;
        if(bothNonZero)
        {
            aMean = a.sum()/a.nnz();
            bMean = b.sum()/b.nnz();
        }
        else
        {
            aMean = a.mean();
            bMean = b.mean();
        }

        double r = 0;
        double aSqrd = 0, bSqrd = 0;

        if (a.isSparse() || b.isSparse())
        {
            Iterator<IndexValue> aIter = a.getNonZeroIterator();
            Iterator<IndexValue> bIter = b.getNonZeroIterator();

            //if one is empty, then a zero forms on the denomrinator
            if (!aIter.hasNext() && !bIter.hasNext())
                return 1;
            if (!aIter.hasNext() || !bIter.hasNext())
                return Double.MAX_VALUE;

            IndexValue aCur = null;
            IndexValue bCur = null;

            boolean newA = true, newB = true;
            int lastObservedIndex = -1;
            do
            {

                if (newA)
                {
                    if (!aIter.hasNext())
                        break;
                    aCur = aIter.next();
                    newA = false;
                }
                if (newB)
                {
                    if (!bIter.hasNext())
                        break;
                    bCur = bIter.next();
                    newB = false;
                }

                if (aCur.getIndex() == bCur.getIndex())
                {
                    //accumulate skipped positions where both are zero
                    if(!bothNonZero)
                        r += aMean * bMean * (aCur.getIndex()-lastObservedIndex - 1);
                    lastObservedIndex = aCur.getIndex();

                    double aVal = aCur.getValue() - aMean;
                    double bVal = bCur.getValue() - bMean;
                    r += aVal * bVal;

                    aSqrd += aVal * aVal;
                    bSqrd += bVal * bVal;

                    newA = newB = true;
                }
                else if (aCur.getIndex() > bCur.getIndex())
                {
                    if (!bothNonZero)
                    {
                        //accumulate skipped positions where both are zero
                        r += aMean * bMean * (bCur.getIndex()-lastObservedIndex - 1);
                        lastObservedIndex = bCur.getIndex();

                        double bVal = bCur.getValue() - bMean;
                        r += -aMean * bVal;
                        bSqrd += bVal * bVal;
                    }
                    newB = true;
                }
                else if (aCur.getIndex() < bCur.getIndex())
                {
                    if (!bothNonZero)
                    {
                        //accumulate skipped positions where both are zero
                        r += aMean * bMean * (aCur.getIndex()-lastObservedIndex - 1);
                        lastObservedIndex = aCur.getIndex();
                    
                        double aVal = aCur.getValue() - aMean;
                        r += aVal * -bMean;
                        aSqrd += aVal * aVal;
                    }
                    newA = true;
                }
            }
            while (true);

            if (!bothNonZero)
            {
                //only one of the loops bellow will execute
                while (!newA || (newA && aIter.hasNext()))
                {
                    if(newA)
                        aCur = aIter.next();
                    //accumulate skipped positions where both are zero
                    r += aMean * bMean * (aCur.getIndex()-lastObservedIndex - 1);
                    lastObservedIndex = aCur.getIndex();

                    double aVal = aCur.getValue() - aMean;
                    r += aVal * -bMean;
                    aSqrd += aVal * aVal;
                    newA = true;
                }

                while (!newB || (newB && bIter.hasNext()))
                {
                    if(newB)
                        bCur = bIter.next();
                    //accumulate skipped positions where both are zero
                    r += aMean * bMean * (bCur.getIndex()-lastObservedIndex - 1);
                    lastObservedIndex = bCur.getIndex();

                    double bVal = bCur.getValue() - bMean;
                    r += -aMean * bVal;
                    bSqrd += bVal * bVal;
                    newB = true;
                }

                r += aMean * bMean * (a.length()-lastObservedIndex - 1);
                aSqrd += aMean * aMean * (a.length()-a.nnz());
                bSqrd += bMean * bMean * (b.length()-b.nnz());
            }
        }
        else//dense!
        {
            for(int i = 0; i < a.length(); i++)
            {
                double aTmp = a.get(i);
                double bTmp = b.get(i);
                if(bothNonZero && (aTmp == 0 || bTmp == 0))
                    continue;
                double aVal = aTmp-aMean;
                double bVal = bTmp-bMean;
                r += aVal*bVal;
                aSqrd += aVal*aVal;
                bSqrd += bVal*bVal;
            }
        }
        
        if(bSqrd == 0 && aSqrd == 0)
            return 0;
        else if(bSqrd == 0 || aSqrd == 0)
            return r/Math.sqrt((aSqrd+1e-10)*(bSqrd+1e-10));

        return r/Math.sqrt(aSqrd*bSqrd);
    }
    
    /*
     * TODO Accerlation for Pearson can be done, its a little complicated (you 
     * cache the means and Sqrd values - so that you can do just 1 pass over all
     * values). But thats a good bit of code, and the above needs to be cleaned
     * up before implementing that. 
     */

    @Override
    public boolean supportsAcceleration()
    {
        return false;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs)
    {
        return null;
    }

    @Override
    public double dist(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        return dist(vecs.get(a), vecs.get(b));
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        return dist(vecs.get(a), b);
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        return null;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs, ExecutorService threadpool)
    {
        return null;
    }
    
    @Override
    public double dist(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        return dist(vecs.get(a), b);
    }

}
