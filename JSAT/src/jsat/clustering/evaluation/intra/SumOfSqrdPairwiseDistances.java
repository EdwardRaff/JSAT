package jsat.clustering.evaluation.intra;

import java.util.List;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * Evaluates a cluster's validity by computing the normalized sum of pairwise
 * distances for all points in the cluster. <br>
 * Note, the normalization value for each cluster is <i>1/(2 * n)</i>, where
 * <i>n</i> is the number of points in each cluster. <br>
 * <br>
 * For general distance metrics, this requires O(n<sup>2</sup>) work. The
 * {@link EuclideanDistance} is a special case, and takes only O(n) work.
 *
 * @author Edward Raff
 */
public class SumOfSqrdPairwiseDistances implements IntraClusterEvaluation
{
    private DistanceMetric dm;

    /**
     * Creates a new evaluator that uses the Euclidean distance
     */
    public SumOfSqrdPairwiseDistances()
    {
        this(new EuclideanDistance());
    }

    /**
     * Creates a new cluster evaluator using the given distance metric
     *
     * @param dm the distance metric to use
     */
    public SumOfSqrdPairwiseDistances(final DistanceMetric dm)
    {
        this.dm = dm;
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public SumOfSqrdPairwiseDistances(final SumOfSqrdPairwiseDistances toCopy)
    {
        this(toCopy.dm.clone());
    }

    /**
     * Sets the distance metric to be used whenever this object is called to 
     * evaluate a cluster
     * @param dm the distance metric to use
     */
    public void setDistanceMetric(final DistanceMetric dm)
    {
        this.dm = dm;
    }

    /**
     * 
     * @return the distance metric being used for evaluation
     */
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }
    
    @Override
    public double evaluate(final int[] designations, final DataSet dataSet, final int clusterID)
    {
        int N = 0;
        double sum = 0;
        final List<Vec> X = dataSet.getDataVectors();
        final List<Double> cache = dm.getAccelerationCache(X);

        if (dm instanceof EuclideanDistance)//special case, can compute in O(N) isntead
        {
            final Vec mean = new DenseVector(X.get(0).length());
            for (int i = 0; i < dataSet.getSampleSize(); i++)
            {
                if (designations[i] != clusterID) {
                  continue;
                }
                mean.mutableAdd(X.get(i));
                N++;
            }
            mean.mutableDivide((N + 1e-10));//1e-10 incase N=0

            final List<Double> qi = dm.getQueryInfo(mean);
            for (int i = 0; i < dataSet.getSampleSize(); i++)
            {
                if (designations[i] == clusterID) {
                  sum += Math.pow(dm.dist(i, mean, qi, X, cache), 2);
                }
            }

            return sum;
        }
        //regulare case, O(N^2)

        for (int i = 0; i < dataSet.getSampleSize(); i++)
        {
            if (designations[i] != clusterID) {
              continue;
            }
            N++;

            for (int j = i + 1; j < dataSet.getSampleSize(); j++)
            {
                if (designations[j] == clusterID) {
                  sum += 2*Math.pow(dm.dist(i, j, X, cache), 2);
                }
            }
        }

        return sum / (N * 2);
    }

    @Override
    public double evaluate(final List<DataPoint> dataPoints)
    {
        return evaluate(new int[dataPoints.size()], new SimpleDataSet(dataPoints), 0);
    }

    @Override
    public SumOfSqrdPairwiseDistances clone() 
    {
        return new SumOfSqrdPairwiseDistances(this);
    }
}
