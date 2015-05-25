package jsat.linear.distancemetrics;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.datatransform.UnitVarianceTransform;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import jsat.linear.VecOps;
import jsat.math.FunctionBase;
import jsat.math.MathTricks;
import jsat.regression.RegressionDataSet;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.ParallelUtils;

/**
 * Implementation of the Normalized Euclidean Distance Metric. The normalized 
 * version divides each variable by its standard deviation, and then continues 
 * as the normal {@link EuclideanDistance}. <br>
 * The same results can be achieved by first applying 
 * {@link UnitVarianceTransform} to a data set before using the 
 * L2 norm.<br> 
 * It is equivalent to the {@link MahalanobisDistance} if only the diagonal 
 * values were used. 
 * 
 * 
 * @author Edward Raff
 */
public class NormalizedEuclideanDistance extends TrainableDistanceMetric
{

	private static final long serialVersionUID = 210109457671623688L;
	private Vec invStndDevs;

    /**
     * Creates a new Normalized Euclidean distance metric
     */
    public NormalizedEuclideanDistance()
    {
    }

    @Override
    public <V extends Vec> void train(List<V> dataSet)
    {
        invStndDevs = MatrixStatistics.covarianceDiag(MatrixStatistics.meanVector(dataSet), dataSet);
        invStndDevs.applyFunction(MathTricks.sqrdFunc);
        invStndDevs.applyFunction(MathTricks.invsFunc);
    }

    @Override
    public <V extends Vec> void train(List<V> dataSet, ExecutorService threadpool)
    {
        train(dataSet);
    }

    @Override
    public void train(DataSet dataSet)
    {
        invStndDevs = dataSet.getColumnMeanVariance()[1];
        invStndDevs.applyFunction(MathTricks.sqrdFunc);
        invStndDevs.applyFunction(MathTricks.invsFunc);
    }

    @Override
    public void train(DataSet dataSet, ExecutorService threadpool)
    {
        train(dataSet);
    }

    @Override
    public void train(ClassificationDataSet dataSet)
    {
        train((DataSet)dataSet);
    }

    @Override
    public void train(ClassificationDataSet dataSet, ExecutorService threadpool)
    {
        train(dataSet);
    }

    @Override
    public boolean supportsClassificationTraining()
    {
        return true;
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train((DataSet)dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadpool)
    {
        train(dataSet);
    }

    @Override
    public boolean supportsRegressionTraining()
    {
        return true;
    }

    @Override
    public boolean needsTraining()
    {
        return invStndDevs == null;
    }

    @Override
    public NormalizedEuclideanDistance clone()
    {
        NormalizedEuclideanDistance clone = new NormalizedEuclideanDistance();
        if(this.invStndDevs != null)
            clone.invStndDevs = this.invStndDevs.clone();
        return clone;
    }

    @Override
    public double dist(Vec a, Vec b)
    {
        double r = VecOps.accumulateSum(invStndDevs, a, b, new FunctionBase() 
        {
            private static final long serialVersionUID = 3190953661114076430L;

            @Override
            public double f(Vec x)
            {
                return Math.pow(x.get(0), 2);
            }
        });

        return Math.sqrt(r);
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

    /*
     * TODO when moving to java8, convert TrainableDistanceMetric into an 
     * interface, fix this class up. Then extend WeightedEuclideanDistance 
     */
    
    @Override
    public boolean supportsAcceleration()
    {
        return true;
    }

    @Override
    public List<Double> getAccelerationCache(List<? extends Vec> vecs)
    {
        DoubleList cache = new DoubleList(vecs.size());
        
        for(Vec v : vecs)
            cache.add(VecOps.weightedDot(invStndDevs, v, v));
        
        return cache;
    }
    
    @Override
    public List<Double> getAccelerationCache(final List<? extends Vec> vecs, ExecutorService threadpool)
    {
        if(threadpool == null || threadpool instanceof FakeExecutor)
            return getAccelerationCache(vecs);
        final double[] cache = new double[vecs.size()];
        
        final int P = Math.min(SystemInfo.LogicalCores, vecs.size());
        final CountDownLatch latch = new CountDownLatch(P);

        for(int ID = 0; ID < P; ID++)
        {
            final int start = ParallelUtils.getStartBlock(cache.length, ID, P);
            final int end = ParallelUtils.getEndBlock(cache.length, ID, P);
            threadpool.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    for(int i = start; i < end; i++)
                        cache[i] = VecOps.weightedDot(invStndDevs, vecs.get(i), vecs.get(i));
                    latch.countDown();
                }
            });
        }
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(NormalizedEuclideanDistance.class.getName()).log(Level.SEVERE, null, ex);
        }

        return DoubleList.view(cache, cache.length);
    }

    @Override
    public double dist(int a, int b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), vecs.get(b));
        
        return Math.sqrt(cache.get(a)+cache.get(b)-2*VecOps.weightedDot(invStndDevs, vecs.get(a), vecs.get(b)));
    }

    @Override
    public double dist(int a, Vec b, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        return Math.sqrt(cache.get(a)+VecOps.weightedDot(invStndDevs, b, b)-2*VecOps.weightedDot(invStndDevs, vecs.get(a), b));
    }

    @Override
    public List<Double> getQueryInfo(Vec q)
    {
        DoubleList qi = new DoubleList(1);
        qi.add(VecOps.weightedDot(invStndDevs, q, q));
        return qi;
    }

    @Override
    public double dist(int a, Vec b, List<Double> qi, List<? extends Vec> vecs, List<Double> cache)
    {
        if(cache == null)
            return dist(vecs.get(a), b);
        
        return Math.sqrt(cache.get(a)+qi.get(0)-2*VecOps.weightedDot(invStndDevs, vecs.get(a), b));
    }
}
