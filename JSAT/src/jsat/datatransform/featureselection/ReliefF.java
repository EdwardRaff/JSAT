
package jsat.datatransform.featureselection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.datatransform.DataTransform;
import jsat.datatransform.DataTransformFactory;
import jsat.datatransform.RemoveAttributeTransform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.utils.FakeExecutor;
import jsat.utils.IndexTable;
import jsat.utils.SystemInfo;

/**
 * Provides an implementation of the ReliefF algorithm for feature importance computing. 
 * Because JSAT does not support neighbor searching for categorical values, it does not 
 * provides weights for categorical variables. <br>
 * Weight values are in the range [-1, 1]. The value is a measure of corelation, so the
 * absolute value of the individual weights would form its relative importance to the 
 * others. <br>
 * The ReliefF algorithm is meant for classification problems, and is computed in a
 * nearest neighbor fashion. <br><br>
 * See:<br>Kononenko, I., Simec, E., & Robnik-Sikonja, M. (1997). 
 * <i><a href="http://www.springerlink.com/index/W174714344273004.pdf">
 * Overcoming the myopia of inductive learning algorithms with RELIEFF</a></i>. 
 * Applied Intelligence, 7, 39–55. 
 * 
 * @author Edward Raff
 */
public class ReliefF extends RemoveAttributeTransform
{
    private double[] w;
    
    /**
     * Creates a new ReliefF object to measure the importance of the variables with 
     * respect to a classification task. Only numeric features will be removed. 
     * Categorical features will be ignored and left in tact by the transformation
     * 
     * @param cds the data set to measure numeric variable importance from
     * @param featureCount the number of features to keep
     * @param m the number of learning iterations to perform
     * @param n the number of neighbors to measure importance from
     * @param dm the distance metric to use
     */
    public ReliefF(final ClassificationDataSet cds, int featureCount, final int m, final int n, final DistanceMetric dm)
    {
        this(cds, featureCount, m, n, dm, new DefaultVectorCollectionFactory<Vec>());
    }
    
    /**
     * Creates a new ReliefF object to measure the importance of the variables with 
     * respect to a classification task. Only numeric features will be removed. 
     * Categorical features will be ignored and left in tact by the transformation
     * 
     * @param cds the data set to measure numeric variable importance from
     * @param featureCount the number of features to keep
     * @param m the number of learning iterations to perform
     * @param n the number of neighbors to measure importance from
     * @param dm the distance metric to use
     * @param threadPool the source of threads to use for the computation
     */
    public ReliefF(final ClassificationDataSet cds, int featureCount, final int m, final int n, final DistanceMetric dm, ExecutorService threadPool)
    {
        this(cds, featureCount, m, n, dm, new DefaultVectorCollectionFactory<Vec>(), threadPool);
    }
    
    /**
     * Creates a new ReliefF object to measure the importance of the variables with 
     * respect to a classification task. Only numeric features will be removed. 
     * Categorical features will be ignored and left in tact by the transformation
     * 
     * @param cds the data set to measure numeric variable importance from
     * @param featureCount the number of features to keep
     * @param m the number of learning iterations to perform
     * @param n the number of neighbors to measure importance from
     * @param dm the distance metric to use
     * @param vcf the factor to create accelerating structures for nearest neighbor
     */
    public ReliefF(final ClassificationDataSet cds, int featureCount, final int m, final int n, final DistanceMetric dm, VectorCollectionFactory<Vec> vcf)
    {
        this(cds, featureCount, m, n, dm, vcf, null);
    }

    /**
     * Creates a new ReliefF object to measure the importance of the variables with 
     * respect to a classification task. Only numeric features will be removed. 
     * Categorical features will be ignored and left in tact by the transformation
     * 
     * @param cds the data set to measure numeric variable importance from
     * @param featureCount the number of features to keep
     * @param m the number of learning iterations to perform
     * @param n the number of neighbors to measure importance from
     * @param dm the distance metric to use
     * @param vcf the factor to create accelerating structures for nearest neighbor
     * @param threadPool the source of threads to use for the computation
     */
    public ReliefF(final ClassificationDataSet cds, int featureCount, final int m, final int n, final DistanceMetric dm, VectorCollectionFactory<Vec> vcf, ExecutorService threadPool)
    {
        super();
        this.w = new double[cds.getNumNumericalVars()];
        final double[] minVals = new double[w.length];
        Arrays.fill(minVals, Double.POSITIVE_INFINITY);
        final double[] normalizer = new double[w.length];
        Arrays.fill(normalizer, Double.NEGATIVE_INFINITY);
        
        final double[] priors = cds.getPriors();
        final List<Vec> allVecs = cds.getDataVectors();
        for(Vec v : allVecs)
            for(int i = 0; i < v.length(); i++)
            {
                minVals[i] = Math.min(minVals[i], v.get(i));
                normalizer[i] = Math.max(normalizer[i], v.get(i));
            }
        for(int i = 0; i < normalizer.length; i++)
            normalizer[i] -= minVals[i];

        final List<VectorCollection< Vec>> classVC = new ArrayList<VectorCollection< Vec>>(priors.length);
        
        
        TrainableDistanceMetric.trainIfNeeded(dm, cds, threadPool);
        int curStart = 0;
        

        for (int i = 0; i < priors.length; i++)
        {
            int classCount = cds.classSampleCount(i);
            if(threadPool == null)
                classVC.add(vcf.getVectorCollection(allVecs.subList(curStart, curStart+classCount), dm));
            else
                classVC.add(vcf.getVectorCollection(allVecs.subList(curStart, curStart+classCount), dm, threadPool));
            curStart += classCount;
        }
        
        final int toUse = threadPool == null ? 1 : SystemInfo.LogicalCores;
        if(threadPool == null)
            threadPool = new FakeExecutor();
        final int blockSize = m/toUse;
        
        
        final CountDownLatch latch = new CountDownLatch(toUse);
        for(int id = 0; id < toUse; id++)
        {
            final int mm;
            if(id < m%toUse)
                mm = blockSize+1;
            else
                mm = blockSize;
            threadPool.submit(new Runnable() 
            {

                @Override
                public void run()
                {
                    double[] wLocal = new double[w.length];
                    Random rand = new Random();
                    for(int iter = 0; iter < mm; iter++)
                    {
                        final int k = rand.nextInt(cds.getSampleSize());
                        final Vec x_k = allVecs.get(k);
                        final int y_k = cds.getDataPointCategory(k);

                        for (int y = 0; y < priors.length; y++)//# classes = C
                        {
                            int searchFor = y == y_k ? n + 1 : n;//+1 so we dont search for ourselves
                            List<? extends VecPaired<Vec, Double>> nNearestC = classVC.get(y).search(x_k, searchFor);
                            if (searchFor != n)
                                nNearestC = nNearestC.subList(1, searchFor);//chop off the first value which is ourselves
                            for (int i = 0; i < w.length; i++)
                                for (VecPaired<Vec, Double> x_jy : nNearestC)// j loop
                                {
                                    if (y == y_k)
                                        wLocal[i] -= diff(i, x_k, x_jy.getVector(), normalizer)/(m*n);
                                    else
                                        wLocal[i] += priors[y]/(1-priors[y_k])*diff(i, x_k, x_jy.getVector(), normalizer)/(m*n);
                                }
                        }
                    }
                    
                    synchronized(w)
                    {
                        for(int i = 0; i < w.length; i++)
                            w[i] += wLocal[i];
                    }
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
            Logger.getLogger(ReliefF.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        IndexTable it = new IndexTable(w);
        
        Set<Integer> numericalToRemove = new HashSet<Integer>(w.length*2);
        
        for(int i = 0; i < w.length-featureCount; i++)
            numericalToRemove.add(it.index(i));
        setUp(cds, Collections.EMPTY_SET, numericalToRemove);
    }
    
    /**
     * Returns accesses to the learned weight data. Altering the values will be 
     * reflected in this original ReliefF object. 
     * @return access to the raw weight values
     */
    public Vec getWeights()
    {
        return new DenseVector(w);
    }

    private double diff(int i, Vec xj, Vec xk, double[] normalzer)
    {
        if(normalzer[i] == 0)
            return 0;
        return Math.abs(xj.get(i) - xk.get(i))/normalzer[i];
    }
    
    public static class ReliefFFactory implements DataTransformFactory
    {
        private int featureCount;
        private int iterations;
        private int neighbors;
        private DistanceMetric dm;

        /**
         * Creates a new ReliefF object to measure the importance of the variables 
         * with respect to a classification task. Only numeric features will be 
         * removed. Categorical features will be ignored
         * 
         * and left in tact by the transformation
         *
         * @param featureCount the number of features to keep
         * @param iterations the number of learning iterations to perform
         * @param neighbors the number of neighbors to measure importance from
         * @param dm the distance metric to use
         */
        public ReliefFFactory(int featureCount, int iterations, int neighbors, DistanceMetric dm)
        {
            this.featureCount = featureCount;
            this.iterations = iterations;
            this.neighbors = neighbors;
            this.dm = dm;
        }

        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            if(!(dataset instanceof ClassificationDataSet))
                throw new FailedToFitException("ReliefF transforms can only be learned from classification data sets");
            return new ReliefF((ClassificationDataSet)dataset, featureCount, iterations, neighbors, dm);
        }
    }
}
