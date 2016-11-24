
package jsat.clustering.kmeans;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.utils.SystemInfo;

/**
 * An implementation of the naive algorithm for performing kernel k-means. 
 * 
 * @author Edward Raff
 */
public class LloydKernelKMeans extends KernelKMeans
{

    private static final long serialVersionUID = 1280985811243830450L;

    /**
     * Creates a new Kernel K Means object
     * @param kernel the kernel to use
     */
    public LloydKernelKMeans(KernelTrick kernel)
    {
        super(kernel);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public LloydKernelKMeans(LloydKernelKMeans toCopy)
    {
        super(toCopy);
    }

    @Override
    public int[] cluster(DataSet dataSet, final int K, ExecutorService threadpool, int[] designations)
    {
        if(K < 2)
            throw new FailedToFitException("Clustering requires at least 2 clusters");
        
        final int N = dataSet.getSampleSize();
        if(designations == null)
            designations = new int[N];
        
        X = dataSet.getDataVectors();
        
        setup(K, designations, dataSet.getDataWeights());
        final int[] assignments = designations;

        int changed;
        int iter = 0;
        do
        {
            changed = 0;
            //find new closest center
            final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
            for(int id = 0; id < SystemInfo.LogicalCores; id++)
            {
                final int ID = id;
                threadpool.submit(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        for (int i = ID; i < N; i+=SystemInfo.LogicalCores)
                        {
                            double minDist = Double.POSITIVE_INFINITY;
                            int min_indx = 0;
                            for (int k = 0; k < K; k++)
                            {
                                double dist_k = distance(i, k, assignments);
                                if (dist_k < minDist)
                                {
                                    minDist = dist_k;
                                    min_indx = k;
                                }
                            }
                            
                            newDesignations[i] = min_indx;
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
                Logger.getLogger(LloydKernelKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
            //now we have all the new assignments, we can compute the changes
            List<Future<Integer>> futureChanges = new ArrayList<Future<Integer>>(SystemInfo.LogicalCores);
            for(int id = 0; id < SystemInfo.LogicalCores; id++)
            {
                final int ID = id;
                futureChanges.add(threadpool.submit(new Callable<Integer>()
                {

                    @Override
                    public Integer call() throws Exception
                    {
                        double[] sqrdChange = new double[K];
                        double[] ownerChange = new double[K];
                        
                        int localChagne = 0;
                        for (int i = ID; i < N; i+=SystemInfo.LogicalCores)
                            localChagne += updateMeansFromChange(i, assignments, sqrdChange, ownerChange);
                        
                        synchronized(assignments)
                        {
                            applyMeanUpdates(sqrdChange, ownerChange);
                        }
                        return localChagne;
                    }
                }));
            }
            
            
            try
            {
                for (Future<Integer> f : futureChanges)
                    changed += f.get();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(LloydKernelKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }
            catch (ExecutionException ex)
            {
                Logger.getLogger(LloydKernelKMeans.class.getName()).log(Level.SEVERE, null, ex);
            }

            //update constatns 
            updateNormConsts();
            //update designations
            System.arraycopy(newDesignations, 0, designations, 0, N);
        }
        while (changed > 0 && ++iter < maximumIterations);

        return designations;
    }

    @Override
    public int[] cluster(DataSet dataSet, int K, int[] designations)
    {
        if(K < 2)
            throw new FailedToFitException("Clustering requires at least 2 clusters");
        
        final int N = dataSet.getSampleSize();
        if(designations == null)
            designations = new int[N];
        
        X = dataSet.getDataVectors();
        
        setup(K, designations, dataSet.getDataWeights());
        

        int changed;
        int iter = 0;
        do
        {
            changed = 0;
            for (int i = 0; i < N; i++)
            {
                double minDist = Double.POSITIVE_INFINITY;
                int min_indx = 0;
                for (int k = 0; k < K; k++)
                {
                    double dist_k = distance(i, k, designations);
                    if (dist_k < minDist)
                    {
                        minDist = dist_k;
                        min_indx = k;
                    }
                }

                newDesignations[i] = min_indx;
            }

            
            for(int i = 0; i < N; i++)
                changed += updateMeansFromChange(i, designations);

            //update constatns 
            updateNormConsts();
            //update designations
            System.arraycopy(newDesignations, 0, designations, 0, N);
            
        }
        while (changed > 0 && ++iter < maximumIterations);

        return designations;
    }

    @Override
    public KernelKMeans clone()
    {
        return new LloydKernelKMeans(this);
    }


}
