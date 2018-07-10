
package jsat.clustering.kmeans;

import jsat.DataSet;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.utils.concurrent.ParallelUtils;

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
    public int[] cluster(DataSet dataSet, final int K, boolean parallel, int[] designations)
    {
        if(K < 2)
            throw new FailedToFitException("Clustering requires at least 2 clusters");
        
        final int N = dataSet.size();
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
            ParallelUtils.run(parallel, N, (start, end)->
            {
                for (int i = start; i < end; i++)
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
            });
            
            //now we have all the new assignments, we can compute the changes
            changed = ParallelUtils.run(parallel, N, (start, end) -> 
            {
                double[] sqrdChange = new double[K];
                double[] ownerChange = new double[K];

                int localChagne = 0;
                for (int i = start; i < end; i++)
                    localChagne += updateMeansFromChange(i, assignments, sqrdChange, ownerChange);

                synchronized(assignments)
                {
                    applyMeanUpdates(sqrdChange, ownerChange);
                }
                return localChagne;
            }, 
            (t, u) -> t+u);

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
        
        final int N = dataSet.size();
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
    public LloydKernelKMeans clone()
    {
        return new LloydKernelKMeans(this);
    }


}
