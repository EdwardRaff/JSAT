package jsat.clustering.kmeans;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.linear.Vec;

/**
 * This class provides a method of performing {@link KMeans} clustering when the
 * value of {@code K} is not known. It works by incrementing the value 
 * of {@code k} up to some specified maximum, and running a full KMeans for each
 * value. <br>
 * <br>
 * Note, by default this implementation uses a heuristic for the max value of 
 * {@code K} that is capped at 100 when using the 
 * {@link #cluster(jsat.DataSet) } type methods. <br>
 * <br>
 * When the value of {@code K} is specified, the implementation will simply call
 * the regular KMeans object it was constructed with. 
 * 
 * See: Pham, D. T., Dimov, S. S.,&amp;Nguyen, C. D. (2005). <i>Selection of K in 
 * K-means clustering</i>. Proceedings of the Institution of Mechanical 
 * Engineers, Part C: Journal of Mechanical Engineering Science, 219(1), 
 * 103–119. doi:10.1243/095440605X8298
 * 
 * @author Edward Raff
 */
public class KMeansPDN extends KMeans
{
    private static final long serialVersionUID = -2358377567814606959L;
    private KMeans kmeans;
    private double[] fKs;
    
    /**
     * Creates a new clusterer. 
     */
    public KMeansPDN()
    {
        this(new HamerlyKMeans());
    }

    /**
     * Creates a new clustered that uses the specified object to perform clustering for all {@code k}. 
     * @param kmeans the k-means object to use for clustering
     */
    public KMeansPDN(KMeans kmeans)
    {
        super(kmeans.dm, kmeans.seedSelection, kmeans.rand);
        this.kmeans = kmeans;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public KMeansPDN(KMeansPDN toCopy)
    {
        super(toCopy);
        this.kmeans = toCopy.kmeans.clone();
        if(toCopy.fKs != null)
            this.fKs = Arrays.copyOf(toCopy.fKs, toCopy.fKs.length);
    }

    /**
     * Returns the array of {@code f(K)} values generated for the last data set.
     * The value at index {@code i} is the score for cluster {@code i+1}. 
     * Smaller values indicate better clusterings. 
     * 
     * @return the array of {@code f(K)} values, or {@code null} if no data set 
     * has been clustered
     */
    public double[] getfKs()
    {
        return fKs;
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, null, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, 1, (int) Math.min(Math.max(Math.sqrt(dataSet.getSampleSize()), 10), 100), threadpool, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        if(highK == lowK)
            return cluster(dataSet, lowK, threadpool, designations);
        else if(highK < lowK)
            throw new IllegalArgumentException("low value of k (" + lowK + ") must be higher than the high value of k(" + highK + ")");
        final int N = dataSet.getSampleSize();
        final int D = dataSet.getNumNumericalVars();
        fKs = new double[highK-1];//we HAVE to start from k=2
        fKs[0] = 1.0;//see eq(2)
        
        int[] bestCluster = new int[N];
        double minFk = lowK == 1 ? 1.0 : Double.POSITIVE_INFINITY;//If our low k is > 1, force the check later to kick in at the first candidate k by making fK appear Inf
        
        if(designations == null || designations.length < N)
            designations = new int[N];
        
        
        double alphaKprev = 0, S_k_prev = 0;
        
        //re used every iteration
        List<Vec> curMeans = new ArrayList<Vec>(highK);
        means = new ArrayList<Vec>();//the best set of means
        //pre-compute cache instead of re-computing every time
        List<Double> accelCache = dm.getAccelerationCache(dataSet.getDataVectors(), threadpool);
        
        for(int k = 2; k < highK; k++)
        {
            curMeans.clear();
            //kmeans objective function result is the same as S_k
            double S_k = cluster(dataSet, accelCache, k, curMeans, designations, true, threadpool, true, null);//TODO could add a flag to make approximate S_k an option. Though it dosn't seem to work great on toy problems, might be fine on more realistic data


            double alpha_k;
            if(k == 2)
                alpha_k = 1 - 3.0/(4*D); //eq(3a)
            else 
                alpha_k = alphaKprev + (1-alphaKprev)/6;//eq(3b)
            
            double fK;//eq(2)
            if(S_k_prev == 0)
                fKs[k-1] = fK = 1;
            else
                fKs[k-1] = fK = S_k/(alpha_k*S_k_prev);
            
            alphaKprev = alpha_k;
            S_k_prev = S_k;
            
            if(k >= lowK && minFk > fK)
            {
                System.arraycopy(designations, 0, bestCluster, 0, N);
                minFk = fK;
                means.clear();
                for(Vec mean : curMeans)
                    means.add(mean.clone());
            }
        }
        
        //contract is we return designations with the data in it if we can, so copy the values back
        System.arraycopy(bestCluster, 0, designations, 0, N);
        return designations;
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        return cluster(dataSet, lowK, highK, null, designations);
    }

    @Override
    protected double cluster(DataSet dataSet, List<Double> accelCache, int k, List<Vec> means, int[] assignment, boolean exactTotal, ExecutorService threadpool, boolean returnError, Vec dataPointWeights)
    {
        return kmeans.cluster(dataSet, accelCache, k, means, assignment, exactTotal, threadpool, returnError, null);
    }

    @Override
    public KMeansPDN clone()
    {
        return new KMeansPDN(this);
    }
    
}
