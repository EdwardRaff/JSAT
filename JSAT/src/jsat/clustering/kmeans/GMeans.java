
package jsat.clustering.kmeans;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.Normal;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.clustering.SeedSelectionMethods;
import jsat.linear.*;
import static java.lang.Math.*;

/**
 * This class provides a method of performing {@link KMeans} clustering when the
 * value of {@code K} is not known. It works by recursively splitting means 
 * up to some specified maximum.
 * value. <br>
 * <br>
 * When the value of {@code K} is specified, the implementation will simply call
 * the regular KMeans object it was constructed with. <br>
 * <br>
 * See: Hamerly, G.,&amp;Elkan, C. (2003). <i>Learning the K in K-Means</i>. In 
 * seventeenth annual conference on neural information processing systems (NIPS)
 * (pp. 281–288). Retrieved from 
 * <a href="http://papers.nips.cc/paper/2526-learning-the-k-in-k-means.pdf">here
 * </a>
 *
 * @author Edward Raff
 */
public class GMeans extends KMeans 
{
    private static final long serialVersionUID = 7306976407786792661L;
    private boolean trustH0 = true;
    private boolean iterativeRefine = true;
    
    private int minClusterSize = 25;
    private KMeans kmeans;

    public GMeans()
    {
        this(new HamerlyKMeans());
    }
    
    public GMeans(KMeans kmeans)
    {
        super(kmeans.dm, kmeans.seedSelection, kmeans.rand);
        this.kmeans = kmeans;
        kmeans.setStoreMeans(true);
    }

    public GMeans(GMeans toCopy)
    {
        super(toCopy);
        this.kmeans = toCopy.kmeans.clone();
        this.trustH0 = toCopy.trustH0;
        this.iterativeRefine = toCopy.iterativeRefine;
        this.minClusterSize = toCopy.minClusterSize;
    }

    /**
     * Each new cluster will be tested for normality, with the null hypothesis 
     * H0 being that the cluster is normal. If this is set to {@code true} then 
     * an optimization is done that once a center fails to reject the null 
     * hypothesis, it will never be tested again. This is a safe assumption when
     * {@link #setIterativeRefine(boolean) } is set to {@code false}, but 
     * otherwise may not quite be true. <br>
     * <br>
     * When {@code trustH0} is {@code true} (the default option), G-Means will 
     * make at most O(k) runs of k-means for the final value of k chosen. When 
     * {@code false}, at most O(k<sup>2</sup>) runs of k-means will occur. 
     * 
     * @param trustH0 {@code true} if a centroid shouldn't be re-tested once it 
     * fails to split. 
     */
    public void setTrustH0(boolean trustH0)
    {
        this.trustH0 = trustH0;
    }

    /**
     * 
     * @return {@code true} if cluster that fail to split wont be re-tested. 
     * {@code false} if they will. 
     */
    public boolean getTrustH0()
    {
        return trustH0;
    }

    /**
     * Sets the minimum size for splitting a cluster. 
     * @param minClusterSize the minimum number of data points that must be present in a 
     * cluster to consider splitting it
     */
    public void setMinClusterSize(int minClusterSize)
    {
        if(minClusterSize < 2)
            throw new IllegalArgumentException("min cluster size that could be split is 2, not " + minClusterSize);
        this.minClusterSize = minClusterSize;
    }

    /**
     * 
     * @return the minimum number of data points that must be present in a 
     * cluster to consider splitting it
     */
    public int getMinClusterSize()
    {
        return minClusterSize;
    }

    /**
     * Sets whether or not the set of all cluster centers should be refined at 
     * every iteration. By default this is {@code true} and part of how the 
     * GMeans algorithm is described. Setting this to {@code false} can result
     * in large speedups at the potential cost of quality. 
     * @param refineCenters {@code true} to refine the cluster centers at every 
     * step, {@code false} to skip this step of the algorithm. 
     */
    public void setIterativeRefine(boolean refineCenters)
    {
        this.iterativeRefine = refineCenters;
    }

    /**
     * 
     * @return {@code true} if the cluster centers are refined at every 
     * step, {@code false} if skipping this step of the algorithm. 
     */
    public boolean getIterativeRefine()
    {
        return iterativeRefine;
    }
    

    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, 1, Math.max(dataSet.getSampleSize()/20, 10), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, 1, Math.max(dataSet.getSampleSize()/20, 10), threadpool, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        final int N = dataSet.getSampleSize();
        //initiate
        if(lowK >= 2)
        {
            designations = kmeans.cluster(dataSet, lowK, threadpool, designations);
            means = new ArrayList<Vec>(kmeans.getMeans());
        }
        else//1 mean of all the data
        {
            if(designations == null || designations.length < N)
                designations = new int[N];
            else
                Arrays.fill(designations, 0);
            means = new ArrayList<Vec>(Arrays.asList(MatrixStatistics.meanVector(dataSet)));
        }
        
        
        int[] subS = new int[designations.length];
        int[] subC = new int[designations.length];
        
        Vec v = new DenseVector(dataSet.getNumNumericalVars());
        double[] xp = new double[N];
        //tract if we should stop testing a mean or not
        List<Boolean> dontRedo = new ArrayList<Boolean>(Collections.nCopies(means.size(), false));
        
        //pre-compute acceleration cache instead of re-computing every refine call
        List<Double> accelCache = dm.getAccelerationCache(dataSet.getDataVectors(), threadpool);
        
        double thresh = 1.8692;//TODO make this configurable
        int origMeans;
        do
        {
            
            origMeans = means.size();
            for(int c = 0; c < origMeans; c++)
            {
                if(dontRedo.get(c))
                    continue;
                //2. Initialize two centers, called “children” of c. 
                //for now lets just let k-means decide
                List<DataPoint> X = getDatapointsFromCluster(c, designations, dataSet, subS);
                final int n = X.size();//NOTE, not the same as N. PAY ATENTION
                
                if(X.size() < minClusterSize || means.size() == highK)
                    continue;//this loop with force it to exit when we hit max K
                SimpleDataSet subSet = new SimpleDataSet(X);
                //3. Run k-means on these two centers in X. Let c1, c2 be the child centers chosen by k-means
                subC = kmeans.cluster(subSet, 2, threadpool, subC);
                List<Vec> subMean = kmeans.getMeans();
                Vec c1 = subMean.get(0);
                Vec c2 = subMean.get(1);
                

                /* 4. 
                 * Let v = c1 − c2 be a d-dimensional vector that connects the two 
                 * centers. This is the direction that k-means believes to be 
                 * important for clustering. Then project X onto v: 
                 * x'_i = <x_i, v>/||v||^2. X' is a 1-dimensional representation of
                 * the data projected onto v. Transform X' so that it has mean 0 and
                 * variance 1.
                 */
                c1.copyTo(v);
                v.mutableSubtract(c2);
                double vNrmSqrd = Math.pow(v.pNorm(2), 2);
                if(Double.isNaN(vNrmSqrd) || vNrmSqrd < 1e-6)
                    continue;//can happen when cluster is all the same item (or nearly so)
                for(int i = 0; i < X.size(); i++)
                    xp[i] = X.get(i).getNumericalValues().dot(v)/vNrmSqrd;
                //we need this in sorted order later, so lets just sort them now
                Arrays.sort(xp, 0, X.size());
                DenseVector Xp = new DenseVector(xp, 0, X.size());
                
                Xp.mutableSubtract(Xp.mean());
                Xp.mutableDivide(Math.max(Xp.standardDeviation(), 1e-6));
                

                //5. 
                for(int i = 0; i < Xp.length(); i++)
                    Xp.set(i, Normal.cdf(Xp.get(i), 0, 1));
                double A = 0;
                for(int i = 1; i <= Xp.length(); i++)
                {
                    double phi = Xp.get(i-1);
                    A += (2*i-1)*log(phi) + (2*(n-i)+1)*log(1-phi);
                }
                
                A/=-n;
                A += -n;
                //eq(2)
                A *= 1 + 4.0/n - 25.0/(n*n);

                if(A <= thresh)
                {
                    if(trustH0)//if we are going to trust that H0 is true forever, mark it
                        dontRedo.set(c, true);
                    continue;//passed the test, do not split
                }
                //else, accept the split

                //first, update assignment array. Cluster '0' stays as is, re-set cluster '1'
                for(int i = 0; i < X.size(); i++)
                    if(subC[i] == 1)
                        designations[subS[i]] = means.size();
                //replace current mean and add new one
                means.set(c, c1.clone());//cur index in dontRedo stays false
                means.add(c2.clone());//add a 'false' for new center
                dontRedo.add(false);
            }
            //"Between each round of splitting, we run k-means on the entire dataset and all the centers to refine the current solution"
            if(iterativeRefine && means.size() > 1)
                kmeans.cluster(dataSet, accelCache, means.size(), means, designations, false, threadpool, false, null);
        }
        while (origMeans < means.size());
        
        if(!iterativeRefine && means.size() > 1)//if we havn't been refining we need to do so now!
            kmeans.cluster(dataSet, accelCache, means.size(), means, designations, false, threadpool, false, null);
        return designations;
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        return cluster(dataSet, lowK, highK, null, designations);
    }

    @Override
    public int getIterationLimit()
    {
        return kmeans.getIterationLimit();
    }

    @Override
    public void setIterationLimit(int iterLimit)
    {
        kmeans.setIterationLimit(iterLimit);
    }

    @Override
    public void setSeedSelection(SeedSelectionMethods.SeedSelection seedSelection)
    {
    	//XXX when called from constructor in superclass seed is ignored
        if(kmeans != null)//needed when initing
            kmeans.setSeedSelection(seedSelection);
    }

    @Override
    public SeedSelectionMethods.SeedSelection getSeedSelection()
    {
        return kmeans.getSeedSelection();
    }
    

    @Override
    protected double cluster(DataSet dataSet, List<Double> accelCache, int k, List<Vec> means, int[] assignment, boolean exactTotal, ExecutorService threadpool, boolean returnError, Vec dataPointWeights)
    {
        return kmeans.cluster(dataSet, accelCache, k, means, assignment, exactTotal, threadpool, returnError, null);
    }

    @Override
    public GMeans clone()
    {
        return new GMeans(this);
    }
}
