
package jsat.clustering.kmeans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.clustering.KClustererBase;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.ConstantVector;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;

/**
 * Base class for various Kernel K Means implementations. Because the Kernelized 
 * version is more computationally expensive, only the clustering methods where 
 * the number of clusters is specified apriori are supported. <br>
 * <br>
 * KernelKMeans keeps a reference to the data passed in for clustering so that 
 * queries can be conveniently answered, such as getting 
 * {@link #findClosestCluster(jsat.linear.Vec) the closest cluster} or finding 
 * the {@link #meanToMeanDistance(int, int) distance between means}
 * 
 * @author Edward Raff
 */
public abstract class KernelKMeans extends KClustererBase implements Parameterized
{

    private static final long serialVersionUID = -5294680202634779440L;

    /**
     * The kernel trick to use
     */
    @ParameterHolder
    protected KernelTrick kernel;
    
    /**
     * The list of data points that this was trained on 
     */
    protected List<Vec> X;
    /**
     * The weight of each data point
     */
    protected Vec W;
    /**
     * THe acceleration cache for the kernel
     */
    protected List<Double> accel;
    /**
     * The value of k(x,x) for every point in {@link #X} 
     */
    protected double[] selfK;
    
    /**
     * The value of the un-normalized squared norm for each mean
     */
    protected double[] meanSqrdNorms;
    
    /**
     * The normalizing constant for each mean. General this would be 
     * 1/owned[k]<sup>2</sup>
     */
    protected double[] normConsts;
    
    /**
     * The weighted number of dataums owned by each mean
     */
    protected double[] ownes;
       
    /**
     * A temporary space for updating ownership designations for each datapoint. 
     * When done, this will store the final designations for each point
     */
    protected int[] newDesignations;
    protected int maximumIterations = Integer.MAX_VALUE;
    
    /**
     * 
     * @param kernel the kernel to use
     */
    public KernelKMeans(KernelTrick kernel)
    {
        this.kernel = kernel;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public KernelKMeans(KernelKMeans toCopy)
    {
        this.kernel = toCopy.kernel.clone();
        this.maximumIterations = toCopy.maximumIterations;
        if(toCopy.X != null)
        {
            this.X = new ArrayList<Vec>(toCopy.X.size());
            for( Vec v : toCopy.X)
                this.X.add(v.clone());
            
        }
        if(toCopy.accel != null)
            this.accel = new DoubleList(toCopy.accel);
        if(toCopy.selfK != null)
            this.selfK = Arrays.copyOf(toCopy.selfK, toCopy.selfK.length);
        
        if(toCopy.meanSqrdNorms != null)
            this.meanSqrdNorms = Arrays.copyOf(toCopy.meanSqrdNorms, toCopy.meanSqrdNorms.length);
        
        if(toCopy.normConsts != null)
            this.normConsts = Arrays.copyOf(toCopy.normConsts, toCopy.normConsts.length);
        
        if(toCopy.ownes != null)
            this.ownes = Arrays.copyOf(toCopy.ownes, toCopy.ownes.length);
        
        if(toCopy.newDesignations != null)
            this.newDesignations = Arrays.copyOf(toCopy.newDesignations, toCopy.newDesignations.length);
        
        if(toCopy.W != null)
            this.W = toCopy.W.clone();
    }
    
    
    
    /**
     * Sets the maximum number of iterations allowed
     * @param iterLimit the maximum number of iterations of the KMeans algorithm 
     */
    public void setMaximumIterations(int iterLimit)
    {
        if(iterLimit <= 0)
            throw new IllegalArgumentException("iterations must be a positive value, not " + iterLimit);
        this.maximumIterations = iterLimit;
    }

    /**
     * Returns the maximum number of iterations of the KMeans algorithm that will be performed. 
     * @return the maximum number of iterations of the KMeans algorithm that will be performed. 
     */
    public int getMaximumIterations()
    {
        return maximumIterations;
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported.");
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported.");
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported.");
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        throw new UnsupportedOperationException("Not supported.");
    }
    
    /**
     * Computes the kernel sum of data point {@code i} against all the points in
     * cluster group {@code clusterID}. 
     * @param i the index of the data point to query for
     * @param clusterID the cluster index to get the sum of kernel products
     * @param d
     * @return the sum <big>&Sigma;</big>k(x<sub>i</sub>, x<sub>j</sub>), &forall; j, d[<i>j</i>] == <i>clusterID</i>
     */
    protected double evalSumK(int i, int clusterID, int[] d)
    {
        double sum = 0;
        for(int j = 0; j < X.size(); j++)
            if(d[j] == clusterID)
                sum += W.get(j) * kernel.eval(i, j, X, accel);
        return sum;
    }
    
    /**
     * Computes the kernel sum of the given data point against all the points in
     * cluster group {@code clusterID}. 
     * @param x the data point to get the kernel sum of
     * @param qi the query information for the given data point generated from the kernel in use. See {@link KernelTrick#getQueryInfo(jsat.linear.Vec) }
     * @param clusterID the cluster index to get the sum of kernel products
     * @param d the array of cluster assignments 
     * @return the sum <big>&Sigma;</big>k(x<sub>i</sub>, x<sub>j</sub>), &forall; j, d[<i>j</i>] == <i>clusterID</i>
     */
    protected double evalSumK(Vec x, List<Double> qi, int clusterID, int[] d)
    {
        double sum = 0;
        for(int j = 0; j < X.size(); j++)
            if(d[j] == clusterID)
                sum += W.get(j) * kernel.eval(j, x, qi, X, accel);
        return sum;
    }
    
    /**
     * Sets up the internal structure for KenrelKMeans. Should be called first before any work is done
     * @param K the number of clusters to find
     * @param designations the initial designations array to fill with values
     * @param W the weight for each individual data point
     */
    protected void setup(int K, int[] designations, Vec W)
    {
        accel = kernel.getAccelerationCache(X);
        
        final int N = X.size();
        selfK = new double[N];
        for(int i = 0; i < selfK.length; i++)
            selfK[i] = kernel.eval(i, i, X, accel);
        ownes = new double[K];
        meanSqrdNorms = new double[K];
        newDesignations = new int[N];
        
        if(W  == null)
            this.W = new ConstantVector(1.0, N);
        else
            this.W = W;
        
        Random rand = RandomUtil.getRandom();
        for (int i = 0; i < N; i++)
        {
            int to = rand.nextInt(K);
            ownes[to] += this.W.get(i);
            newDesignations[i] = designations[i] = to;
        }
        
        normConsts = new double[K];
        updateNormConsts();
        
        
        for (int i = 0; i < N; i++)
        {
            int i_k = designations[i];
            final double w_i = this.W.get(i);
            meanSqrdNorms[i_k] += w_i * selfK[i];
            for (int j = i + 1; j < N; j++)
                if (i_k == designations[j])
                    meanSqrdNorms[i_k] += 2 * w_i * this.W.get(j) * kernel.eval(i, j, X, accel);
        }
    }

    /**
     * Updates the normalizing constants for each mean. Should be called after
     * every change in ownership
     */
    protected void updateNormConsts()
    {
        for(int i = 0; i < normConsts.length; i++)
            normConsts[i] = 1.0/(ownes[i]*ownes[i]);
    }
    
    /**
     * Computes the distance between one data point and a specified mean
     * @param i the data point to get the distance for
     * @param k the mean index to get the distance to 
     * @param designations the array if ownership designations for each cluster to use
     * @return the distance between data point {@link #X x}<sub>i</sub> and mean {@code k}
     */
    protected double distance(int i, int k, int[] designations)
    {
        return Math.sqrt(Math.max(selfK[i] - 2.0/ownes[k] * evalSumK(i, k, designations) + meanSqrdNorms[k]*normConsts[k], 0));
    }
    
    /**
     * Returns the distance between the given data point and the the specified cluster
     * @param x the data point to get the distance for
     * @param k the cluster id to get the distance to
     * @return the distance between the given data point and the specified cluster
     */
    public double distance(Vec x, int k)
    {
        return distance(x, kernel.getQueryInfo(x), k);
    }
    
    /**
     * Returns the distance between the given data point and the the specified cluster
     * @param x the data point to get the distance for
     * @param qi the query information for the given data point generated for the kernel in use. See {@link KernelTrick#getQueryInfo(jsat.linear.Vec) }
     * @param k the cluster id to get the distance to
     * @return the distance between the given data point and the specified cluster
     */
    public double distance(Vec x, List<Double> qi, int k)
    {
        if(k >= meanSqrdNorms.length || k < 0)
            throw new IndexOutOfBoundsException("Only " + meanSqrdNorms.length + " clusters. " + k + " is not a valid index");
        return Math.sqrt(Math.max(kernel.eval(0, 0, Arrays.asList(x), qi) - 2.0/ownes[k] * evalSumK(x, qi, k, newDesignations) + meanSqrdNorms[k]*normConsts[k], 0));
    }
    
    /**
     * Finds the cluster ID that is closest to the given data point
     * @param x the data point to get the closest cluster for
     * @return the index of the closest cluster
     */
    public int findClosestCluster(Vec x)
    {
        return findClosestCluster(x, kernel.getQueryInfo(x));
    }
    
    /**
     * Finds the cluster ID that is closest to the given data point
     * @param x the data point to get the closest cluster for
     * @param qi the query information for the given data point generated for the kernel in use. See {@link KernelTrick#getQueryInfo(jsat.linear.Vec) }
     * @return the index of the closest cluster
     */
    public int findClosestCluster(Vec x, List<Double> qi)
    {
        double min = Double.MAX_VALUE;
        int min_indx = -1;
        for(int i  = 0; i < meanSqrdNorms.length; i++)
        {
            double dist = distance(x, qi, i);
            if(dist < min)
            {
                min = dist;
                min_indx = i;
            }
        }
        
        return min_indx;
    }
    
    
    /**
     * Updates the means based off the change of a specific data point
     * @param i the index of the data point to try and update the means based on its movement
     * @param designations the old assignments for ownership of each data point to one of the means
     * @return {@code 1} if the index changed ownership, {@code 0} if the index did not change ownership
     */
    protected int updateMeansFromChange(int i, int[] designations)
    {
        return updateMeansFromChange(i, designations, meanSqrdNorms, ownes);
    }
    /**
     * Accumulates the updates to the means and ownership into the provided 
     * arrays. This does not update {@link #meanSqrdNorms}, and is meant to 
     * accumulate the change. To apply the changes pass the same arrays to {@link #applyMeanUpdates(double[], int[]) }
     * @param i the index of the data point to try and update the means based on its movement
     * @param designations the old assignments for ownership of each data point to one of the means
     * @param sqrdNorms the array to place the changes to the squared norms in
     * @param ownership the array to place the changes to the ownership counts in
     * @return {@code 1} if the index changed ownership, {@code 0} if the index did not change ownership
     */
    protected int updateMeansFromChange(final int i, final int[] designations, final double[] sqrdNorms, final double[] ownership)
    {
        final int old_d = designations[i];
        final int new_d = newDesignations[i];
        
        if (old_d == new_d)//this one has not changed!
            return 0;
        
        final int N = X.size();
        final double w_i = W.get(i);
        ownership[old_d] -= w_i;
        ownership[new_d] += w_i;

        for (int j = 0; j < N; j++)
        {
            final double w_j = W.get(j);
            final int oldD_j = designations[j];
            final int newD_j = newDesignations[j];
            if (i == j)//diagonal is an easy case
            {
                sqrdNorms[old_d] -= w_i*selfK[i];
                sqrdNorms[new_d] += w_i*selfK[i];
            }
            else
            {
                //handle removing contribution from old mean
                if (old_d == oldD_j)
                {
                        //only do this for items that were apart of the OLD center

                    if (i > j && oldD_j != newD_j)
                    {
                        /*
                         * j,j is also being removed from this center.
                         * To avoid removing the value k_ij twice, the
                         * person with the later index gets to do the update
                         */
                    }
                    else//safe to remove the k_ij contribution
                        sqrdNorms[old_d] -= 2 * w_i * w_j * kernel.eval(i, j, X, accel);
                }
                //handle adding contributiont to new mean
                if (new_d == newD_j)
                {
                        //only do this for items that are apart of the NEW center

                    if (i > j && oldD_j != newD_j)
                    {
                        /*
                         * j,j is also being added to this center.
                         * To avoid adding the value k_ij twice, the
                         * person with the later index gets to do the update
                         */
                    }
                    else
                        sqrdNorms[new_d] += 2 * w_i * w_j * kernel.eval(i, j, X, accel);
                }
            }
        }

        return 1;
    }
    
    protected void applyMeanUpdates(double[] sqrdNorms, double[] ownerships)
    {
        for(int i = 0; i < sqrdNorms.length; i++)
        {
            meanSqrdNorms[i] += sqrdNorms[i];
            ownes[i] += ownerships[i];
        }
    }
    
    /**
     * Computes the distance between two of the means in the clustering
     * @param k0 the index of the first mean
     * @param k1 the index of the second mean
     * @return the distance between the two
     */
    public double meanToMeanDistance(int k0, int k1)
    {
        if(k0 >= meanSqrdNorms.length || k0 < 0)
            throw new IndexOutOfBoundsException("Only " + meanSqrdNorms.length + " clusters. " + k0 + " is not a valid index");
        if(k1 >= meanSqrdNorms.length || k1 < 0)
            throw new IndexOutOfBoundsException("Only " + meanSqrdNorms.length + " clusters. " + k1 + " is not a valid index");
        
        return meanToMeanDistance(k0, k1, newDesignations);
    }
    
    protected double meanToMeanDistance(int k0, int k1, int[] assignments)
    {
        double d = meanSqrdNorms[k0]*normConsts[k0]+meanSqrdNorms[k1]*normConsts[k1]-2*dot(k0, k1, assignments);
        return Math.sqrt(Math.max(0, d));//Avoid rare cases wehre 2*dot might be slightly larger
    }
    
    protected double meanToMeanDistance(int k0, int k1, int[] assignments, ExecutorService ex)
    {
        double d = meanSqrdNorms[k0]*normConsts[k0]+meanSqrdNorms[k1]*normConsts[k1]-2*dot(k0, k1, assignments, ex);
        return Math.sqrt(Math.max(0, d));//Avoid rare cases wehre 2*dot might be slightly larger
    }
    
    /**
     * 
     * @param k0 the index of the first cluster 
     * @param k1 the index of the second cluster
     * @param assignments0 the array of assignments to use for index k0
     * @param assignments1 the array of assignments to use for index k1
     * @param k1SqrdNorm the <i>normalized</i> squared norm for the mean 
     * indicated by {@code k1}. (ie: {@link #meanSqrdNorms} multiplied by {@link #normConsts}
     * @return 
     */
    protected double meanToMeanDistance(int k0, int k1, int[] assignments0, int[] assignments1, double k1SqrdNorm)
    {
        double d = meanSqrdNorms[k0]*normConsts[k0]+k1SqrdNorm-2*dot(k0, k1, assignments0, assignments1);
        return Math.sqrt(Math.max(0, d));//Avoid rare cases wehre 2*dot might be slightly larger
    }
    
    /**
     * 
     * @param k0 the index of the first cluster 
     * @param k1 the index of the second cluster
     * @param assignments0 the array of assignments to use for index k0
     * @param assignments1 the array of assignments to use for index k1
     * @param k1SqrdNorm the <i>normalized</i> squared norm for the mean 
     * indicated by {@code k1}. (ie: {@link #meanSqrdNorms} multiplied by {@link #normConsts}
     * @param ex source of threads for parallel execution
     * @return 
     */
    protected double meanToMeanDistance(int k0, int k1, int[] assignments0, int[] assignments1, double k1SqrdNorm, ExecutorService ex)
    {
        double d = meanSqrdNorms[k0]*normConsts[k0]+k1SqrdNorm-2*dot(k0, k1, assignments0, assignments1, ex);
        return Math.sqrt(Math.max(0, d));//Avoid rare cases wehre 2*dot might be slightly larger
    }
    
    /**
     * dot product between two different clusters from one set of cluster assignments  
     * @param k0 the index of the first cluster 
     * @param k1 the index of the second cluster
     * @param assignment the array of assignments for cluster ownership
     * @return the dot product between the two clusters.
     */
    private double dot(final int k0, final int k1, final int[] assignment)
    {
        return dot(k0, k1, assignment, assignment);
    }
    
    /**
     * dot product between two different clusters from one set of cluster assignments  
     * @param k0 the index of the first cluster 
     * @param k1 the index of the second cluster
     * @param assignment the array of assignments for cluster ownership
     * @param ex source of threads for parallel execution
     * @return the dot product between the two clusters.
     */
    private double dot(final int k0, final int k1, final int[] assignment, ExecutorService ex)
    {
        return dot(k0, k1, assignment, assignment, ex);
    }
    
    /**
     * dot product between two different clusters from different sets of cluster
     * assignments. Two different assignment arrays are used to allow
     * overlapping assignment of points to the clusters.
     *
     * @param k0 the first cluster to take the dot product with
     * @param k1 the second cluster to take the dot product with.
     * @param assignment0 vector containing assignment values, will be used to
     * determine which points belong to k0
     * @param assignment1 vector containing assignment values, will be used to
     * determine which points belong to k1
     * @return the dot product between the two clusters.
     */
    private double dot(final int k0, final int k1, final int[] assignment0, final int[] assignment1)
    {
        double dot = 0;
        final int N = X.size();
        double a = 0, b = 0;
        /*
         * Below, unless i&amp;j are somehow in the same cluster - nothing bad will happen 
         */
        for(int i = 0; i < N; i++)
        {
            final double w_i = W.get(i);
            if(assignment0[i] != k0)
                continue;
            a += w_i;
            for(int j = 0; j < N; j++)
            {
                if(assignment1[j] != k1)
                    continue;
                final double w_j = W.get(j);
                dot += w_i * w_j * kernel.eval(i, j, X, accel);
            }
        }
        for(int j = 0; j < N; j++)
            if(assignment1[j] == k1)
                b += W.get(j);
        return dot/(a*b);
    }
    
    /**
     * dot product between two different clusters from different sets of cluster
     * assignments. Two different assignment arrays are used to allow
     * overlapping assignment of points to the clusters.
     *
     * @param k0 the first cluster to take the dot product with
     * @param k1 the second cluster to take the dot product with.
     * @param assignment0 vector containing assignment values, will be used to
     * determine which points belong to k0
     * @param assignment1 vector containing assignment values, will be used to
     * determine which points belong to k1
     * @param ex source of threads for parallel execution
     * @return the dot product between the two clusters.
     */
    private double dot(final int k0, final int k1, final int[] assignment0, final int[] assignment1, ExecutorService ex)
    {
        double dot = 0;
        final int N = X.size();
        double a = 0, b = 0;
        
        List<Future<Double>> dotPartials = new ArrayList<Future<Double>>();
        /*
         * Below, unless i&amp;j are somehow in the same cluster - nothing bad will happen 
         */
        for(int i = 0; i < N; i++)
        {
            final double w_i = W.get(i);
            if(assignment0[i] != k0)
                continue;
            a += w_i;
            final int I = i;
            dotPartials.add(ex.submit(new Callable<Double>()
            {
                @Override
                public Double call() throws Exception
                {
                    double localDot = 0;
                    for(int j = 0; j < N; j++)
                    {
                        if(assignment1[j] != k1)
                            continue;
                        final double w_j = W.get(j);
                        localDot += w_i * w_j * kernel.eval(I, j, X, accel);
                    }
                    
                    return localDot;
                }
            }));
        }
        for(int j = 0; j < N; j++)
            if(assignment1[j] == k1)
                b += W.get(j);
        try
        {
            for (double pDot : ListUtils.collectFutures(dotPartials))
                dot += pDot;
        }
        catch (ExecutionException ex1)
        {
            Logger.getLogger(KernelKMeans.class.getName()).log(Level.SEVERE, null, ex1);
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(KernelKMeans.class.getName()).log(Level.SEVERE, null, ex1);
        }
        return dot/(a*b);
    }
    
    @Override
    abstract public KernelKMeans clone();

    @Override
    public boolean supportsWeightedData()
    {
        return true;
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
