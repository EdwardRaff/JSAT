package jsat.clustering;

import jsat.clustering.kmeans.ElkanKMeans;
import static java.lang.Math.log;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.distributions.multivariate.MultivariateDistribution;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.*;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.ListUtils;
import static jsat.utils.SystemInfo.LogicalCores;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * An implementation of Gaussian Mixture models that learns the specified number of Gaussians using Expectation Maximization algorithm. 
 * 
 * @author Edward Raff
 */
public class EMGaussianMixture extends KClustererBase implements MultivariateDistribution
{
    private SeedSelection seedSelection;
    private static final long serialVersionUID = 2606159815670221662L;
    private List<NormalM> gaussians;
    /**
     * The coefficients for the gaussians 
     */
    private double[] a_k;
    private double tolerance = 1e-3;
    /**
     * Control the maximum number of iterations to perform. 
     */
    protected int MaxIterLimit = Integer.MAX_VALUE;

    public EMGaussianMixture(SeedSelection seedSelection)
    {
        setSeedSelection(seedSelection);
    }

    public EMGaussianMixture()
    {
        this(SeedSelection.KPP);
    }
    
    /**
     * Sets the method of seed selection to use for this algorithm.
     * {@link SeedSelection#KPP} is recommended for this algorithm in
     * particular.
     *
     * @param seedSelection the method of seed selection to use
     */
    public void setSeedSelection(SeedSelectionMethods.SeedSelection seedSelection)
    {
        this.seedSelection = seedSelection;
    }

    /**
     * 
     * @return the method of seed selection used
     */
    public SeedSelectionMethods.SeedSelection getSeedSelection()
    {
        return seedSelection;
    }
    
    /**
     * Sets the maximum number of iterations allowed
     * @param iterLimit the maximum number of iterations of the ElkanKMeans algorithm 
     */
    public void setIterationLimit(int iterLimit)
    {
        if(iterLimit < 1)
            throw new IllegalArgumentException("Iterations must be a positive value, not " + iterLimit);
        this.MaxIterLimit = iterLimit;
    }

    /**
     * Returns the maximum number of iterations of the ElkanKMeans algorithm that will be performed. 
     * @return the maximum number of iterations of the ElkanKMeans algorithm that will be performed. 
     */
    public int getIterationLimit()
    {
        return MaxIterLimit;
    }
    
    
    /**
     * Copy constructor. The new Gaussian Mixture can be altered without effecting <tt>gm</tt>
     * @param gm the Guassian Mixture to duplicate
     */
    public EMGaussianMixture(EMGaussianMixture gm)
    {
        if(gm.gaussians != null && !gm.gaussians.isEmpty())
        {
            this.gaussians = new ArrayList<NormalM>(gm.gaussians.size());
            for(NormalM gaussian : gm.gaussians)
                this.gaussians.add(gaussian.clone());
        }
        if(gm.a_k != null)
            this.a_k = Arrays.copyOf(gm.a_k, gm.a_k.length);
        this.MaxIterLimit = gm.MaxIterLimit;
        this.tolerance = gm.tolerance;
    }
    
    /**
     * Copy constructor 
     * @param gaussians value to copy
     * @param a_k value to copy
     * @param tolerance value to copy
     */
    @SuppressWarnings("unused")
	private EMGaussianMixture(List<NormalM> gaussians, double[] a_k, double tolerance)
    {
        this.gaussians = new ArrayList<NormalM>(a_k.length);
        this.a_k = new double[a_k.length];
        for(int i = 0; i < a_k.length; i++)
        {
            this.gaussians.add(gaussians.get(i).clone());
            this.a_k[i] = a_k[i];
        }
    }
    
    protected double cluster(final DataSet dataSet, final List<Double> accelCache, final int K, final List<Vec> means, final int[] assignment, boolean exactTotal, ExecutorService threadpool, boolean returnError)
    {
        EuclideanDistance dm = new EuclideanDistance();
        List<List<Double>> means_qi = new ArrayList<List<Double>>();
        //Pick some initial centers
        if(means.size() < K)
        {
            means.clear();
            means.addAll(SeedSelectionMethods.selectIntialPoints(dataSet, K, dm, accelCache, RandomUtil.getRandom(), seedSelection, threadpool));
            for(Vec v : means)
                means_qi.add(dm.getQueryInfo(v));
        }
        
        
        //Use the initial result to initalize GuassianMixture 
        List<Matrix> covariances = new ArrayList<Matrix>(K);
        int dimension = dataSet.getNumNumericalVars();
        for(int k = 0; k < means.size(); k++)
            covariances.add(new DenseMatrix(dimension, dimension));
        
        a_k = new double[K];
        double sum = dataSet.getSampleSize();
        
        //Compute inital Covariances
        Vec scratch = new DenseVector(dimension);
        List<Vec> X = dataSet.getDataVectors();
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            Vec x = dataSet.getDataPoint(i).getNumericalValues();
            //find out which this belongs to 
            double closest = dm.dist(i, means.get(0), means_qi.get(0), X, accelCache);
            int k = 0;
            for(int j = 1; j < K; j++)//TODO move out and make parallel
            {
                double d_ij = dm.dist(i, means.get(j), means_qi.get(j), X, accelCache);
                if(d_ij < closest)
                {
                    closest = d_ij;
                    k = j;
                }
            }
            assignment[i] = k;
            a_k[k]++;
            x.copyTo(scratch);
            scratch.mutableSubtract(means.get(k));
            Matrix.OuterProductUpdate(covariances.get(k), scratch, scratch, 1.0);
        }
        
        for(int k = 0; k < means.size(); k++)
        {
            covariances.get(k).mutableMultiply(1.0 / a_k[k]);
            a_k[k] /= sum;
        }
        
        
        return clusterCompute(K, dataSet, assignment, means, covariances, threadpool);
    }
    
    protected double clusterCompute(int K, DataSet dataSet, int[] assignment, List<Vec> means, List<Matrix> covs, ExecutorService execServ)
    {
        List<DataPoint> dataPoints = dataSet.getDataPoints();
        int N = dataPoints.size();
        
        double currentLogLike = -Double.MAX_VALUE;

        gaussians = new ArrayList<NormalM>(K);
        
        //Set up initial covariance matrices
        for(int k = 0; k < means.size(); k++)
            gaussians.add(new NormalM(means.get(k), covs.get(k)));
        
        double[][] p_ik = new double[dataPoints.size()][K];
        
        while(true)
        {
            
            
            try
            {
                //E-Step:
                double logLike = eStep(N, dataPoints, K, p_ik, execServ);

                //Convergence check! 
                double logDifference = Math.abs(currentLogLike - logLike);
                if(logDifference < tolerance)
                    break;//We accept this as converged. Probablities could be refined, but no one should be changing class anymore
                else
                    currentLogLike = logLike;
                
                mStep(means, N, dataPoints, K, p_ik, covs, execServ);
            }
            catch (ExecutionException ex)
            {
                Logger.getLogger(EMGaussianMixture.class.getName()).log(Level.SEVERE, null, ex);
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(EMGaussianMixture.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        //Hard asignments based on most probable outcome
        for(int i = 0; i < p_ik.length; i++)
            for(int k = 0; k < K; k++)
                if(p_ik[i][k] > p_ik[i][assignment[i]])
                    assignment[i] = k;
        
        return -currentLogLike;
    }

    private void mStep(final List<Vec> means,final int N,final List<DataPoint> dataPoints, final int K, final double[][] p_ik, final List<Matrix> covs, final ExecutorService execServ) throws InterruptedException
    {
        /**
         * Dimensions
         */
        final int D = means.get(0).length();
        //M-Step

        /**
         *            n
         *          =====
         *          \
         *           >    p
         *          /      i k
         *          =====
         *  p + 1   i = 1
         * a      = ----------
         *  k            n
         */
        //Recompute a_k and update means in the same loop

        for(Vec mean : means)
            mean.zeroOut();

        Arrays.fill(a_k, 0.0);
        
        if(execServ == null)
        {
            for(int i = 0; i < N; i++)
            {
                Vec x_i = dataPoints.get(i).getNumericalValues();
                for(int k = 0; k < K; k++)
                {
                    a_k[k] += p_ik[i][k];
                    means.get(k).mutableAdd(p_ik[i][k], x_i);
                }
            }
        }
        else//Parllalle version is limited in scalability to the number of clusters k, as we are updating the k's so we can not distribute row wise, but must give each therad its own k
        {   
            final CountDownLatch latch = new CountDownLatch(LogicalCores);
            int start = 0; 
            int step = N / LogicalCores;
            int remainder = N % LogicalCores;
            while( start < N)
            {
                final int to = Math.min((remainder-- > 0  ? 1 : 0) + start + step, N);
                final int Start = start;
                start = to;
                execServ.submit(new Runnable() {

                    @Override
                    public void run()
                    {
                        Vec[] partialMean = new Vec[means.size()];
                        for(int i = 0; i < partialMean.length; i++)
                            partialMean[i] = new DenseVector(means.get(i).length());
                        double[] partial_a_k = new double[a_k.length];
                        
                        for(int i = Start; i < to; i++)
                        {
                            Vec x_i = dataPoints.get(i).getNumericalValues();
                            for(int k = 0; k < K; k++)
                            {
                                partial_a_k[k] += p_ik[i][k];
                                partialMean[k].mutableAdd(p_ik[i][k], x_i);
                            }
                        }
                        
                        synchronized(means)
                        {
                            for(int k = 0; k < a_k.length; k++)
                            {
                                a_k[k] += partial_a_k[k];
                                means.get(k).mutableAdd(partialMean[k]);
                            }
                        }
                        latch.countDown();
                        
                    }
                });
            }
            
            latch.await();
        }

        //We can now dived all the means by their sums, which are stored in a_k, and then normalized a_k after
        for(int k = 0; k < a_k.length; k++)
            means.get(k).mutableDivide(a_k[k]);

        //We hold off on nomralized a_k, becase we will use its values to update the covariances

        for(Matrix cov : covs)
            cov.zeroOut();

        if(execServ == null)
        {
            for(int k = 0; k < K; k++)
            {
                Matrix covariance = covs.get(k);
                Vec mean = means.get(k);
                Vec scratch = new DenseVector(mean.length());
                for(int i = 0; i < dataPoints.size(); i++)
                {
                    DataPoint dp = dataPoints.get(i);
                    Vec x = dp.getNumericalValues();
                    x.copyTo(scratch);
                    scratch.mutableSubtract(mean);
                    Matrix.OuterProductUpdate(covariance, scratch, scratch, p_ik[i][k]);
                }
                covariance.mutableMultiply(1.0 / (a_k[k]));
            }
        }
        else
        {
            final CountDownLatch latch = new CountDownLatch(LogicalCores);
            int start = 0; 
            int step = N / LogicalCores;
            int remainder = N % LogicalCores;
            while( start < N)
            {
                final int to = Math.min((remainder-- > 0  ? 1 : 0) + start + step, N);
                final int Start = start;
                start = to;
                execServ.submit(new Runnable() {

                    @Override
                    public void run()
                    {
                        Matrix[] partialCovs = new Matrix[K];
                        for(int i = 0; i < partialCovs.length; i++)
                            partialCovs[i] = new DenseMatrix(D, D);
                        
                        for(int i = Start; i < to; i++)
                        {
                            DataPoint dp = dataPoints.get(i);
                            Vec x = dp.getNumericalValues();
                            Vec scratch = new DenseVector(x.length());
                            
                            for(int k = 0; k < K; k++)
                            {
                                Matrix covariance = partialCovs[k];
                                Vec mean = means.get(k);
                                
                                x.copyTo(scratch);
                                scratch.mutableSubtract(mean);
                                Matrix.OuterProductUpdate(covariance, scratch, scratch, p_ik[i][k]);
                            }
                            
                        }
                        
                        synchronized(covs)
                        {
                            for(int  k = 0; k < K; k++)
                                covs.get(k).mutableAdd(partialCovs[k]);
                        }
                        
                        
                        latch.countDown();
                    }
                });
            }
            latch.await();
            
            for(int k = 0; k < K; k++)
                covs.get(k).mutableMultiply(1.0/a_k[k]);
            
        }

        //Finaly, normalize the coefficents
        for(int k = 0; k < K; k++)
            a_k[k] /= N;
        
        //And update the Normals
        for(int k = 0; k < means.size(); k++)
            gaussians.get(k).setMeanCovariance(means.get(k), covs.get(k));
    }

    private double eStep(final int N, final List<DataPoint> dataPoints, final int K, final double[][] p_ik, final ExecutorService execServ) throws InterruptedException, ExecutionException
    {
        double logLike = 0;
       /*
        *            p  /   |     p       p\
        *           a  P|x  | mean , Sigma |
        *            k  \ i |     k       k/
        * p    = ------------------------------
        *  i k     K
        *        =====
        *        \      p  /   |     p       p\
        *         >    a  P|x  | mean , Sigma |
        *        /      k  \ i |     k       k/
        *        =====
        *        k = 1
        */

       /*
        * We will piggy back off the E step to compute the log likelyhood 
        * 
        *                 N      /  K                           \
        *               =====    |=====                         |
        *               \        |\                             |
        * L(x, Theat) =  >    log| >    a  P/x  | mean , Sigma \|
        *               /        |/      k  \ i |     k       k/|
        *               =====    |=====                         |
        *               i = 1    \k = 1                         /
        */
        
        if(execServ == null)
        {
            for(int i = 0; i < N; i++)
            {
                Vec x_i = dataPoints.get(i).getNumericalValues();
                double p_ikNormalizer = 0.0;

                for(int k = 0; k < K; k++)
                {
                    double tmp = a_k[k] * gaussians.get(k).pdf(x_i);
                    p_ik[i][k] = tmp;
                    p_ikNormalizer += tmp;
                }

                //Normalize previous values
                for(int k = 0; k < K; k++)
                    p_ik[i][k] /= p_ikNormalizer;

                //Add to part of the log likelyhood 
                logLike += Math.log(p_ikNormalizer);
            }
        }
        else 
        {
            List<Future<Double>> partialLogLikes = new ArrayList<Future<Double>>(LogicalCores);
            int start = 0; 
            int step = N / LogicalCores;
            int remainder = N % LogicalCores;
            while( start < N)
            {
                final int to = Math.min((remainder-- > 0  ? 1 : 0) + start + step, N);
                final int Start = start;
                start = to;
                
                partialLogLikes.add(execServ.submit(new Callable<Double>() 
                {

                    @Override
                    public Double call() throws Exception
                    {
                        double partialLog = 0;
                        for(int i = Start; i < to; i++)
                        {
                            Vec x_i = dataPoints.get(i).getNumericalValues();
                            double p_ikNormalizer = 0.0;

                            for(int k = 0; k < K; k++)
                            {
                                double tmp = a_k[k] * gaussians.get(k).pdf(x_i);
                                p_ik[i][k] = tmp;
                                p_ikNormalizer += tmp;
                            }

                            //Normalize previous values
                            for(int k = 0; k < K; k++)
                                p_ik[i][k] /= p_ikNormalizer;

                            //Add to part of the log likelyhood 
                            partialLog += Math.log(p_ikNormalizer);
                        }
                        
                        return partialLog;
                    }
                }));
            }
            
            for(double partialLogLike : ListUtils.collectFutures(partialLogLikes))
                logLike += partialLogLike;
        }
        return logLike;
    }

    @Override
    public double logPdf(double... x)
    {
        return logPdf(DenseVector.toDenseVec(x));
    }

    @Override
    public double logPdf(Vec x)
    {
        double pdf = pdf(x);
        if(pdf == 0)
            return -Double.MAX_VALUE;
        return log(pdf);
    }

    @Override
    public double pdf(double... x)
    {
        return pdf(DenseVector.toDenseVec(x));
    }

    @Override
    public double pdf(Vec x)
    {
        double PDF = 0.0;
        for(int i = 0; i < a_k.length; i++)
            PDF += a_k[i] * gaussians.get(i).pdf(x);
        return PDF;
    }

    @Override
    public <V extends Vec> boolean setUsingData(List<V> dataSet)
    {
        List<DataPoint> dataPoints = new ArrayList<DataPoint>(dataSet.size());
        for(Vec x :  dataSet)
            dataPoints.add(new DataPoint(x, new int[0], new CategoricalData[0]));
        return setUsingDataList(dataPoints);
    }

    @Override
    public boolean setUsingDataList(List<DataPoint> dataPoint)
    {
        return setUsingData(new SimpleDataSet(dataPoint));
    }

    @Override
    public boolean setUsingData(DataSet dataSet)
    {
        try
        {
            cluster(dataSet);
            return true;
        }
        catch (ArithmeticException ex)
        {
            return false;
        }
    }

    @Override
    public boolean setUsingData(DataSet dataSet, ExecutorService threadpool)
    {
        return setUsingData(dataSet);
    }

    @Override
    public <V extends Vec> boolean setUsingData(List<V> dataSet, ExecutorService threadpool)
    {
        return setUsingData(dataSet);
    }

    @Override
    public boolean setUsingDataList(List<DataPoint> dataPoints, ExecutorService threadpool)
    {
        return setUsingDataList(dataPoints);
    }

    @Override
    public EMGaussianMixture clone()
    {
        return new EMGaussianMixture(this);
    }

    @Override
    public List<Vec> sample(int count, Random rand)
    {
        List<Vec> samples = new ArrayList<Vec>(count);
        
        //First we need the figure out which of the mixtures to sample from
        //So generate [0,1] uniform values to determine 
        double[] priorTargets = new double[count];
        for(int i = 0; i < count; i++)
            priorTargets[i] = rand.nextDouble();
        Arrays.sort(priorTargets);
        int subSampleSize = 0;
        int currentGaussian = 0;
        int pos = 0;
        double a_kSum = 0.0;
        while(currentGaussian < a_k.length)
        {
            a_kSum += a_k[currentGaussian];
            while(pos < count && priorTargets[pos++] < a_kSum)
                subSampleSize++;
            samples.addAll(gaussians.get(currentGaussian++).sample(subSampleSize, rand));
        }
        
        return samples;
    }

    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        return cluster(dataSet, 2, (int)Math.sqrt(dataSet.getSampleSize()/2), threadpool, designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < clusters)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        
        List<Vec> means = new ArrayList<Vec>(clusters);
        cluster(dataSet, null, clusters, means, designations, false, threadpool, false);
        return designations;
    }

    @Override
    public int[] cluster(DataSet dataSet, int clusters, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        if(dataSet.getSampleSize() < clusters)
            throw new ClusterFailureException("Fewer data points then desired clusters, decrease cluster size");
        List<Vec> means = new ArrayList<Vec>(clusters);
        cluster(dataSet, null, clusters, means, designations, false, null, false);
        
        return designations;
    }
    
    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, ExecutorService threadpool, int[] designations)
    {
        throw new UnsupportedOperationException("EMGaussianMixture does not supported determining the number of clusters"); 
    }

    @Override
    public int[] cluster(DataSet dataSet, int lowK, int highK, int[] designations)
    {
        throw new UnsupportedOperationException("EMGaussianMixture does not supported determining the number of clusters"); 
    }
}
