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
import jsat.utils.ListUtils;
import static jsat.utils.SystemInfo.LogicalCores;

/**
 * An implementation of Gaussian Mixture models that learns the specified number of Gaussians using Expectation Maximization algorithm. 
 * 
 * @author Edward Raff
 */
public class EMGaussianMixture extends ElkanKMeans implements MultivariateDistribution
{

	private static final long serialVersionUID = 2606159910420221662L;
	private List<NormalM> gaussians;
    /**
     * The coefficients for the gaussians 
     */
    private double[] a_k;
    private double tolerance = 1e-3;

    public EMGaussianMixture(final DistanceMetric dm, final Random rand, final SeedSelection seedSelection)
    {
        super(dm, rand, seedSelection);
    }

    public EMGaussianMixture(final DistanceMetric dm, final Random rand)
    {
        super(dm, rand);
    }

    public EMGaussianMixture(final DistanceMetric dm)
    {
        super(dm);
    }

    public EMGaussianMixture()
    {
        super();
    }
    
    /**
     * Copy constructor. The new Gaussian Mixture can be altered without effecting <tt>gm</tt>
     * @param gm the Guassian Mixture to duplicate
     */
    public EMGaussianMixture(final EMGaussianMixture gm)
    {
        if(gm.gaussians != null && !gm.gaussians.isEmpty())
        {
            this.gaussians = new ArrayList<NormalM>(gm.gaussians.size());
            for(final NormalM gaussian : gm.gaussians) {
              this.gaussians.add(gaussian.clone());
            }
        }
        if(gm.a_k != null) {
          this.a_k = Arrays.copyOf(gm.a_k, gm.a_k.length);
        }
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
	private EMGaussianMixture(final List<NormalM> gaussians, final double[] a_k, final double tolerance)
    {
        this.gaussians = new ArrayList<NormalM>(a_k.length);
        this.a_k = new double[a_k.length];
        for(int i = 0; i < a_k.length; i++)
        {
            this.gaussians.add(gaussians.get(i).clone());
            this.a_k[i] = a_k[i];
        }
    }
    
    @Override
    protected double cluster(final DataSet dataSet, final List<Double> accelCache, final int K, final List<Vec> means, final int[] assignment, final boolean exactTotal, final ExecutorService threadpool, final boolean returnError)
    {
        //Perform intial clustering with ElkanKMeans 
        super.cluster(dataSet, accelCache, K, means, assignment, exactTotal, threadpool, false);
        
        //Use the ElkanKMeans result to initalize GuassianMixture 
        final List<Matrix> covariances = new ArrayList<Matrix>(K);
        final int dimension = dataSet.getNumNumericalVars();
        for(int k = 0; k < means.size(); k++) {
          covariances.add(new DenseMatrix(dimension, dimension));
        }
        
        a_k = new double[K];
        final double sum = dataSet.getSampleSize();
        
        //Compute inital Covariances
        final Vec scratch = new DenseVector(dimension);
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            final Vec x = dataSet.getDataPoint(i).getNumericalValues();
            final int k = assignment[i];
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
    
    protected double clusterCompute(final int K, final DataSet dataSet, final int[] assignment, final List<Vec> means, final List<Matrix> covs, final ExecutorService execServ)
    {
        final List<DataPoint> dataPoints = dataSet.getDataPoints();
        final int N = dataPoints.size();
        
        double currentLogLike = -Double.MAX_VALUE;

        gaussians = new ArrayList<NormalM>(K);
        
        //Set up initial covariance matrices
        for(int k = 0; k < means.size(); k++) {
          gaussians.add(new NormalM(means.get(k), covs.get(k)));
        }
        
        final double[][] p_ik = new double[dataPoints.size()][K];
        
        while(true)
        {
            
            
            try
            {
                //E-Step:
                final double logLike = eStep(N, dataPoints, K, p_ik, execServ);

                //Convergence check! 
                final double logDifference = Math.abs(currentLogLike - logLike);
                if(logDifference < tolerance) {
                  break;//We accept this as converged. Probablities could be refined, but no one should be changing class anymore
                } else {
                  currentLogLike = logLike;
                }
                
                mStep(means, N, dataPoints, K, p_ik, covs, execServ);
            }
            catch (final ExecutionException ex)
            {
                Logger.getLogger(EMGaussianMixture.class.getName()).log(Level.SEVERE, null, ex);
            }
            catch (final InterruptedException ex)
            {
                Logger.getLogger(EMGaussianMixture.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        //Hard asignments based on most probable outcome
        for(int i = 0; i < p_ik.length; i++) {
          for (int k = 0; k < K; k++) {
            if (p_ik[i][k] > p_ik[i][assignment[i]]) {
              assignment[i] = k;
            }
          }
        }
        
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

        for(final Vec mean : means) {
          mean.zeroOut();
        }

        Arrays.fill(a_k, 0.0);
        
        if(execServ == null)
        {
            for(int i = 0; i < N; i++)
            {
                final Vec x_i = dataPoints.get(i).getNumericalValues();
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
            final int step = N / LogicalCores;
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
                        final Vec[] partialMean = new Vec[means.size()];
                        for(int i = 0; i < partialMean.length; i++) {
                          partialMean[i] = new DenseVector(means.get(i).length());
                        }
                        final double[] partial_a_k = new double[a_k.length];
                        
                        for(int i = Start; i < to; i++)
                        {
                            final Vec x_i = dataPoints.get(i).getNumericalValues();
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
        for(int k = 0; k < a_k.length; k++) {
          means.get(k).mutableDivide(a_k[k]);
        }

        //We hold off on nomralized a_k, becase we will use its values to update the covariances

        for(final Matrix cov : covs) {
          cov.zeroOut();
        }

        if(execServ == null)
        {
            for(int k = 0; k < K; k++)
            {
                final Matrix covariance = covs.get(k);
                final Vec mean = means.get(k);
                final Vec scratch = new DenseVector(mean.length());
                for(int i = 0; i < dataPoints.size(); i++)
                {
                    final DataPoint dp = dataPoints.get(i);
                    final Vec x = dp.getNumericalValues();
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
            final int step = N / LogicalCores;
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
                        final Matrix[] partialCovs = new Matrix[K];
                        for(int i = 0; i < partialCovs.length; i++) {
                          partialCovs[i] = new DenseMatrix(D, D);
                        }
                        
                        for(int i = Start; i < to; i++)
                        {
                            final DataPoint dp = dataPoints.get(i);
                            final Vec x = dp.getNumericalValues();
                            final Vec scratch = new DenseVector(x.length());
                            
                            for(int k = 0; k < K; k++)
                            {
                                final Matrix covariance = partialCovs[k];
                                final Vec mean = means.get(k);
                                
                                x.copyTo(scratch);
                                scratch.mutableSubtract(mean);
                                Matrix.OuterProductUpdate(covariance, scratch, scratch, p_ik[i][k]);
                            }
                            
                        }
                        
                        synchronized(covs)
                        {
                            for(int  k = 0; k < K; k++) {
                              covs.get(k).mutableAdd(partialCovs[k]);
                            }
                        }
                        
                        
                        latch.countDown();
                    }
                });
            }
            latch.await();
            
            for(int k = 0; k < K; k++) {
              covs.get(k).mutableMultiply(1.0/a_k[k]);
            }
            
        }

        //Finaly, normalize the coefficents
        for(int k = 0; k < K; k++) {
          a_k[k] /= N;
        }
        
        //And update the Normals
        for(int k = 0; k < means.size(); k++) {
          gaussians.get(k).setMeanCovariance(means.get(k), covs.get(k));
        }
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
                final Vec x_i = dataPoints.get(i).getNumericalValues();
                double p_ikNormalizer = 0.0;

                for(int k = 0; k < K; k++)
                {
                    final double tmp = a_k[k] * gaussians.get(k).pdf(x_i);
                    p_ik[i][k] = tmp;
                    p_ikNormalizer += tmp;
                }

                //Normalize previous values
                for(int k = 0; k < K; k++) {
                  p_ik[i][k] /= p_ikNormalizer;
                }

                //Add to part of the log likelyhood 
                logLike += Math.log(p_ikNormalizer);
            }
        }
        else 
        {
            final List<Future<Double>> partialLogLikes = new ArrayList<Future<Double>>(LogicalCores);
            int start = 0; 
            final int step = N / LogicalCores;
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
                            final Vec x_i = dataPoints.get(i).getNumericalValues();
                            double p_ikNormalizer = 0.0;

                            for(int k = 0; k < K; k++)
                            {
                                final double tmp = a_k[k] * gaussians.get(k).pdf(x_i);
                                p_ik[i][k] = tmp;
                                p_ikNormalizer += tmp;
                            }

                            //Normalize previous values
                            for(int k = 0; k < K; k++) {
                              p_ik[i][k] /= p_ikNormalizer;
                            }

                            //Add to part of the log likelyhood 
                            partialLog += Math.log(p_ikNormalizer);
                        }
                        
                        return partialLog;
                    }
                }));
            }
            
            for(final double partialLogLike : ListUtils.collectFutures(partialLogLikes)) {
              logLike += partialLogLike;
            }
        }
        return logLike;
    }

    @Override
    public double logPdf(final double... x)
    {
        return logPdf(DenseVector.toDenseVec(x));
    }

    @Override
    public double logPdf(final Vec x)
    {
        final double pdf = pdf(x);
        if(pdf == 0) {
          return -Double.MAX_VALUE;
        }
        return log(pdf);
    }

    @Override
    public double pdf(final double... x)
    {
        return pdf(DenseVector.toDenseVec(x));
    }

    @Override
    public double pdf(final Vec x)
    {
        double PDF = 0.0;
        for(int i = 0; i < a_k.length; i++) {
          PDF += a_k[i] * gaussians.get(i).pdf(x);
        }
        return PDF;
    }

    @Override
    public <V extends Vec> boolean setUsingData(final List<V> dataSet)
    {
        final List<DataPoint> dataPoints = new ArrayList<DataPoint>(dataSet.size());
        for(final Vec x :  dataSet) {
          dataPoints.add(new DataPoint(x, new int[0], new CategoricalData[0]));
        }
        return setUsingDataList(dataPoints);
    }

    @Override
    public boolean setUsingDataList(final List<DataPoint> dataPoint)
    {
        return setUsingData(new SimpleDataSet(dataPoint));
    }

    @Override
    public boolean setUsingData(final DataSet dataSet)
    {
        try
        {
            cluster(dataSet);
            return true;
        }
        catch (final ArithmeticException ex)
        {
            return false;
        }
    }

    @Override
    public boolean setUsingData(final DataSet dataSet, final ExecutorService threadpool)
    {
        return setUsingData(dataSet);
    }

    @Override
    public <V extends Vec> boolean setUsingData(final List<V> dataSet, final ExecutorService threadpool)
    {
        return setUsingData(dataSet);
    }

    @Override
    public boolean setUsingDataList(final List<DataPoint> dataPoints, final ExecutorService threadpool)
    {
        return setUsingDataList(dataPoints);
    }

    @Override
    public EMGaussianMixture clone()
    {
        return new EMGaussianMixture(this);
    }

    @Override
    public List<Vec> sample(final int count, final Random rand)
    {
        final List<Vec> samples = new ArrayList<Vec>(count);
        
        //First we need the figure out which of the mixtures to sample from
        //So generate [0,1] uniform values to determine 
        final double[] priorTargets = new double[count];
        for(int i = 0; i < count; i++) {
          priorTargets[i] = rand.nextDouble();
        }
        Arrays.sort(priorTargets);
        int subSampleSize = 0;
        int currentGaussian = 0;
        int pos = 0;
        double a_kSum = 0.0;
        while(currentGaussian < a_k.length)
        {
            a_kSum += a_k[currentGaussian];
            while(pos < count && priorTargets[pos++] < a_kSum) {
              subSampleSize++;
            }
            samples.addAll(gaussians.get(currentGaussian++).sample(subSampleSize, rand));
        }
        
        return samples;
    }
}
