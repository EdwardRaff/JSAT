package jsat.clustering;

import jsat.linear.DenseVector;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.distributions.multivariate.MultivariateDistribution;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import static jsat.clustering.SeedSelectionMethods.*;
import static java.lang.Math.*;

/**
 * An implementation of Gaussian Mixture models that learns the specified number of Gaussians using Expectation Maximization algorithm. 
 * 
 * @author Edward Raff
 */
public class EMGaussianMixture extends KMeans implements MultivariateDistribution
{
    private List<NormalM> gaussians;
    /**
     * The coefficients for the gaussians 
     */
    private double[] a_k;
    private double tolerance = 1e-3;

    public EMGaussianMixture(DistanceMetric dm, Random rand, SeedSelection seedSelection)
    {
        super(dm, rand, seedSelection);
    }

    public EMGaussianMixture(DistanceMetric dm, Random rand)
    {
        super(dm, rand);
    }

    public EMGaussianMixture(DistanceMetric dm)
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
        this.iterLimit = gm.iterLimit;
        this.tolerance = gm.tolerance;
    }
    
    /**
     * Copy constructor 
     * @param gaussians value to copy
     * @param a_k value to copy
     * @param tolerance value to copy
     */
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
    
    @Override
    protected double cluster(DataSet dataSet, List<Vec> means, int[] assignment, boolean exactTotal)
    {
        //Perform intial clustering with KMeans 
        super.cluster(dataSet, means, assignment, exactTotal);
        
        int K = means.size();
        //Use the KMeans result to initalize GuassianMixture 
        List<Matrix> covariances = new ArrayList<Matrix>(K);
        int dimension = dataSet.getNumNumericalVars();
        for(int k = 0; k < means.size(); k++)
            covariances.add(new DenseMatrix(dimension, dimension));
        
        a_k = new double[K];
        double sum = dataSet.getSampleSize();
        
        //Compute inital Covariances
        Vec scratch = new DenseVector(dimension);
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            Vec x = dataSet.getDataPoint(i).getNumericalValues();
            int k = assignment[i];
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
        
        
        return clusterCompute(K, dataSet, assignment, means, covariances);
    }
    
    protected double clusterCompute(int K, DataSet dataSet, int[] assignment, List<Vec> means, List<Matrix> covs)
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
            //E-Step:
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
            
            double logLike = 0;
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
            
            //Convergence check! 
            double logDifference = Math.abs(currentLogLike - logLike);
            if(logDifference < tolerance)
                break;//We accept this as converged. Probablities could be refined, but no one should be changing class anymore
            else
                currentLogLike = logLike;
        


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
            for(int i = 0; i < N; i++)
            {
                Vec x_i = dataPoints.get(i).getNumericalValues();
                for(int k = 0; k < K; k++)
                {
                    a_k[k] += p_ik[i][k];
                    means.get(k).mutableAdd(p_ik[i][k], x_i);
                }
            }

            //We can now dived all the means by their sums, which are stored in a_k, and then normalized a_k after
            for(int k = 0; k < a_k.length; k++)
                means.get(k).mutableDivide(a_k[k]);

            //We hold off on nomralized a_k, becase we will use its values to update the covariances

            for(Matrix cov : covs)
                cov.zeroOut();

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

            //Finaly, normalize the coefficents
            for(int k = 0; k < K; k++)
                a_k[k] /= N;
            
            //And update the Normals
            for(int k = 0; k < means.size(); k++)
                gaussians.get(k).setMeanCovariance(means.get(k), covs.get(k));
        }
        
        //Hard asignments based on most probable outcome
        for(int i = 0; i < p_ik.length; i++)
            for(int k = 0; k < K; k++)
                if(p_ik[i][k] > p_ik[i][assignment[i]])
                    assignment[i] = k;
        
        return -currentLogLike;
    }

    public double logPdf(double... x)
    {
        return logPdf(DenseVector.toDenseVec(x));
    }

    public double logPdf(Vec x)
    {
        double pdf = pdf(x);
        if(pdf == 0)
            return -Double.MAX_VALUE;
        return log(pdf);
    }

    public double pdf(double... x)
    {
        return pdf(DenseVector.toDenseVec(x));
    }

    public double pdf(Vec x)
    {
        double PDF = 0.0;
        for(int i = 0; i < a_k.length; i++)
            PDF += a_k[i] * gaussians.get(i).pdf(x);
        return PDF;
    }

    public boolean setUsingData(List<Vec> dataSet)
    {
        List<DataPoint> dataPoints = new ArrayList<DataPoint>(dataSet.size());
        for(Vec x :  dataSet)
            dataPoints.add(new DataPoint(x, new int[0], new CategoricalData[0]));
        return setUsingDataList(dataPoints);
    }

    public boolean setUsingDataList(List<DataPoint> dataPoint)
    {
        return setUsingData(new SimpleDataSet(dataPoint));
    }

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
    public EMGaussianMixture clone()
    {
        return new EMGaussianMixture(this);
    }

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
}
