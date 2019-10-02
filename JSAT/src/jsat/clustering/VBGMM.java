/*
 * This code contributed under the Public Domain. 
 */
package jsat.clustering;

import static java.lang.Math.log;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.kmeans.HamerlyKMeans;
import jsat.distributions.multivariate.MultivariateDistribution;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import jsat.math.MathTricks;
import jsat.math.SpecialMath;
import jsat.utils.concurrent.ParallelUtils;

/**
 * The Variational Bayesian Guassian Mixture Model (VBGMM) extends the standard
 * {@link EMGaussianMixture GMM} to adaptively select the number of clusters in
 * the data.<br>
 * <br>
 * See:
 * <ul>
 * <li>﻿ H. Attias, “A Variational Baysian Framework for Graphical Models,” in
 * Advances in Neural Information Processing Systems 12, S. A. Solla, T. K.
 * Leen, and K. Müller, Eds. MIT Press, 2000, pp. 209–215.</li>
 * <li>﻿A. Corduneanu and C. M. Bishop, “Variational Bayesian Model Selection
 * for Mixture Distributions,” in Proceedings Eighth International Conference on
 * Artificial Intelligence and Statistics, 2001, pp. 27–34.</li>
 * </ul>
 *
 * @author Edward Raff
 */
public class VBGMM implements Clusterer, MultivariateDistribution
{
    /**
     * Prior for the Dirichlet distribution
     */
    protected double alpha_0 = 1e-5;
    /**
     * Prior for the mean of each normal
     * 
     */
    protected double beta_0 = 1.0;
    
    private double prune_tol = 1e-5;
    
    protected NormalM[] normals;
    /**
     * The log of the distribution weights pi_{1 ... k}
     */
    protected double[] log_pi;
    
    protected int max_k = 200;
    
    private int maxIterations = 2000;

    public VBGMM() {
    }

    public VBGMM(VBGMM toCopy) 
    {
        this.max_k = toCopy.max_k;
        this.maxIterations = toCopy.maxIterations;
        this.prune_tol = toCopy.prune_tol;
        this.beta_0 = toCopy.beta_0;
        this.alpha_0 = toCopy.alpha_0;
        if(toCopy.normals != null)
        {
            this.normals = Arrays.copyOf(toCopy.normals, toCopy.normals.length);
            for(int i = 0; i < this.normals.length; i++)
                this.normals[i] = this.normals[i].clone();
            this.log_pi = Arrays.copyOf(toCopy.log_pi, toCopy.log_pi.length);
        }
    }
    
    

    @Override
    public int[] cluster(DataSet dataSet, boolean parallel, int[] designations) 
    {
        int k_max = Math.min(max_k, dataSet.size()/2);
        int N = dataSet.size();
        int d = dataSet.getNumNumericalVars();
        List<Vec> X = dataSet.getDataVectors();
        
        normals = new NormalM[k_max];
        boolean[] active = new boolean[k_max];
        Arrays.fill(active, true);
        
        double[][] r = new double[N][k_max];
        
        
        //(10.51)
        double[] N_k = new double[k_max];
        Vec[] X_bar_k = new Vec[k_max];
        Matrix[] S_k = new Matrix[k_max];
        double[] beta = new double[k_max];
        Arrays.fill(beta, d);
        
        double log_prune_tol = Math.log(prune_tol);
        
        /**
         * Dirichlet distribution parameters alpha 
         */
        double[] alpha = new double[k_max];
        
        /**
         * Prior over the means of the dataset, should be set to the mean of the dataset (could be given by the user, but not dealing with that)
         */
        Vec m_0 = new DenseVector(d);
        MatrixStatistics.meanVector(m_0, dataSet);
        /**
         * Prior over the covariances of the dataset. Set from the dataset cov,
         * could be given, but not dealing with that. Its inverse because Bishop
         * deals with the precision matrix, which is the inverse of the
         * covariance.
         */
        Matrix W_inv_0 = new DenseMatrix(d, d);
        MatrixStatistics.covarianceMatrix(m_0, dataSet, W_inv_0);
        
        /**
         * The estimated mean for each cluster
         */
        Vec[] m_k = new Vec[k_max];
        /**
         * The estimated covariance matrix for each cluster
         */
        Matrix[] W_inv_k = new Matrix[k_max];
        for(int k = 0; k < k_max; k++)
        {
            m_k[k] = new DenseVector(d);
            W_inv_k[k] = new DenseMatrix(d, d);
        }
        
        /**
         * Prior over the degrees of freedom in the model. 
         */
        double nu_0 = d;
        
        /**
         * The degrees of freedom
         */
        double[] nu_k = new double[k_max];
        Arrays.fill(nu_k, 1.0);
        
        

        log_pi = new double[k_max];
        /**
         * The log precision term for each component (10.65) in Bishop, or (21.131) in Murphy
         */
        double[] log_precision = new double[k_max];
        
        //Initialization by k-means 
        HamerlyKMeans kMeans = new HamerlyKMeans();
        designations = kMeans.cluster(dataSet, k_max, parallel, designations);
        //Everything is set to 0 right now, so assign to closest
        for(int n = 0; n < N; n++)
        {
            r[n][designations[n]] = 1.0;
            log_pi[designations[n]] += 1;
        }
        //Set central locations based on k-means
        for(int k = 0; k < k_max; k++)
        {
            kMeans.getMeans().get(k).copyTo(m_k[k]);
            if(log_pi[k] == 0)
                active[k]= false;
            log_pi[k] = Math.log(log_pi[k])- Math.log(N);
        }
        //We will leave log_precision alonge as all zeros, since init would have same prior to everyone. So no need to compute
        

        double prevLog = Double.POSITIVE_INFINITY;
        
        for(int iteration = 0; iteration < maxIterations; iteration++)
        {
            
            //M-Step
            ParallelUtils.run(parallel, k_max, (k)->
            {
                if(!active[k])
                    return;

                double Nk = 0.0;
                DenseVector xk = new DenseVector(d);

                for(int n = 0; n < N; n++)
                {
                    double r_nk = r[n][k];
                    Vec x_n = X.get(n);
                    Nk += r_nk;//(10.51) in Bishop
                    xk.mutableAdd(r_nk, x_n);//(10.52) is Bishop
                }

                N_k[k] = Nk;

                //(10.52) is Bishop, finish average
                xk.mutableDivide(Nk + 1e-6);
                X_bar_k[k] = xk;

                Matrix Sk = new DenseMatrix(d, d);
                //(10.53) in Bishop
                DenseVector tmp = new DenseVector(d);
                for(int n = 0; n < N; n++)
                {
                    double r_nk = r[n][k];
                    X.get(n).copyTo(tmp);
                    tmp.mutableSubtract(xk);

                    Matrix.OuterProductUpdate(Sk, tmp, tmp, r_nk);
                }
                Sk.mutableMultiply(1.0/(Nk + 1e-6));
                S_k[k] = Sk;

                //(10.58) in Bishop
                alpha[k] = alpha_0 + Nk;
                //(10.60) in Bishop
                beta[k] = beta_0 + Nk;
                //(10.61) in Bishop
                m_k[k].zeroOut();
                m_k[k].mutableAdd(beta_0, m_0);
                m_k[k].mutableAdd(Nk, xk);
                m_k[k].mutableDivide(beta[k] + 1e-6);
                //(10.62) in Bishop
                W_inv_0.copyTo(W_inv_k[k]);
                W_inv_k[k].mutableAdd(Nk, Sk);
                //adding small value to diagonal to make cov stable
                for(int i = 0; i < d; i++)
                    W_inv_k[k].increment(i, i, 1e-6);
                //note (beta_0 + Nk) denominator is same as (10.60)
                double β0_Nk_over_β0_plus_Nk = beta_0 * Nk / beta[k] ;
                xk.copyTo(tmp);
                tmp.mutableSubtract(m_0);
                Matrix.OuterProductUpdate(W_inv_k[k], tmp, tmp, β0_Nk_over_β0_plus_Nk);
                //(10.63)
                nu_k[k] = nu_0 + Nk;
                
                //Normalize the covariance matrix now so that we don't have to 
                //multiply by nu_k later in in (10.64), makig it easier to re-use
                //the NormalM class
                W_inv_k[k].mutableMultiply(1.0/nu_k[k]);

            });

            //E-step prep
            double alpha_sum = DenseVector.toDenseVec(alpha).sum();
            ParallelUtils.run(parallel, k_max, (k)->
            {
                if(!active[k])
                    return;
                normals[k] = new NormalM(m_k[k], W_inv_k[k]);

                //(10.66) in Bishop
                log_pi[k] = SpecialMath.digamma(alpha[k]) - SpecialMath.digamma(alpha_sum);
                if(log_pi[k] < log_prune_tol)//This cluster has gotten too small, prune it out
                    active[k] = false;
//                else
//                    System.out.println("\t" +Math.exp(log_pi[k]));

                //(10.65) in Bishop, sans log(det) term which will be added by NormalM class later
                log_precision[k] = d * Math.log(2);// + normals[k].getLogCovarianceDeterminant();
                for(int i = 0; i < d; i++)
                    log_precision[k] += SpecialMath.digamma((nu_k[k]-i)/2.0);
                log_precision[k] /= 2;// b/c log(Δ^(1/2)) = 1/2 * log(Δ), and 
                //(10.65) give us log(Δ) but we only use log(Δ^(1/2)) later on
            });

            
            
            //E-Step
            //Fully equation of r is:
            //\ln \rho_{n k}=E\left[\ln \pi_{k}\right]+\frac{1}{2} E\left[\ln \left|\lambda_{k}\right|\right]-\frac{D}{2} \ln (2 \pi)-\frac{1}{2} E_{\mu_{k} \Delta_{k}}\left[\left(\mathbf{x}_{n}-\mu_{k}\right)^{T} \Lambda_{k}\left(\mathbf{x}_{k}-\mu_{k}\right)\right]
            //where r_{n k}=\frac{\rho_{n k}}{\sum_{i=1}^{K} \rho_{n j}}
            double log_prob_sum = ParallelUtils.run(parallel, k_max, (k)->
            {
                if(!active[k])
                {
                    //You have no log prob contribution
                    return 0.0;
                }

                double log_prob_contrib = 0;
                //(10.64) in Bishop, applied to every data point
                for(int n = 0; n < N; n++)
                {
                    //no nu_k multiply in stated (10.64) b/c we normalized the 
                    //covariance matrix earlier

                    //The call to normals also include the log_det factor that 
                    //was supposed to be in (10.65)
                    double proj =  normals[k].logPdf(X.get(n));
                    proj -= d/(2*beta[k]);

                    log_prob_contrib += (r[n][k] = proj + log_pi[k] + log_precision[k]);
                }

                return log_prob_contrib;
            }, (a,b)->a+b);
            
            
//            System.out.println(Math.abs((prevLog-log_prob_sum)/prevLog) + " " + log_prob_sum);
            if(Math.abs((prevLog-log_prob_sum)/prevLog) < 1e-5)
                break;
            prevLog = log_prob_sum;
            

            //Apply exp to r to go from log form to responsibility/contribution form
            //include extra normalization to deal with roundoff of non-active components
            ParallelUtils.run(parallel, N, (n)->
            {
                double sum = 0;
                for(int k = 0; k < k_max; k++)
                    if(active[k])
                        sum += (r[n][k] = Math.exp(r[n][k]));
                for(int k = 0; k < k_max; k++)
                    if(active[k])
                        r[n][k] /= sum;
            });
        }
        
        //How many clusters did we get?
        int still_active = active.length;
        for(boolean still_good : active)
            if(!still_good)
                still_active--;
        int final_k = still_active;
        
        //We've got clusters, lets do some pruning now
        for(int n = 0; n < N; n++)
        {
            int cur_pos = 0;
            //move active indexes up
            int k_max_indx = 0;
            double k_max_value = 0;
            for(int k = 0; k < k_max; k++)
                if(active[k])
                {
                    normals[cur_pos] = normals[k];
                    log_pi[cur_pos] = log_pi[k];
                    if((r[n][cur_pos] = r[n][k]) > k_max_value)
                    {
                        k_max_indx = cur_pos;
                        k_max_value = r[n][cur_pos];
                    }
                    cur_pos++;
                }
            //Mark final cluster id
            designations[n] = k_max_indx;
            r[n] = Arrays.copyOf(r[n], final_k);
        }
        
        normals = Arrays.copyOf(normals, final_k);
        log_pi = Arrays.copyOf(log_pi, final_k);
        
        return designations;
    }

    public void setAlphaPrior(double alpha_0) {
        this.alpha_0 = alpha_0;
    }

    public double getAlphaPrior() {
        return alpha_0;
    }

    public void setBetaPrior(double beta_0) {
        this.beta_0 = beta_0;
    }

    public double getBetaPrior() {
        return beta_0;
    }

    public void setMaxIterations(int maxIterations) 
    {
        this.maxIterations = maxIterations;
    }

    public int getMaxIterations() 
    {
        return maxIterations;
    }
    
    @Override
    public VBGMM clone() 
    {
        return new VBGMM(this);
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
    public double pdf(Vec x) 
    {
        double pdf = 0;
        for(int i = 0; i < normals.length; i++)
            pdf += Math.exp(log_pi[i] + normals[i].logPdf(x));
        return pdf;
    }
    
    public double[] mixtureAssignments(Vec x)
    {
        double[] assignments = new double[normals.length];
        for(int i = 0; i < normals.length; i++)
            assignments[i] = log_pi[i] + normals[i].logPdf(x);
        MathTricks.softmax(assignments, false);
        return assignments;
    }
            
    @Override
    public <V extends Vec> boolean setUsingData(List<V> dataSet, boolean parallel) 
    {
        SimpleDataSet sds = new SimpleDataSet(dataSet.stream().map(v->new DataPoint(v)).collect(Collectors.toList()));
        this.cluster(sds, parallel);
        return true;
    }

    @Override
    public List<Vec> sample(int count, Random rand) 
    {
        List<Vec> samples = new ArrayList<>(count);
        
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
        while(currentGaussian < normals.length)
        {
            a_kSum += Math.exp(log_pi[currentGaussian]);
            while(pos < count && priorTargets[pos++] < a_kSum)
                subSampleSize++;
            samples.addAll(normals[currentGaussian++].sample(subSampleSize, rand));
        }
        
        return samples;
    }
    
}
