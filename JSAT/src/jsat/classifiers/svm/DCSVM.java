/*
 * Copyright (C) 2016 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.classifiers.svm;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.classifiers.neuralnetwork.RBFNet;
import jsat.clustering.kmeans.ElkanKernelKMeans;
import jsat.clustering.kmeans.KernelKMeans;
import jsat.clustering.kmeans.LloydKernelKMeans;
import jsat.distributions.kernels.KernelTrick;
import jsat.distributions.kernels.RBFKernel;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;

/**
 * This is an implementation of the Divide-and-Conquer Support Vector Machine
 * (DC-SVM). It uses a a combination of clustering and warm-starting to train
 * faster, as well as an early stopping strategy to provide a fast approximate
 * SVM solution. The final accuracy should often be at or near that of a normal
 * SVM, while being faster to train. <br>
 * <br>
 * The current implementation is based on {@link SVMnoBias}, meaning this code
 * does not have a bias term and it only works with normalized kernels. Any
 * non-normalized kernel will be normalized automatically. This is not a problem
 * for the common RBF kernel.<br>
 * <br>
 *
 * See:
 * <ul>
 * <li>Hsieh, C.-J., Si, S., & Dhillon, I. S. (2014). <i>A Divide-and-Conquer
 * Solver for Kernel Support Vector Machines</i>. In Proceedings of the 31st
 * International Conference on Machine Learning. Beijing, China.</li>
 * </ul>
 *
 * @author Edward Raff
 */
public class DCSVM extends SupportVectorLearner implements Classifier, Parameterized, BinaryScoreClassifier
{
    private double C = 1;
    private double tolerance = 1e-3;
    
    private KernelKMeans clusters;
    private int m = 2000;
    private int l_max = 4;
    private int l_early = 3;
    private int k = 4;
    
    
    private Map<Integer, SVMnoBias> early_models;
    private long cache_size = 0;

    /**
     * Creates a new DC-SVM for the given kernel
     * @param k the kernel to use
     */
    public DCSVM(KernelTrick k)
    {
        super(k, CacheMode.ROWS);
        this.cache_size = Runtime.getRuntime().freeMemory()/2;
    }

    /**
     * Creates a new DC-SVM for the RBF kernel
     */
    public DCSVM()
    {
        this(new RBFKernel());
    }

    /**
     * Copy Constructor
     * @param toCopy object to copy
     */
    public DCSVM(DCSVM toCopy)
    {
        super(toCopy);
        this.C = toCopy.C;
        this.tolerance = toCopy.tolerance;
        if(toCopy.clusters != null)
            this.clusters = toCopy.clusters.clone();
        this.cache_size = toCopy.cache_size;
        this.m = toCopy.m;
        this.l_early = toCopy.l_early;
        this.l_max = toCopy.l_max;
        this.k = toCopy.k;
        if(toCopy.early_models != null)
        {
            this.early_models = new ConcurrentHashMap<Integer, SVMnoBias>();
            for(Map.Entry<Integer, SVMnoBias> x : toCopy.early_models.entrySet())
                this.early_models.put(x.getKey(), x.getValue().clone());
        }
    }
    
    

    /**
     * The DC-SVM algorithm works by creating a hierarchy of levels, and
     * iteratively refining the solution from one level to the next. Level 0
     * corresponds to the exact SVM solution, and higher levels are courser
     * approximations. This method controls which level the training starts at. 
     * 
     * @param l_max which level to start the training at. 
     */
    public void setStartLevel(int l_max)
    {
        if(l_max < 0)
            throw new IllegalArgumentException("l_max must be a non-negative integer, not " + l_max);
        this.l_max = l_max;
    }

    /**
     * 
     * @return which level to start the training at. 
     */
    public int getStartLevel()
    {
        return l_max;
    }

    /**
     * The DC-SVM algorithm works by creating a hierarchy of levels, and
     * iteratively refining the solution from one level to the next. Level 0
     * corresponds to the exact SVM solution, and higher levels are courser
     * approximations. This method controls which level the training stops at,
     * with 0 being the latest it can stop. The default stopping level is 3.
     *
     * @param l_early which level to stop the training at, and use for
     * classification.
     */
    public void setEndLevel(int l_early)
    {
        if(l_early < 0)
            throw new IllegalArgumentException("l_early must be a non-negative integer, not " + l_early);
        this.l_early = l_early;
    }

    /**
     * 
     * @return which level to stop the training at, and use for
     * classification.
     */
    public int getEndLevel()
    {
        return l_early;
    }

    /**
     * At each level of the DC-SVM training, a clustering algorithm is used to
     * divide the dataset into sub-groups for independent training. Increasing
     * the number of points used for clustering improves model accuracy, but
     * also increases training time. The default value is 2000. This value may
     * need to be increased if using a higher staring level.
     *
     * @param m the number of data points to sample for each cluster size
     */
    public void setClusterSampleSize(int m)
    {
        if(m <= 0)
            throw new IllegalArgumentException("Cluster Sample Size must be a positive integer, not " + m);
        this.m = m;
    }

    public int getClusterSampleSize()
    {
        return m;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);

        double sum = getScore(data);

        if (sum > 0)
            cr.setProb(1, 1);
        else
            cr.setProb(0, 1);

        return cr;
    }
    
    @Override
    public double getScore(DataPoint dp)
    {
        if (vecs == null)
            throw new UntrainedModelException("Classifier has yet to be trained");
        
        Vec x = dp.getNumericalValues();
        int c;
        if(early_models.size() > 1)
            c = clusters.findClosestCluster(x, getKernel().getQueryInfo(x));
        else
            c = 0;
        return early_models.get(c).getScore(dp);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        final int threads_to_use;
        if(threadPool instanceof FakeExecutor)
            threads_to_use = 1;
        else
            threads_to_use = SystemInfo.LogicalCores;
        
        final int N = dataSet.getSampleSize();
        vecs = dataSet.getDataVectors();
        early_models = new ConcurrentHashMap<Integer, SVMnoBias>();
//        weights = dataSet.getDataWeights();
//        label = new short[N];
//        for(int i = 0; i < N; i++)
//            label[i] = (short) (dataSet.getDataPointCategory(i)*2-1);
        setCacheMode(CacheMode.NONE);//Initiates the accel cache
        //initialize alphas array to all zero
        alphas = new double[N];//zero is default value
        
        /**
         * Used to keep track of which sub cluster each training datapoint belongs to
         */
        final int[] group = new int[N];
                
        /**
         * Used to select subsamples of data points for clustering, and to map them back to their original indicies 
         */
        IntList indicies = new IntList();
        //for l = lmax, . . . , 1 do
        for(int l = l_max; l >= l_early; l--)
        {
//            System.out.println("Level " + l);
            early_models.clear();
            //sub-sampled dataset to use for clustering
            ClassificationDataSet toCluster = new ClassificationDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories(), dataSet.getPredicting());
            //Set number of clusters in the current level k_l = k^l
            int k_l = (int) Math.pow(k, l);
            
            //number of datapoints to use in clustering 
            //increase M = m by default. Increase to M=7 m if less than 7 points per cluster
            int M;
            if( N/k_l < 7 )
                M = k_l*7;
            else
                M = m;
            
            if(l == l_max)
            {
                ListUtils.addRange(indicies, 0, N, 1);
                Collections.shuffle(indicies);
                for(int i = 0; i < Math.min(M, N); i++)
                    toCluster.addDataPoint(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i));
            }
            else
            {
                indicies.clear();
                for(int i = 0; i < N; i++)
                    if(alphas[i] != 0)
                        indicies.add(i);
                Collections.shuffle(indicies);
                for(int i = 0; i < Math.min(M, indicies.size()); i++)
                    toCluster.addDataPoint(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i));
            }
            //Run kernel kmeans on {xi1, . . . ,xim} to get cluster centers c1, . . . , ckl ;
            clusters = new ElkanKernelKMeans(getKernel());
            clusters.setMaximumIterations(100);
//            System.out.println("Finding " + k_l + " clusters");
            k_l = Math.min(k_l, toCluster.getSampleSize()/2);//Few support vectors? Make clustering smaller then
            int[] sub_results;
            if(k_l <= 1)//dont run cluster, we are doing final refinement step!
            {
                sub_results = new int[N];//will be all 0, for 1 'cluster'
                indicies.clear();
                ListUtils.addRange(indicies, 0, N, 1);
            }
            else
                sub_results = clusters.cluster(toCluster, k_l, threadPool, (int[])null);
            
            //create partitioning
            //First, don't bother with distance computations for people we just clustered
            Arrays.fill(group, -1);
            Set<Integer> found_clusters = new HashSet<Integer>(k_l);
            for(int i = 0; i < sub_results.length; i++)
            {
                group[indicies.get(i)] = sub_results[i];
                found_clusters.add(sub_results[i]);
            }
            //find who everyone else belongs to
            final CountDownLatch latch = new CountDownLatch(threads_to_use);
            for(int id  = 0; id < threads_to_use; id++)
            {
                final int ID = id;
                threadPool.submit(new Runnable()
                {
                    @Override
                    public void run()
                    {
                        for(int i = ID; i < N; i+=threads_to_use)
                        {
                            if(group[i] >= 0)
                                continue;//you already got assigned above

                            List<Double> qi = null;
                            if(accelCache != null)
                            {
                                int multiplier = accelCache.size()/N;
                                qi = accelCache.subList(i*multiplier, i*multiplier+multiplier);
                            }
                            group[i] = clusters.findClosestCluster(vecs.get(i), qi);
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
                throw new FailedToFitException(ex);
            }
            //everyone has now been assigned to their closest cluster
            
            //build SVM model for each cluster
            for(int c : found_clusters)
            {
//                System.out.println("\tBuilding model for " + c);
                ClassificationDataSet V_c = new ClassificationDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories(), dataSet.getPredicting());
                DoubleList V_alphas = new DoubleList();
                IntList orig_index = new IntList();
                for (int i = 0; i < N; i++)
                {
                    if (group[i] != c)
                        continue;//well get to you later
                    //else, create dataset
                    V_c.addDataPoint(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i));
                    V_alphas.add(Math.abs(alphas[i]));
                    orig_index.add(i);
                }

                SVMnoBias svm = new SVMnoBias(getKernel());
                if(cache_size > 0)
                    svm.setCacheSize(V_alphas.size(), cache_size);
                else
                    svm.setCacheMode(CacheMode.NONE);
                
                //Train model
                if(l == l_max)//first round, no warm start
                    svm.trainC(V_c, threadPool);
                else//warm start
                {
                    svm.trainC(V_c, V_alphas.getBackingArray(), threadPool);
                }
                early_models.put(c, svm);
                
                //Update larger set of alphas
                for(int i = 0; i < orig_index.size(); i++)
                    this.alphas[orig_index.get(i)] = svm.alphas[i];
            }
        }
        
        if(l_early == 0)//fully solve the problem! Refinement step was done implicitly in above loop 
        {
            SVMnoBias svm = new SVMnoBias(getKernel());
            if (cache_size > 0)
                svm.setCacheSize(dataSet.getSampleSize(), cache_size );
            else
                svm.setCacheMode(CacheMode.NONE);
            svm.trainC(dataSet, Arrays.copyOf(this.alphas, this.alphas.length), threadPool);
            
            early_models.clear();
            early_models.put(0, svm);

            //Update all alphas
            for (int i = 0; i < N; i++)
                this.alphas[i] = svm.alphas[i];
        }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public DCSVM clone()
    {
        return new DCSVM(this);
    }
 
    /**
     * Sets the complexity parameter of SVM. The larger the C value the harder 
     * the margin SVM will attempt to find. Lower values of C allow for more 
     * misclassification errors. 
     * @param C the soft margin parameter
     */
    @Parameter.WarmParameter(prefLowToHigh = true)
    public void setC(double C)
    {
        if(C <= 0)
            throw new ArithmeticException("C must be a positive constant");
        this.C = C;
    }

    /**
     * Returns the soft margin complexity parameter of the SVM
     * @return the complexity parameter of the SVM
     */
    public double getC()
    {
        return C;
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
