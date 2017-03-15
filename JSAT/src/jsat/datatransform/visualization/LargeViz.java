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
package jsat.datatransform.visualization;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.distributions.Uniform;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * LargeViz is an algorithm for creating low dimensional embeddings for
 * visualization. It is meant to be faster and better quality than
 * {@link TSNE t-SNE} without requiring any parameter tuning to get good
 * results. LargeViz is related to t-SNE in how the neighbor graph is
 * constructed, and the {@link #setPerplexity(double) perplexity} parameter for
 * LargeViz has the same meaning and impact as the perplexity parameter in
 * t-SNE.<br>
 * <br>
 * NOTE: The origina LargeViz paper includes a faster scheme for approximately
 * constructing the nearest neighbor graph. This is not yet implemented, but has
 * no negative impact on the quality of the result.
 * <br>
 * See: Tang, J., Liu, J., Zhang, M., & Mei, Q. (2016). Visualizing Large-scale
 * and High-dimensional Data. In Proceedings of the 25th International
 * Conference on World Wide Web (pp. 287–297). Republic and Canton of Geneva,
 * Switzerland: International World Wide Web Conferences Steering Committee.
 * doi:10.1145/2872427.2883041
 * @author Edward Raff
 */
public class LargeViz implements VisualizationTransform
{
    private DistanceMetric dm_source = new EuclideanDistance();
    private DistanceMetric dm_embed = new EuclideanDistance();
    private double perplexity = 50;
    private int dt = 2;
    
    /**
     * This is the number of negative samples to take for each vertex <br>
     * "number of negative samples is set as 5"
     */
    private int M = 5;
    
    /**
     * "γ is set as 7 by default"
     */
    private double gamma = 7;
    
    /**
     * Sets the target perplexity of the gaussian used over each data point. The
     * perplexity can be thought of as a quasi desired number of nearest
     * neighbors to be considered, but is adapted based on the distribution of
     * the data. Increasing the perplexity can increase the amount of time it
     * takes to get an embedding. Using a value in the range of [5, 100] is
     * recommended.
     *
     * @param perplexity the quasi number of neighbors to consider for each data point
     */
    public void setPerplexity(double perplexity)
    {
        if(perplexity <= 0 || Double.isNaN(perplexity) || Double.isInfinite(perplexity))
            throw new IllegalArgumentException("perplexity must be positive, not " + perplexity);
        this.perplexity = perplexity;
    }

    /**
     * 
     * @return the target perplexity to use for each data point
     */
    public double getPerplexity()
    {
        return perplexity;
    }

    /**
     * Sets the distance metric to use for the original space. This will 
     * determine the target nearest neighbors  to keep close to each other in 
     * the embedding space
     * 
     * @param dm the distance metric to use
     */
    public void setDistanceMetricSource(DistanceMetric dm)
    {
        this.dm_source = dm;
    }
    
    /**
     * Sets the distance metric to use for the embedded space. This will 
     * determine the actual nearest neighbors as the occur in the embedded space. 
     * 
     * @param dm the distance metric to use
     */
    public void setDistanceMetricEmbedding(DistanceMetric dm)
    {
        this.dm_embed = dm;
    }

    /**
     * Sets the number of negative neighbor samples to obtain for each data
     * point. The default recommended value is 5.
     *
     * @param M the number of negative samples to use for each update
     */
    public void setNegativeSamples(int M)
    {
        if(M < 1)
            throw new IllegalArgumentException("Number of negative samples must be positive, not " + M);
        this.M = M;
    }

    /**
     * 
     * @return the number of negative samples to use for each update
     */
    public int getNegativeSamples()
    {
        return M;
    }

    /**
     * Gamma controls the negative weight assigned to negative edges in the
     * optimization problem. Large values will place a higher emphasis on
     * separating non-neighbors in the embedded space. The default recommend
     * value is 7.
     *
     * @param gamma the weight for negative edge samples
     */
    public void setGamma(double gamma)
    {
        if(Double.isInfinite(gamma) || Double.isNaN(gamma) || gamma <= 0)
            throw new IllegalArgumentException("Gamma must be positive, not " + gamma);
        this.gamma = gamma;
    }

    /**
     * 
     * @return the weight for negative edge samples
     */
    public double getGamma()
    {
        return gamma;
    }

    @Override
    public int getTargetDimension()
    {
        return dt;
    }

    @Override
    public boolean setTargetDimension(int target)
    {
        if(target < 2)
            return false;
        dt = target;
        return true;
    }

    @Override
    public <Type extends DataSet> Type transform(DataSet<Type> d)
    {
        return transform(d, new FakeExecutor());
    }

    @Override
    public <Type extends DataSet> Type transform(DataSet<Type> d, ExecutorService ex)
    {
        Random rand = RandomUtil.getRandom();
        final ThreadLocal<Random> local_rand = new ThreadLocal<Random>()
        {
            @Override
            protected Random initialValue()
            {
                return RandomUtil.getRandom();
            }
        };
        final int N = d.getSampleSize();
        //If perp set too big, the search size would be larger than the dataset size. So min to N
        /**
         * form sec 4.1: "we compute the sparse approximation by finding the
         * floor(3u) nearest neighbors of each of the N input objects (recall
         * that u is the perplexity of the conditional distributions)"
         */
        final int knn = (int) Math.min(Math.floor(3*perplexity), N-1);
        
        /**
         * P_ij does not change at this point, so lets compute these values only
         * once please! j index matches up to the value stored in nearMe. 
         * Would be W_ij in notation of LargeViz paper, but P_ij form TSNE paper
         */
        final double[][] nearMePij = new double[N][knn];
        
        /**
         * Each row is the set of 3*u indices returned by the NN search
         */
        final int[][] nearMe = new int[N][knn];
        
        TSNE.computeP(d, ex, rand, knn, nearMe, nearMePij, dm_source, perplexity);
        
        final double[][] nearMeSample = new double[N][knn];
        
        /**
         * Array of the sample weights used to perform the negative sampling. 
         * 
         * Initial value is out-degree defined in LINE paper, section 4.1.2. 
         */
        final double[] negSampleWeight = new double[N];
        
        double negSum = 0;
        for(int i = 0; i < N; i++)
        {
            double sum = DenseVector.toDenseVec(nearMePij[i]).sum();
            sum += nearMePij[i].length*Double.MIN_VALUE;
            negSampleWeight[i] = sum;
            
            nearMeSample[i][0] = nearMePij[i][0];
            for(int j = 1; j < knn; j++)//make cumulative
                nearMeSample[i][j] = Math.ulp(nearMePij[i][j]) + nearMePij[i][j] + nearMeSample[i][j-1];
            for(int j = 1; j < knn; j++)//normalize
                nearMeSample[i][j] /= sum;
            negSampleWeight[i] = Math.pow(negSampleWeight[i], 0.75);
            negSum += negSampleWeight[i];
            if(i > 0)
                negSampleWeight[i] += negSampleWeight[i-1];
        }
        //normalize to [0, 1] range
        for(int i = 0; i < N; i++)
            negSampleWeight[i]/= negSum;
        
        final List<Vec> embeded = new ArrayList<Vec>();
        Uniform initDistribution = new Uniform(-0.00005/dt, 0.00005/dt);
        for(int i = 0; i < N; i++)
            embeded.add(initDistribution.sampleVec(dt, rand));
        
        /**
         * Number of threads to use. Paper suggests asynch updates and just
         * ignore unsafe alters b/c diff should be minor. Adding some extra
         * logic so that we have at least a good handful of points per thread to
         * avoid excessive edits on small datasets.
         */
        final int threads_to_use = Math.max(Math.min(N/(200*M), SystemInfo.LogicalCores), 1);
        
        final CountDownLatch latch = new CountDownLatch(threads_to_use);
        
        /*
         * Objective is 
         *  w*(log(1/(1+g(x)^2)) + y log(1−1/(1+g(x)^2 )))
         * where g(x) is the euclidean distance adn G(x) is g(x)^2
        
         * d/x of ||x-y||   =   (x-y)/||x-y||
         * d/y of ||x-y||   =  -(x-y)/||x-y||
        
         * left hand side derivative of log(1/(1+g(x))) = 
         * = -(2 g(x) g'(x))/(g(x)^2+1)
         * = -(2 ||x-y|| (x-y)/||x-y||)/(||x-y||^2+1)
         * = -(2 (x-y))/(||x-y||^2+1)
         * for d/y 
         * = -(2 (y-x))/(||x-y||^2+1)
         *
         * Right hand side portion 
         * derivative of z* log(1-1/(1+g(x)^2))
         * =  (2 z g'(x))/(g(x) (g(x)^2+1))
         * =  (2 z (x-y))/(||x-y||^2 (||x-y||^2+1))
         * or for d/y
         * =  (2 z (y-x))/(||x-y||^2 (||x-y||^2+1))
        
         * NOTE: My derivative dosn't work. But adding 
         * an extra multiplication by ||x-y|| seems to fix everything? Want to 
         * come back and figure this out better. 
         */
        
        final double eta_0 = 1.0;
        final long iterations = 1000L*N;
        final ThreadLocal<Vec> local_grad_i = new ThreadLocal<Vec>()
        {
            @Override
            protected Vec initialValue()
            {
                return new DenseVector(dt);
            }
        };
        final ThreadLocal<Vec> local_grad_j = new ThreadLocal<Vec>()
        {
            @Override
            protected Vec initialValue()
            {
                return new DenseVector(dt);
            }
        };
        final ThreadLocal<Vec> local_grad_k = new ThreadLocal<Vec>()
        {
            @Override
            protected Vec initialValue()
            {
                return new DenseVector(dt);
            }
        };
        
        
        for(int id = 0; id < threads_to_use; id++)
        {
            ex.submit(new Runnable()
            {
                @Override
                public void run()
                {
                    Random l_rand = local_rand.get();
                    //b/c indicies are selected at random everyone can use same iterator order
                    //more important is to make sure the range length is the same so that 
                    //eta has the same range and effect in aggregate
                    for(long iteration = 0; iteration < iterations; iteration+=threads_to_use)
                    {
                        double eta = eta_0*(1-iteration/(double)iterations);
                        eta = Math.max(eta, 0.0001);

                        int i = l_rand.nextInt(N);
                        //sample neighbor weighted by distance
                        int j = Arrays.binarySearch(nearMeSample[i], l_rand.nextDouble());
                        if (j < 0)
                            j = -(j) - 1;
                        if(j >= knn)///oops. Can be hard to sample / happen with lots of near by near 0 dists
                        {
                            //lets fall back to picking someone at random
                            j = l_rand.nextInt(knn);
                        }
                        j = nearMe[i][j];

                        Vec y_i = embeded.get(i);
                        Vec y_j = embeded.get(j);
                        //right hand side update for the postive sample
                        final double dist_ij = dm_embed.dist(i, j, embeded, null);
                        final double dist_ij_sqrd = dist_ij*dist_ij;
                        if(dist_ij <= 0 )
                            continue;//how did that happen?

                        Vec grad_i = local_grad_i.get();
                        Vec grad_j = local_grad_j.get();
                        Vec grad_k = local_grad_k.get();
                        y_i.copyTo(grad_j);
                        grad_j.mutableSubtract(y_j);
                        grad_j.mutableMultiply(-2*dist_ij/(dist_ij_sqrd+1));


                        grad_j.copyTo(grad_i);

                        //negative sampling time
                        for(int k = 0; k < M; k++)
                        {
                            int jk = -1;
                            do
                            {
                                jk = Arrays.binarySearch(negSampleWeight, l_rand.nextDouble());
                                if (jk < 0)
                                    jk = -(jk) - 1;

                                if(jk  == i || jk == j)
                                    jk  = -1;

                                //code to reject neighbors for sampling if too close
                                //Not sure if this code helps or hurts... not mentioned in paper
                                for(int search = 0; search < nearMe[i].length; search++)
                                    if(nearMe[i][search] == jk && nearMeSample[i][search] < 0.98)
                                    {
                                        jk = -1;//too close to me!
                                        break;
                                    }
                            }
                            while(jk < 0);
                            //(2 z (y-x))/(||x-y||^2 (||x-y||^2+1))


                            Vec y_k = embeded.get(jk);
                            final double dist_ik = dm_embed.dist(i, jk, embeded, null);//dist(y_i, y_k);
                            final double dist_ik_sqrd = dist_ik*dist_ik;
                            if (dist_ik < 1e-12)
                                continue; 

                            y_i.copyTo(grad_k);
                            grad_k.mutableSubtract(y_k);
                            grad_k.mutableMultiply(2*gamma/(dist_ik*(dist_ik_sqrd+1)));

                            grad_i.mutableAdd(grad_k);

                            y_k.mutableSubtract(eta, grad_k);

                        }

                        y_i.mutableAdd( eta, grad_i);
                        y_j.mutableAdd(-eta, grad_j);
                    }
                    latch.countDown();
                }
            });
        }
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(LargeViz.class.getName()).log(Level.SEVERE, null, ex1);
        }
        
        DataSet<Type> toRet = d.shallowClone();
        
        final IdentityHashMap<DataPoint, Integer> indexMap = new IdentityHashMap<DataPoint, Integer>(N);
        for(int i = 0; i < N; i++)
            indexMap.put(d.getDataPoint(i), i);
        
        toRet.applyTransform(new DataTransform()
        {
            @Override
            public DataPoint transform(DataPoint dp)
            {
                int i = indexMap.get(dp);
                
                return new DataPoint(embeded.get(i), dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight());
            }

            @Override
            public void fit(DataSet data)
            {
                
            }

            @Override
            public DataTransform clone()
            {
                return this;
            }
        });
        
        return (Type) toRet;
    }
    
}