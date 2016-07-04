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
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.distributions.Uniform;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.FakeExecutor;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class LargeViz implements VisualizationTransform
{
    private DistanceMetric dm = new EuclideanDistance();
    private double perplexity = 50;
    int dt = 2;
    
    /**
     * This is the number of negative samples to take for each vertex <br>
     * "number of negative samples is set as 5"
     */
    int M = 5;
    
    /**
     * "γ is set as 7 by default"
     */
    double gamma = 7;


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
        Random rand = new XORWOW();
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
        
        TSNE.computeP(d, ex, rand, knn, nearMe, nearMePij, dm, perplexity);
        
        final double[][] nearMeSample = new double[N][knn];
        
        /**
         * Array of the sample weights used to perform the negative sampling. 
         * 
         * Initial value is out-degree defined in LINE paper, section 4.1.2. 
         */
        double[] negSampleWeight = new double[N];
        
        double negSum = 0;
        for(int i = 0; i < N; i++)
        {
            double sum = negSampleWeight[i] = DenseVector.toDenseVec(nearMePij[i]).sum();
            nearMeSample[i][0] = nearMePij[i][0];
            for(int j = 1; j < knn; j++)//make cumulative
                nearMeSample[i][j] = nearMePij[i][j] + nearMeSample[i][j-1];
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
        
        double eta_0 = 1.0;
        long iterations = 1000L*N;
        Vec grad_i = new DenseVector(dt);
        Vec grad_j = new DenseVector(dt);
        Vec grad_k = new DenseVector(dt);
        
        for(long iteration = 0; iteration < iterations; iteration++)
        {
            double eta = eta_0*(1-iteration/(double)iterations);
            
            int i = rand.nextInt(N);
            //sample neighbor weighted by distance
            int j = Arrays.binarySearch(nearMeSample[i], rand.nextDouble());
            if (j < 0)
                j = -(j) - 1;
            j = nearMe[i][j];
            
            Vec y_i = embeded.get(i);
            Vec y_j = embeded.get(j);
            //right hand side update for the postive sample
            final double dist_ij = dm.dist(i, j, embeded, null);
            final double dist_ij_sqrd = dist_ij*dist_ij;
            if(dist_ij <= 0 )
                continue;//how did that happen?

            y_i.copyTo(grad_j);
            grad_j.mutableSubtract(y_j);
            grad_j.mutableMultiply(-2*dist_ij/(dist_ij_sqrd+1));


            grad_i.zeroOut();

            //negative sampling time
            for(int k = 0; k < M; k++)
            {
                int jk = -1;
                do
                {
                    jk = Arrays.binarySearch(negSampleWeight, rand.nextDouble());
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
                final double dist_ik = dm.dist(i, jk, embeded, null);//dist(y_i, y_k);
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
            public DataTransform clone()
            {
                return this;
            }
        });
        
        return (Type) toRet;
    }
    
}
