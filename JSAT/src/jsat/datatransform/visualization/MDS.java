/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.OnLineStatistics;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;
import jsat.utils.concurrent.AtomicDouble;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * Multidimensional scaling is an algorithm for finding low dimensional
 * embeddings of arbitrary distance matrices. MDS will attempt to find an
 * embedding that maintains the same pair-wise distances between all items in
 * the distance matrix. MDS is a non-convex problem, so different runs can
 * produce different results. <br>
 * <br>
 * MDS can be used on arbitrary dissimilarity matrices by calling {@link #transform(jsat.linear.Matrix, java.util.concurrent.ExecutorService)
 * }.
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class MDS implements VisualizationTransform
{
    private static DistanceMetric embedMetric = new EuclideanDistance();
    private DistanceMetric dm = new EuclideanDistance();
    private double tolerance = 1e-3;
    private int maxIterations = 300;
    private int targetSize = 2;

    /**
     * Sets the tolerance parameter for determining convergence. 
     * @param tolerance the tolerance for declaring convergence
     */
    public void setTolerance(double tolerance)
    {
        if(tolerance < 0 || Double.isInfinite(tolerance) || Double.isNaN(tolerance))
            throw new IllegalArgumentException("tolerance must be a non-negative value, not " + tolerance);
        this.tolerance = tolerance;
    }

    /**
     * 
     * @return the tolerance parameter 
     */
    public double getTolerance()
    {
        return tolerance;
    }

    /**
     * Sets the distance metric to use when creating the initial dissimilarity
     * matrix of a new dataset. By default the {@link EuclideanDistance Euclidean
     * } distance is used, but any distance may be substituted.The chosen
     * distance need not be a valid metric, its only requirement is symmetry.
     *
     * @param embedMetric the distance metric to use when creating the
     * dissimilarity matrix.
     */
    public void setEmbeddingMetric(DistanceMetric embedMetric)
    {
        this.embedMetric = embedMetric;
    }
    
    /**
     * 
     * @return the distance metric used when creating a dissimilarity matrix
     */
    public DistanceMetric getEmbeddingMetric()
    {
        return embedMetric;
    }
    
    @Override
    public <Type extends DataSet> Type transform(final DataSet<Type> d, boolean parallel)
    {
        final List<Vec> orig_vecs = d.getDataVectors();
        final List<Double> orig_distCache = dm.getAccelerationCache(orig_vecs, parallel);
        final int N = orig_vecs.size();
        
        //Delta is the true disimilarity matrix
        final Matrix delta = new DenseMatrix(N, N);

        
        OnLineStatistics avg = ParallelUtils.run(parallel, N, (i)->
        {
            OnLineStatistics local_avg = new OnLineStatistics();
            for(int j = i+1; j < d.size(); j++)
            {
                double dist = dm.dist(i, j, orig_vecs, orig_distCache);
                local_avg.add(dist);
                delta.set(i, j, dist);
                delta.set(j, i, dist);
            } 
            return local_avg;
        }, (a,b)->OnLineStatistics.add(a, b));
        

        SimpleDataSet embeded = transform(delta, parallel);

        //place the solution in a dataset of the correct type
        DataSet<Type> transformed = d.shallowClone();
        transformed.replaceNumericFeatures(embeded.getDataVectors());
        return (Type) transformed;
    }

    public SimpleDataSet transform(Matrix delta)
    {
        return transform(delta, false);
    }
    
    public SimpleDataSet transform(final Matrix delta, boolean parallel)
    {
        final int N = delta.rows();
        Random rand = RandomUtil.getRandom();
        
        final Matrix X = new DenseMatrix(N, targetSize);
        final List<Vec> X_views = new ArrayList<>();
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < targetSize; j++)
                X.set(i, j, rand.nextDouble());
            X_views.add(X.getRowView(i));
        }
        final List<Double> X_rowCache = embedMetric.getAccelerationCache(X_views, parallel);
        
        //TODO, special case solution when all weights are the same, want to add general case as well
        Matrix V_inv = new DenseMatrix(N, N);
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    V_inv.set(i, j, (1.0-1.0/N)/N);
                else
                    V_inv.set(i, j, (0.0-1.0/N)/N);
            }
        
        double stressChange = Double.POSITIVE_INFINITY;
        double oldStress = stress(X_views, X_rowCache, delta, parallel);
        
        //the gutman transform matrix
        final Matrix B = new DenseMatrix(N, N);
        final Matrix X_new = new DenseMatrix(X.rows(), X.cols());
                
        for(int iter = 0; iter < maxIterations && stressChange > tolerance; iter++ )
        {
            ParallelUtils.run(parallel, B.rows(), (i)->
            {
                for (int j = i + 1; j < B.rows(); j++)
                {
                    double d_ij = embedMetric.dist(i, j, X_views, X_rowCache);

                    if(d_ij > 1e-5)//avoid creating silly huge values
                    {
                        double b_ij = -delta.get(i, j)/d_ij;//-w_ij if we support weights in the future
                        B.set(i, j, b_ij);
                        B.set(j, i, b_ij);
                    }
                    else
                    {
                        B.set(i, j, 0);
                        B.set(j, i, 0);
                    }
                }
            });
            
            X_new.zeroOut();
            
            //set the diagonal values
            for(int i = 0; i < B.rows(); i++)
            {   
                B.set(i, i, 0);
                for (int k = 0; k < B.cols(); k++)
                    if (k != i)
                        B.increment(i, i, -B.get(i, k));
            }
            
//            Matrix X_new = V_inv.multiply(B, ex).multiply(X, ex);
            
            B.multiply(X, X_new, ParallelUtils.CACHED_THREAD_POOL);
            X_new.mutableMultiply(1.0/N);
            
            X_new.copyTo(X);
            X_rowCache.clear();
            X_rowCache.addAll(embedMetric.getAccelerationCache(X_views, parallel));
            
            double newStress = stress(X_views, X_rowCache, delta, parallel);
            stressChange = Math.abs(oldStress-newStress);
            oldStress = newStress;
        }
        
        SimpleDataSet sds = new SimpleDataSet(targetSize, new CategoricalData[0]);
        for(Vec v : X_views)
            sds.add(new DataPoint(v));
        return sds;
    }
    
    private static double stress(final List<Vec> X_views, final List<Double> X_rowCache, final Matrix delta, boolean parallel)
    {
        return ParallelUtils.run(parallel, delta.rows(), (i)->
        {
            double localStress = 0;
            for(int j = i+1; j < delta.rows(); j++)
            {
                double tmp = embedMetric.dist(i, j, X_views, X_rowCache)-delta.get(i, j);
                localStress += tmp*tmp;
            }
            return localStress;
        }, (a,b)->a+b);
    }

    @Override
    public int getTargetDimension()
    {
        return targetSize;
    }

    @Override
    public boolean setTargetDimension(int target)
    {
        if(target < 1)
            return false;
        this.targetSize = target;
        return true;
    }
}
