/*
 * Copyright (C) 2018 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.outlier;

import java.util.ArrayList;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.DefaultVectorCollection;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.utils.DoubleList;
import jsat.utils.IntList;
import jsat.utils.concurrent.ParallelUtils;

/**
 * This class implements the Local Outlier Factor (LOF) algorithm for outlier detection. 
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class LOF implements Outlier
{
    int minPnts;
    private DistanceMetric distanceMetric;
    VectorCollection<Vec> vc = new DefaultVectorCollection<>();
    
    
    /**
     * the points in this collection
     */
    private List<Vec> X;
    /**
     * Stores the distance of an index in X to it's k'th (minPnts) nearest neighbor. 
     */
    private double[] k_distance;
    private double[] lrd_internal;

    public LOF()
    {
        this(15);
    }
    
    public LOF(int minPnts)
    {
        this(minPnts, new EuclideanDistance());
    }
    
    public LOF(int minPnts, DistanceMetric dm)
    {
        setMinPnts(minPnts);
        setDistanceMetric(dm);
    }

    public void setMinPnts(int minPnts)
    {
        this.minPnts = minPnts;
    }

    public int getMinPnts()
    {
        return minPnts;
    }

    public void setDistanceMetric(DistanceMetric distanceMetric)
    {
        this.distanceMetric = distanceMetric;
    }

    public DistanceMetric getDistanceMetric()
    {
        return distanceMetric;
    }
    
    
    

    @Override
    public void fit(DataSet d, boolean parallel)
    {
        X = d.getDataVectors();
        vc.build(parallel, X, distanceMetric);
        
        int N = X.size();
        k_distance = new double[N];
        List<List<Integer>> all_knn = new ArrayList<>();
        List<List<Double>> all_knn_dists = new ArrayList<>();
        
        vc.search(X, minPnts+1, all_knn, all_knn_dists, parallel);//+1 to avoid self distance
        
        ParallelUtils.run(parallel, N, (start, end)->
        {
            for(int i = start; i < end; i++)
                k_distance[i] = all_knn_dists.get(i).get(minPnts);
        });
        
        lrd_internal = new double[N];
        ParallelUtils.run(parallel, N, (start, end)->
        {
            for(int i = start; i < end; i++)
            {
                double reachSum = 0;

                for(int j_indx = 1; j_indx < minPnts+1; j_indx++)
                {
                    int neighbor = all_knn.get(i).get(j_indx);
                    double dist = all_knn_dists.get(i).get(j_indx);
                    reachSum += Math.max(k_distance[neighbor], dist);
                }
                
                //lrd_internal[i] = 1.0/(reachSum/minPnts);
                lrd_internal[i] = minPnts/reachSum;
            }
        });
        
        
    }
    
    double lrd(Vec a, List<Double> qi)
    {
        return 0;
    }

    @Override
    public double score(DataPoint x)
    {
        IntList knn = new IntList(minPnts);
        DoubleList dists = new DoubleList(minPnts);
        
        vc.search(x.getNumericalValues(), minPnts, knn, dists);
        
        double lof = 0;
        double lrd_x = 0;
        for(int i_indx = 0; i_indx < minPnts; i_indx++)
        {
            int neighbor = knn.get(i_indx);
            double dist = dists.get(i_indx);
            double reach_dist = Math.max(k_distance[neighbor], dist);
            
            lof += lrd_internal[neighbor];
            
            lrd_x += reach_dist;
        }
        
        //lrd_x now has the local reachability distance of the query x
        lrd_x = minPnts/lrd_x;
        //now compuate final LOF score
        lof /= minPnts * lrd_x;
        
        //lof, > 1 indicates outlier, <= 1 indicates inlier. 
        //to map to interface (negative = outlier), -1*(lof-1)
        //use -1.25 b/c the boarder around 1 is kinda noisy
        return -(lof-1.25);
    }
    
}
