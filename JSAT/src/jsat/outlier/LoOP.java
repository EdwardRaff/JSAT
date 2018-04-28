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
import jsat.math.SpecialMath;
import jsat.utils.DoubleList;
import jsat.utils.IntList;
import jsat.utils.concurrent.ParallelUtils;

/**
 * This class implements the Local Outlier Probabilities (LoOP) algorithm for outlier detection. 
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class LoOP implements Outlier
{
    int minPnts;
    private double lambda = 3;
    private DistanceMetric distanceMetric;
    VectorCollection<Vec> vc = new DefaultVectorCollection<>();
    
    
    /**
     * Stores the "standard distance" of an index in X to its nearest neighbors
     */
    private double[] standard_distance;
    private double nPLOF;

    public LoOP()
    {
        this(20);
    }
    
    public LoOP(int minPnts)
    {
        this(minPnts, new EuclideanDistance());
    }
    
    public LoOP(int minPnts, DistanceMetric dm)
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

    public void setLambda(double lambda)
    {
        this.lambda = lambda;
    }

    public double getLambda()
    {
        return lambda;
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
        List<Vec> X = d.getDataVectors();
        vc.build(parallel, X, distanceMetric);
        
        int N = X.size();
        standard_distance = new double[N];
        List<List<Integer>> all_knn = new ArrayList<>();
        List<List<Double>> all_knn_dists = new ArrayList<>();
        
        vc.search(X, minPnts+1, all_knn, all_knn_dists, parallel);//+1 to avoid self distance
        
        ParallelUtils.run(parallel, N, (start, end)->
        {
            for(int i = start; i < end; i++)
                standard_distance[i] = Math.sqrt(all_knn_dists.get(i).stream()
                        .mapToDouble(z->z*z).sum()/minPnts+1e-6);
        });
        
        double[] plof_internal = new double[N];
        
        nPLOF = ParallelUtils.run(parallel, N, (start, end)->
        {
            double sqrdPLOF = 0;
            for(int i = start; i < end; i++)
            {
                double neighborSD = 0;

                for(int j_indx = 1; j_indx < minPnts+1; j_indx++)
                {
                    int neighbor = all_knn.get(i).get(j_indx);
                    neighborSD += standard_distance[neighbor];
                }
                
                plof_internal[i] = standard_distance[i]/(neighborSD/minPnts) - 1;
                sqrdPLOF += plof_internal[i]*plof_internal[i];
            }
            
            return sqrdPLOF;
        }, (a,b)->a+b);
        
        nPLOF = Math.sqrt(nPLOF/N);
        
        
    }
    
    @Override
    public double score(DataPoint x)
    {
        IntList knn = new IntList(minPnts);
        DoubleList dists = new DoubleList(minPnts);
        
        vc.search(x.getNumericalValues(), minPnts, knn, dists);
        
        
        
        double e_pdist = 0;
        double stndDist_q = 0;
        for(int i_indx = 0; i_indx < minPnts; i_indx++)
        {
            int neighbor = knn.get(i_indx);
            double dist = dists.get(i_indx);
            
            e_pdist += standard_distance[neighbor];
            
            stndDist_q += dist*dist;
        }
        
        //lrd_x now has the local reachability distance of the query x
        stndDist_q = Math.sqrt(stndDist_q/minPnts+1e-6);
        //normalize pdist of neighbors
        e_pdist /= minPnts;
        
        double plof_os = stndDist_q/e_pdist - 1;
        
        double loop = Math.max(0, SpecialMath.erf(plof_os/(lambda * nPLOF * Math.sqrt(2))));
        
        //loop, > 1/2 indicates outlier, <= 1/2 indicates inlier. 
        //to map to interface (negative = outlier), -1*(loop-1/2)
        
        return -(loop-0.5);
    }
    
}
