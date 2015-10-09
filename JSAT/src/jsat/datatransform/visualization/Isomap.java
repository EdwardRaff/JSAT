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
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.linear.*;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.DefaultVectorCollectionFactory;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;
import jsat.utils.FakeExecutor;
import jsat.utils.FibHeap;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class Isomap
{
    private DistanceMetric dm = new EuclideanDistance();
    private VectorCollectionFactory<VecPaired<Vec, Integer>> vcf = new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>();
    private int searchNeighbors = 15;
    private MDS mds = new MDS();
    private boolean c_isomap = false;

    public void setNeighbors(int searchNeighbors)
    {
        this.searchNeighbors = searchNeighbors;
    }

    public int getNeighbors()
    {
        return searchNeighbors;
    }
    
    public void setCIsomap(boolean c_isomap)
    {
        this.c_isomap = c_isomap;
    }

    public boolean isCIsomap()
    {
        return c_isomap;
    }
    
    
            
    public <Type extends DataSet> Type transform(DataSet<Type> d)
    {
        return transform(d, new FakeExecutor());
    }
    
    public <Type extends DataSet> Type transform(DataSet<Type> d, ExecutorService ex)
    {
        int N = d.getSampleSize();
        Matrix delta = new DenseMatrix(N, N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (i == j)
                    delta.set(i, j, 0);
                else
                    delta.set(i, j, Double.MAX_VALUE);

        
        List<VecPaired<Vec, Integer>> vecs = new ArrayList<VecPaired<Vec, Integer>>(N);
        for(int i = 0; i < N; i++)
            vecs.add(new VecPaired<Vec, Integer>(d.getDataPoint(i).getNumericalValues(), i));
        VectorCollection<VecPaired<Vec, Integer>> vc = vcf.getVectorCollection(vecs, dm, ex);
        List<Double> cache = dm.getAccelerationCache(vecs, ex);
                
        int knn = searchNeighbors+1;//+1 b/c we are closest to ourselves
        
        //bleh, ugly generics...
        List<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>> neighborGraph = new ArrayList<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>>();

        double[] avgNeighborDist = new double[N];
        
        //do knn search and store results so we can do distances
        for (int i = 0; i < N; i++)
        {
            List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors = vc.search(vecs.get(i).getVector(), knn);
            neighborGraph.add(neighbors);
            //Compute stats that may be used for c-isomap version
            for (int z = 1; z < neighbors.size(); z++)
            {
                VecPaired<VecPaired<Vec, Integer>, Double> neighbor = neighbors.get(z);
                int j = neighbor.getVector().getPair();
                double dist = neighbor.getPair();
                avgNeighborDist[i] += dist;
            }
            avgNeighborDist[i] /= (neighbors.size()-1);
        }
        
        if(c_isomap)
        {
            int i = 0;
            for(List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors : neighborGraph)
            {
                for(VecPaired<VecPaired<Vec, Integer>, Double> neighbor : neighbors)
                    neighbor.setPair(neighbor.getPair()/Math.sqrt(avgNeighborDist[neighbor.getVector().getPair()]+avgNeighborDist[i]+1e-6));
                i++;
            }
        }
        
        for(int k = 0; k < N; k++)
        {
            double[] tmp_dist = dijkstra(neighborGraph, k);
            for (int i = 0; i < N; i++)
            {
                tmp_dist[i] = Math.min(tmp_dist[i], delta.get(i, k));
                delta.set(i, k, tmp_dist[i]);
                delta.set(k, i, tmp_dist[i]);
            }
        }
        
        //lets check for any disjoint groupings, replace them with something reasonable
        //we will use the largest obtainable distance to be an offset for our infinity distances
        double largest_natural_dist = 0;
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                if(delta.get(i, j) < Double.MAX_VALUE)
                    largest_natural_dist = Math.max(largest_natural_dist, delta.get(i, j));
        
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
            {
                double d_ij = delta.get(i, j);
                if (d_ij >= Double.MAX_VALUE)//replace with the normal distance + 1 order of magnitude? 
                {
                    d_ij = 10*dm.dist(i, j, vecs, cache)+1.5*largest_natural_dist;
                    delta.set(i, j, d_ij);
                    delta.set(j, i, d_ij);
                }
            }
        
        
        SimpleDataSet emedded = mds.transform(delta, ex);
        
        DataSet<Type> transformed = d.shallowClone();
        transformed.replaceNumericFeatures(emedded.getDataVectors());
        return (Type) transformed;
    }

    private double[] dijkstra(List<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>> neighborGraph, int sourceIndex)
    {
        //TODO generalize and move this out into some other class as a static method 
        final int N = neighborGraph.size();
        double[] dist = new double[N];
        Arrays.fill(dist, Double.POSITIVE_INFINITY);
        dist[sourceIndex] = 0;
        List<FibHeap.FibNode<Integer>> nodes = new ArrayList<FibHeap.FibNode<Integer>>(N);

        FibHeap<Integer> Q = new FibHeap<Integer>();
        for (int i = 0; i < neighborGraph.size(); i++)
            nodes.add(Q.insert(i, dist[i]));

        while (Q.size() > 0)
        {
            FibHeap.FibNode<Integer> u = Q.removeMin();
            int u_indx = u.getValue();

            List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors = neighborGraph.get(u_indx);
            for (int z = 1; z < neighbors.size(); z++)
            {
                VecPaired<VecPaired<Vec, Integer>, Double> neighbor = neighbors.get(z);
                int j = neighbor.getVector().getPair();
                double u_j_dist = neighbor.getPair();
                double alt = dist[u_indx] + u_j_dist;

                if (alt < dist[j])
                {
                    dist[j] = alt;
                    //prev[j] ← u
                    Q.decreaseKey(nodes.get(j), alt);
                }
            }
        }

        return dist;
    }
}
