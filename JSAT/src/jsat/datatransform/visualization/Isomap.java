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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
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
import jsat.utils.SystemInfo;

/**
 * Isomap is an extension of {@link MDS}. It uses a geodesic distance made from
 * a nearest neighbor search of all the points in the data set. This
 * implementation also includes the extension
 * {@link #setCIsomap(boolean) C-Isomap}, which further weights distances by
 * density.<br>
 * <br>
 * Note, that Isomap normally will fail on some datasets when two or more
 * regions can not be connected in the induced neighbor graph. While increasing
 * the number of neighbors considered will eventually resolve this problem, the
 * separated groups may be desirable in practice. This implementation includes a
 * non-standard addition that will forcibly connect such isolated regions with
 * very large values, hoping to preserve the farther distances in the given
 * dataset while maintaining local structure.<br>
 * <br>
 *
 * See:<br>
 * <ul>
 * <li>Tenenbaum, J. B., Silva, V. De, & Langford, J. C. (2000). <i>A Global
 * Geometric Framework for Nonlinear Dimensionality Reduction</i>. Science, 290,
 * 2319–2323. doi:10.1126/science.290.5500.2319</li>
 * <li>De Silva, V., & Tenenbaum, J. B. (2003). <i>Global Versus Local Methods
 * in Nonlinear Dimensionality Reduction</i>. In Advances in Neural Information
 * Processing Systems 15 (pp. 705–712). MIT Press. Retrieved from
 * <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.3407&amp;rep=rep1&amp;type=pdf">here</a></li>
 * </ul>
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class Isomap implements VisualizationTransform
{
    private DistanceMetric dm = new EuclideanDistance();
    private VectorCollectionFactory<VecPaired<Vec, Integer>> vcf = new DefaultVectorCollectionFactory<VecPaired<Vec, Integer>>();
    private int searchNeighbors = 15;
    private MDS mds = new MDS();
    private boolean c_isomap = false;

    /**
     *
     */
    public Isomap()
    {
        this(15, false);
    }

    /**
     *
     * @param searchNeighbors the number of nearest neighbors to consider
     */
    public Isomap(int searchNeighbors)
    {
        this(searchNeighbors, false);
    }

    /**
     * 
     * @param searchNeighbors the number of nearest neighbors to consider
     * @param c_isomap {@code true} to use the C-Isomap extension, {@code false}
     * for normal Isomap.
     */
    public Isomap(int searchNeighbors, boolean c_isomap)
    {
        setNeighbors(searchNeighbors);
        setCIsomap(c_isomap);
    }
    
    /**
     * Set the number of neighbors to consider for the initial graph in Isomap
     * @param searchNeighbors the number of nearest neighbors to consider
     */
    public void setNeighbors(int searchNeighbors)
    {
        if(searchNeighbors < 2)
            throw new IllegalArgumentException("number of neighbors considered must be at least 2, not " + searchNeighbors);
        this.searchNeighbors = searchNeighbors;
    }

    /**
     * 
     * @return the number of neighbors used when creating the initial graph
     */
    public int getNeighbors()
    {
        return searchNeighbors;
    }
    
    /**
     * Controls whether the C-Isomap extension is used. If set true, the initial
     * distances will also be scaled based on the density of the region between
     * the points. If false, normal Isomap will be used.
     *
     * @param c_isomap {@code true} to use the C-Isomap extension, {@code false}
     * for normal Isomap.
     */
    public void setCIsomap(boolean c_isomap)
    {
        this.c_isomap = c_isomap;
    }

    /**
     *
     * @return {@code true} if the C-Isomap extension is in use, {@code false}
     * for normal Isomap.
     */
    public boolean isCIsomap()
    {
        return c_isomap;
    }
            
    @Override
    public <Type extends DataSet> Type transform(DataSet<Type> d)
    {
        return transform(d, new FakeExecutor());
    }
    
    @Override
    public <Type extends DataSet> Type transform(DataSet<Type> d, ExecutorService ex)
    {
        final int N = d.getSampleSize();
        final Matrix delta = new DenseMatrix(N, N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (i == j)
                    delta.set(i, j, 0);
                else
                    delta.set(i, j, Double.MAX_VALUE);

        
        final List<VecPaired<Vec, Integer>> vecs = new ArrayList<VecPaired<Vec, Integer>>(N);
        for(int i = 0; i < N; i++)
            vecs.add(new VecPaired<Vec, Integer>(d.getDataPoint(i).getNumericalValues(), i));
        final VectorCollection<VecPaired<Vec, Integer>> vc = vcf.getVectorCollection(vecs, dm, ex);
        final List<Double> cache = dm.getAccelerationCache(vecs, ex);
                
        final int knn = searchNeighbors+1;//+1 b/c we are closest to ourselves
        
        //bleh, ugly generics...
        final List<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>> neighborGraph = new ArrayList<List<? extends VecPaired<VecPaired<Vec, Integer>, Double>>>();
        for (int i = 0; i < N; i++)
            neighborGraph.add(null);
            
        final double[] avgNeighborDist = new double[N];
        
        //do knn search and store results so we can do distances
        final CountDownLatch latch1 = new CountDownLatch(SystemInfo.LogicalCores);
        for(int id = 0; id < SystemInfo.LogicalCores; id++)
        {
            final int ID = id;
            ex.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    for (int i = ID; i < N; i+=SystemInfo.LogicalCores)
                    {
                        List<? extends VecPaired<VecPaired<Vec, Integer>, Double>> neighbors = vc.search(vecs.get(i).getVector(), knn);
                        neighborGraph.set(i, neighbors);
                        //Compute stats that may be used for c-isomap version
                        for (int z = 1; z < neighbors.size(); z++)
                        {
                            VecPaired<VecPaired<Vec, Integer>, Double> neighbor = neighbors.get(z);
                            double dist = neighbor.getPair();
                            avgNeighborDist[i] += dist;
                        }
                        avgNeighborDist[i] /= (neighbors.size()-1);
                    }
                    latch1.countDown();
                }
            });
        }
        
        try
        {
            latch1.await();
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(Isomap.class.getName()).log(Level.SEVERE, null, ex1);
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
        
        final CountDownLatch latch2 = new CountDownLatch(SystemInfo.LogicalCores);
        for(int id = 0; id < SystemInfo.LogicalCores; id++)
        {
            final int ID = id;
            ex.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    for (int k = ID; k < N; k += SystemInfo.LogicalCores)
                    {
                        double[] tmp_dist = dijkstra(neighborGraph, k);
                        for (int i = 0; i < N; i++)
                        {
                            tmp_dist[i] = Math.min(tmp_dist[i], delta.get(k, i));
                            delta.set(i, k, tmp_dist[i]);
                            delta.set(k, i, tmp_dist[i]);
                        }
                    }
                    latch2.countDown();
                }

            });
        }
        
        try
        {
            latch2.await();
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(Isomap.class.getName()).log(Level.SEVERE, null, ex1);
        }
        
        
        //lets check for any disjoint groupings, replace them with something reasonable
        //we will use the largest obtainable distance to be an offset for our infinity distances
        double largest_natural_dist_tmp = 0;
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                if(delta.get(i, j) < Double.MAX_VALUE)
                    largest_natural_dist_tmp = Math.max(largest_natural_dist_tmp, delta.get(i, j));
        final double largest_natural_dist = largest_natural_dist_tmp;
        
        final CountDownLatch latch3 = new CountDownLatch(SystemInfo.LogicalCores);
        for(int id = 0; id < SystemInfo.LogicalCores; id++)
        {
            final int ID = id;
            ex.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    for (int i = ID; i < N; i += SystemInfo.LogicalCores)
                    {
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
                        latch3.countDown();
                    }
                }
            });
        }
        
        try
        {
            latch3.await();
        }
        catch (InterruptedException ex1)
        {
            Logger.getLogger(Isomap.class.getName()).log(Level.SEVERE, null, ex1);
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
        for (int i = 0; i < N; i++)
            nodes.add(null);
        nodes.set(sourceIndex, Q.insert(sourceIndex, dist[sourceIndex]));

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
                    if(nodes.get(j) == null)
                        nodes.set(j, Q.insert(j, alt));
                    else
                        Q.decreaseKey(nodes.get(j), alt);
                }
            }
        }

        return dist;
    }
    
    @Override
    public int getTargetDimension()
    {
        return mds.getTargetDimension();
    }

    @Override
    public boolean setTargetDimension(int target)
    {
        return mds.setTargetDimension(target);
    }
}
