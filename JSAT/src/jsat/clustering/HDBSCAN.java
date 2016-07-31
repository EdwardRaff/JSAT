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
package jsat.clustering;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.*;
import jsat.utils.FibHeap;
import static java.lang.Math.max;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.*;

/**
 * HDBSCAN is a density based clustering algorithm that is an improvement over
 * {@link DBSCAN}. Unlike its predecessor, HDBSCAN works with variable density
 * datasets and does not need a search radius to be specified. The original
 * paper presents HDBSCAN with two parameters
 * {@link #setMinPoints(int) m<sub>pts</sub>} and
 * {@link #setMinClusterSize(int) m<sub>clSize</sub>}, but recomends that they
 * can be set to the same value and effectively behave as if only one parameter
 * exists. This implementation allows for setting both independtly, but the
 * single parameter constructors will use the same value for both parameters.
 * <br>
 * NOTE: The current implementation has O(N<sup>2</sup>) run time, though
 * this may be improved in the future with more advanced algorithms.<br>
 * <br>
 * See: Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-Based
 * Clustering Based on Hierarchical Density Estimates. In J. Pei, V. Tseng, L.
 * Cao, H. Motoda, & G. Xu (Eds.), Advances in Knowledge Discovery and Data
 * Mining (pp. 160–172). Springer Berlin Heidelberg.
 * doi:10.1007/978-3-642-37456-2_14
 * @author Edward Raff
 */
public class HDBSCAN extends ClustererBase implements Parameterized
{
    private DistanceMetric dm;
    /**
     * minimium number of points
     */
    private int m_pts;
    private int m_clSize;
    private VectorCollectionFactory<Vec> vcf;
    
    /**
     * Creates a new HDBSCAN object using a threshold of 15 points to form a
     * cluster.
     */
    public HDBSCAN()
    {
        this(15);
    }
    
    /**
     * Creates a new HDBSCAN using the simplified form, where the only parameter
     * is a single value.
     *
     * @param m_pts the minimum number of points needed to form a cluster and
     * the number of neighbors to consider
     */
    public HDBSCAN(int m_pts)
    {
        this(new EuclideanDistance(), m_pts);
    }

    /**
     * Creates a new HDBSCAN using the simplified form, where the only parameter
     * is a single value.
     *
     * @param dm the distance metric to use for finding nearest neighbors
     * @param m_pts the minimum number of points needed to form a cluster and
     * the number of neighbors to consider
     */
    public HDBSCAN(DistanceMetric dm, int m_pts)
    {
        this(dm, m_pts, m_pts, new VPTreeMV.VPTreeMVFactory<Vec>());
    }

    /**
     * Creates a new HDBSCAN using the simplified form, where the only parameter
     * is a single value.
     *
     * @param dm the distance metric to use for finding nearest neighbors
     * @param m_pts the minimum number of points needed to form a cluster and
     * the number of neighbors to consider
     * @param vcf the vector collection to use for accelerating nearest neighbor
     * queries
     */
    public HDBSCAN(DistanceMetric dm, int m_pts, VectorCollectionFactory<Vec> vcf)
    {
        this(dm, m_pts, m_pts, vcf);
    }

    /**
     * Creates a new HDBSCAN using the full specification of the algorithm,
     * where two parameters may be altered. In the simplified version both
     * parameters always have the same value.
     *
     * @param dm the distance metric to use for finding nearest neighbors
     * @param m_pts the number of neighbors to consider, acts as a smoothing
     * over the density estimate
     * @param m_clSize the minimum number of data points needed to form a
     * cluster
     * @param vcf the vector collection to use for accelerating nearest neighbor
     * queries
     */
    public HDBSCAN(DistanceMetric dm, int m_pts, int m_clSize, VectorCollectionFactory<Vec> vcf)
    {
        this.dm = dm;
        this.m_pts = m_pts;
        this.m_clSize = m_clSize;
        this.vcf = vcf;
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public HDBSCAN(HDBSCAN toCopy)
    {
        this.dm = dm.clone();
        this.m_pts = toCopy.m_pts;
        this.m_clSize = toCopy.m_clSize;
        this.vcf = toCopy.vcf.clone();
    }

    /**
     * 
     * @param m_clSize the minimum number of data points needed to form a
     * cluster
     */
    public void setMinClusterSize(int m_clSize)
    {
        this.m_clSize = m_clSize;
    }

    /**
     * 
     * @return the minimum number of data points needed to form a
     * cluster
     */
    public int getMinClusterSize()
    {
        return m_clSize;
    }

    /**
     * Sets the distance metric to use for determining closeness between data points
     * @param dm the distance metric to determine nearest neighbors with
     */
    public void setDistanceMetrics(DistanceMetric dm)
    {
        this.dm = dm;
    }

    /**
     * 
     * @return the distance metric to determine nearest neighbors with
     */
    public DistanceMetric getDistanceMetrics()
    {
        return dm;
    }

    /**
     * 
     * @param m_pts the number of neighbors to consider, acts as a smoothing
     * over the density estimate
     */
    public void setMinPoints(int m_pts)
    {
        this.m_pts = m_pts;
    }

    /**
     * 
     * @return the number of neighbors to consider, acts as a smoothing
     * over the density estimate
     */
    public int getMinPoints()
    {
        return m_pts;
    }
    

    @Override
    public HDBSCAN clone()
    {
        return new HDBSCAN(this);
    }

    @Override
    public int[] cluster(DataSet dataSet, int[] designations)
    {
        return cluster(dataSet, new FakeExecutor(), designations);
    }

    @Override
    public int[] cluster(DataSet dataSet, ExecutorService threadpool, int[] designations)
    {
        if(designations == null)
            designations = new int[dataSet.getSampleSize()];
        
        @SuppressWarnings("unchecked")
        final List<Vec> X = dataSet.getDataVectors();
        final int N = X.size();
        List<Double> cache = dm.getAccelerationCache(X, threadpool);
        VectorCollection<Vec> X_vc = vcf.getVectorCollection(X, dm, threadpool);
        //1. Compute the core distance w.r.t. m_pts for all data objects in X.
        /*
         * (Core Distance): The core distance of an object x_p ∈ X w.r.t. m_pts, 
         * d_core(x_p), is the distance from x_p to its m_pts-nearest neighbor (incl. x_p)
         */
        List<List<? extends VecPaired<Vec, Double>>> allNearestNeighbors = VectorCollectionUtils.allNearestNeighbors(X_vc, X, m_pts, threadpool);
        double[] core = new double[N];
        for(int i = 0; i < N; i++)
            core[i] = allNearestNeighbors.get(i).get(m_pts-1).getPair();
        
        //2. Compute an MST of G_{m_pts}, the Mutual Reachability Graph.
        
        //prims algorithm from Wikipedia
        double[] C = new double[N];
        Arrays.fill(C, Double.MAX_VALUE);
        int[] E = new int[N];
        Arrays.fill(E, -1);//-1 "a special flag value indicating that there is no edge connecting v to earlier vertices"
        
        FibHeap<Integer> Q = new FibHeap<Integer>();
        List<FibHeap.FibNode<Integer>> q_nodes = new ArrayList<FibHeap.FibNode<Integer>>(N);
        for(int i = 0; i < N; i++)
            q_nodes.add(Q.insert(i, C[i]));
        Set<Integer> F = new HashSet<Integer>();

        /**
         * First 2 indicate the edges, 3d value is the weight
         */
        List<Tuple3<Integer, Integer, Double>> mst_edges = new ArrayList<Tuple3<Integer, Integer, Double>>(N*2);

        while(Q.size() > 0)
        {
            //a. Find and remove a vertex v from Q having the minimum possible value of C[v]
            FibHeap.FibNode<Integer> node = Q.removeMin();
            int v = node.getValue();
            q_nodes.set(v, null);
            //b. Add v to F and, if E[v] is not the special flag value, also add E[v] to F
            F.add(v);
            
            if(E[v] >= 0)
                mst_edges.add(new Tuple3<Integer, Integer, Double>(v, E[v], C[v]));
            
            /*
             * c. Loop over the edges vw connecting v to other vertices w. For 
             * each such edge, if w still belongs to Q and vw has smaller weight
             * than C[w]:
             *    Set C[w] to the cost of edge vw
             *    Set E[w] to point to edge vw.
             */
            
            for(int w = 0; w < N; w++)
            {
                FibHeap.FibNode<Integer> w_node = q_nodes.get(w);
                if (w_node == null)//this node is already in F
                    continue;

                double mutual_reach_dist_vw = max(core[v], max(core[w], dm.dist(v, w, X, cache)));
                if (mutual_reach_dist_vw < C[w])
                {
                    Q.decreaseKey(w_node, mutual_reach_dist_vw);
                    C[w] = mutual_reach_dist_vw;
                    E[w] = v;
                }

            }
            
        }
        
        //prim is done, we have the MST!
        
        /*
         * 3. Extend the MST to obtain MSText, by adding for each vertex a “self
         * edge” with the core distance of the corresponding object as weight
         */
        
        for(int i = 0; i < N; i++)
            mst_edges.add(new Tuple3<Integer, Integer, Double>(i, i, core[i]));
        
        //4. Extract the HDBSCAN hierarchy as a dendrogram from MSText:
        
        List<UnionFind<Integer>> ufs = new ArrayList<UnionFind<Integer>>(N);
        for(int i = 0; i < N; i++)
            ufs.add(new UnionFind<Integer>(i));
        //sort edges from smallest weight to largest
        PriorityQueue<Tuple3<Integer, Integer, Double>> edgeQ = new PriorityQueue<Tuple3<Integer, Integer, Double>>(2*N, new Comparator<Tuple3<Integer, Integer, Double>>()
        {
            @Override
            public int compare(Tuple3<Integer, Integer, Double> o1, Tuple3<Integer, Integer, Double> o2)
            {
                return o1.getZ().compareTo(o2.getZ());
            }
        });
        edgeQ.addAll(mst_edges);
        
        //everyone starts in their own cluster!
        List<List<Integer>> currentGroups = new ArrayList<List<Integer>>();
        for(int i = 0; i < N; i++)
        {
            IntList il = new IntList(1);
            il.add(i);
            currentGroups.add(il);
        }
        
        int next_cluster_label = 0;
        /**
         * List of all the cluster options we have found
         */
        List<List<Integer>> cluster_options = new ArrayList<List<Integer>>();
        /**
         * Stores a list for each cluster. Each value in the sub list is the
         * weight at which that data point was added to the cluster 
         */
        List<DoubleList> entry_size = new ArrayList<DoubleList>();
        DoubleList birthSize = new DoubleList();
        DoubleList deathSize = new DoubleList();
        List<Pair<Integer, Integer>> children = new ArrayList<Pair<Integer, Integer>>();
        int[] map_to_cluster_label = new int[N];
        Arrays.fill(map_to_cluster_label, -1);
        
        while(!edgeQ.isEmpty())
        {
            Tuple3<Integer, Integer, Double> edge = edgeQ.poll();
            double weight = edge.getZ();
            int from = edge.getX();
            int to = edge.getY();
            
            if(to == from)
                continue;
            
            UnionFind<Integer> union_from  = ufs.get(from);
            UnionFind<Integer> union_to  = ufs.get(to);
            
            int clust_A = union_from.find().getItem();
            int clust_B = union_to.find().getItem();
            
            UnionFind<Integer> clust_A_tmp = union_from.find();
            UnionFind<Integer> clust_B_tmp = union_to.find();
            
            union_from.union(union_to);
            
            int a_size = currentGroups.get(clust_A).size();
            int b_size = currentGroups.get(clust_B).size();
            int new_size = a_size+b_size;
            
            int mergedClust;
            int otherClust;
            if(union_from.find().getItem() == clust_A)
            {
                mergedClust = clust_A;
                otherClust = clust_B;
            }
            else//other way around
            {
                mergedClust = clust_B;
                otherClust = clust_A;
            }
            
            
            if(new_size >= m_clSize && a_size < m_clSize && b_size < m_clSize)
            {//birth of a new cluster!
                cluster_options.add(currentGroups.get(mergedClust));
                
                DoubleList dl = new DoubleList(new_size);
                for(int i = 0; i < new_size; i++)
                    dl.add(weight);
                entry_size.add(dl);
                
                children.add(null);//we have not children! 
                birthSize.add(weight);
                deathSize.add(Double.MAX_VALUE);//we don't know yet
                map_to_cluster_label[mergedClust] = next_cluster_label;
                next_cluster_label++;
            }
            else if(new_size >= m_clSize && a_size >= m_clSize && b_size >= m_clSize)
            {//birth of a new cluster from the death of two others!
                //record the weight that the other two died at
                deathSize.set(map_to_cluster_label[mergedClust], weight);
                deathSize.set(map_to_cluster_label[otherClust], weight);
                
                //replace with new object so that old references in cluster_options are not altered further
                currentGroups.set(mergedClust, new IntList(currentGroups.get(mergedClust)));
                
                cluster_options.add(currentGroups.get(mergedClust));
                DoubleList dl = new DoubleList(new_size);
                for(int i = 0; i < new_size; i++)
                    dl.add(weight);
                entry_size.add(dl);
                
                children.add(new Pair<Integer, Integer>(map_to_cluster_label[mergedClust], map_to_cluster_label[otherClust]));
                birthSize.add(weight);
                deathSize.add(Double.MAX_VALUE);//we don't know yet
                map_to_cluster_label[mergedClust] = next_cluster_label;
                next_cluster_label++;
            }
            else if(new_size >= m_clSize)
            {//existing cluster has grown in size, so add the points and record their weight for use later
                //index may change, so book keeping update
                if(map_to_cluster_label[mergedClust] == -1)//the people being added are the new owners
                {
                    //set to avoid index out of boudns bellow
                    int c = map_to_cluster_label[mergedClust] = map_to_cluster_label[otherClust];
                    //make sure we keep track of the correct list 
                    cluster_options.set(c, currentGroups.get(mergedClust));
                    map_to_cluster_label[otherClust] = -1;
                }
                
                
                for(int indx : currentGroups.get(otherClust))
                    try
                    {
                        entry_size.get(map_to_cluster_label[mergedClust]).add(weight);
                    }
                    catch(IndexOutOfBoundsException ex)
                    {
                        ex.printStackTrace();
                    }
                    
            }
            
            currentGroups.get(mergedClust).addAll(currentGroups.get(otherClust));
            currentGroups.set(otherClust, null);
            
        }
        
        //Remove the last "cluster" because its the dumb one of everything 
        cluster_options.remove(cluster_options.size()-1);
        entry_size.remove(entry_size.size()-1);
        birthSize.remove(birthSize.size()-1);
        deathSize.remove(deathSize.size()-1);
        children.remove(children.size()-1);
        
        /**
         * See equation (3) in paper
         */
        double[] S = new double[cluster_options.size()];
        for(int c = 0; c < S.length; c++)
        {
            double lambda_min = birthSize.getD(c);
            double lambda_max = deathSize.getD(c);
            double s = 0;
            for(double f_x : entry_size.get(c))
                s += Math.min(f_x, lambda_max) - lambda_min;
            S[c] = s; 
        }
        
        boolean[] toKeep = new boolean[S.length];
        double[] S_hat = new double[cluster_options.size()];
        Arrays.fill(toKeep, true);
        Queue<Integer> notKeeping = new ArrayDeque<Integer>();
        
        for(int i = 0; i < S.length; i++)
        {
            Pair<Integer, Integer> child = children.get(i);
            if(child == null)//I'm a leaf!
            {
                //for all leaf nodes, set ˆS(C_h)= S(C_h)
                S_hat[i] = S[i];
                continue;
            }
            int il = child.getFirstItem();
            int ir = child.getSecondItem();
            //If S(C_i) < ˆS(C_il)+ ˆ S(C_ir ), set ˆS(C_i)= ˆS(C_il)+ ˆS(C_ir )and set δi =0.
            if(S[i] < S_hat[il] + S_hat[ir])
            {
                S_hat[i] = S_hat[il] + S_hat[ir];
                toKeep[i] = false;
            }
            else//Else: set ˆS(C_i)= S(C_i)and set δ(·) = 0 for all clusters in C_i’s subtrees.
            {
                S_hat[i] = S[i];
                //place children in q to process and set all sub children as not keeping
                notKeeping.add(il);
                notKeeping.add(ir);
                while(!notKeeping.isEmpty())
                {
                    int c = notKeeping.poll();
                    toKeep[c] = false;
                    Pair<Integer, Integer> c_children = children.get(c);
                    if(c_children == null)
                        continue;
                    notKeeping.add(c_children.getFirstItem());
                    notKeeping.add(c_children.getSecondItem());
                }
            }
        }
        
        
        //initially fill with -1 indicating it was noise
        Arrays.fill(designations, 0, N, -1);
        
        int clusters = 0;
        for(int c = 0; c < toKeep.length; c++)
            if(toKeep[c])
            {
                for(int indx : cluster_options.get(c))
                    designations[indx] = clusters;
                clusters++;
            }
        
        return designations;
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
