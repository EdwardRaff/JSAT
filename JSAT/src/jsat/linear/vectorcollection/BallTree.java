/*
 * Copyright (C) 2018 Edward Raff
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

package jsat.linear.vectorcollection;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;
import jsat.clustering.MEDDIT;
import jsat.clustering.PAM;
import jsat.clustering.TRIKMEDS;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.math.FastMath;
import jsat.utils.BoundedSortedList;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;
import jsat.utils.IntList;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import jsat.utils.Pair;
import jsat.utils.concurrent.AtomicDoubleArray;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 * This class implements the Ball Tree algorithm for accelerating nearest
 * neighbor queries. Contained within this class are multiple methods of
 * building Ball Trees. Options for changing construction can alter the method
 * of {@link ConstructionMethod construction} of the hierarchy is made, or how
 * {@link PivotSelection  pivot} is selected. <br>
 * The default method of construction and pivot selection for ball trees will
 * work for most cases, but is not appicable for all distance metrics. If you
 * are using an exotic distance metric the
 * {@link BallTree.ConstructionMethod#TOP_DOWN_FARTHEST} and
 * {@link PivotSelection#MEDOID} will work for any dataset, but may be
 * slower.<br>
 * <br>
 * See:
 * <ul>
 * <li>Omohundro, S. M. (1989). Five Balltree Construction Algorithms (No.
 * TR-89-063).</li>
 * <li>Moore, A. W. (2000). The Anchors Hierarchy: Using the Triangle Inequality
 * to Survive High Dimensional Data. In Proceedings of the Sixteenth Conference
 * on Uncertainty in Artificial Intelligence (pp. 397–405). San Francisco, CA,
 * USA: Morgan Kaufmann Publishers Inc. Retrieved from
 * <a href="http://dl.acm.org/citation.cfm?id=2073946.2073993">here</a></li>
 * </ul>
 *
 * @author Edward Raff
 * @param <V>
 */
public class BallTree<V extends Vec> implements IncrementalCollection<V>, DualTree<V>
{
    public static final int DEFAULT_LEAF_SIZE = 40;
    private int leaf_size = DEFAULT_LEAF_SIZE;
    private DistanceMetric dm;
    private List<V> allVecs;
    private List<Double> cache;
    private ConstructionMethod construction_method;
    private PivotSelection pivot_method;
    private Node root;

    @Override
    public IndexNode getRoot()
    {
        return root;
    }

    @Override
    public List<Double> getAccelerationCache()
    {
        return cache;
    }
    
    public enum ConstructionMethod
    {
        /**
         * This represents a top-down construction approach, that can be used
         * for any distance metric. At each branch the children are given
         * initial prototypes. The left child has a prototype selected as the
         * point farthest from the pivot, and the right child the point farthest
         * from the left's. The points are split based on which prototype thye
         * are closest too. The process continues recursively. <br>
         *
         * See: Moore, A. W. (2000). The Anchors Hierarchy: Using the Triangle
         * Inequality to Survive High Dimensional Data. In Proceedings of the
         * Sixteenth Conference on Uncertainty in Artificial Intelligence (pp.
         * 397–405). San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.
         * Retrieved from http://dl.acm.org/citation.cfm?id=2073946.2073993 for
         * details.
         */
        TOP_DOWN_FARTHEST,
        /**
         * This represents a top-down construction approach similar to a
         * KD-tree's construction. It requires a metric where it has access to
         * meaningful feature values. At each node, the dimension with the
         * largest spread in values is selected. Then the split is made based on
         * sorting the found feature into two even halves.<br>
         * See: Omohundro, S. M. (1989). Five Balltree Construction Algorithms
         * (No. TR-89-063).
         */
        KD_STYLE,
        /**
         * This represents a "middle-out" construction approach. Computational
         * is works best when the {@link PivotSelection#CENTROID centroid} pivot
         * selection method can be used.<br>
         * See: Moore, A. W. (2000). The Anchors Hierarchy: Using the Triangle
         * Inequality to Survive High Dimensional Data. In Proceedings of the
         * Sixteenth Conference on Uncertainty in Artificial Intelligence (pp.
         * 397–405). San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.
         * Retrieved from http://dl.acm.org/citation.cfm?id=2073946.2073993 for
         * details.
         */
        ANCHORS_HIERARCHY;
    }
    
    public enum PivotSelection
    {
        /**
         * This method selects the pivot by taking the centroid (average) of all
         * the data within a node. This method may not be applicable for all
         * metrics, and can't be used for once for which there is no computable
         * average.
         */
        CENTROID 
        {
            @Override
            public Vec getPivot(boolean parallel, List<Integer> points, List<? extends Vec> data,  DistanceMetric dm, List<Double> cache)
            {
                if (points.size() == 1)
                    return data.get(points.get(0)).clone();
                
                Vec pivot = new DenseVector(data.get(points.get(0)).length());
                for (int i : points)
                    pivot.mutableAdd(data.get(i));
                pivot.mutableDivide(points.size());
                return pivot;
                
            }
        },
        /**
         * This method selects the pivot by searching for the medoid of the
         * data. This can be used in all circumstances, but may be slower.
         */
        MEDOID 
        {
            @Override
            public Vec getPivot(boolean parallel, List<Integer> points, List<? extends Vec> data,  DistanceMetric dm, List<Double> cache)
            {
                if (points.size() == 1)
                    return data.get(points.get(0)).clone();
                int indx;
                if(dm.isValidMetric())
                    indx = TRIKMEDS.medoid(parallel, points, data, dm, cache);
                else
                    indx = PAM.medoid(parallel, points, data, dm, cache);
                return data.get(indx);
            }
        },
        /**
         * This method selects the pivot by searching for an approximate medoid
         * of the data. This can be used in all circumstances, but may be
         * slower.
         */
        MEDOID_APRX
        {
            @Override
            public Vec getPivot(boolean parallel, List<Integer> points, List<? extends Vec> data,  DistanceMetric dm, List<Double> cache)
            {
                if (points.size() == 1)
                    return data.get(points.get(0)).clone();
                int indx;
                //Faster to use exact search
                if(points.size() < 1000)
                {
                    if(dm.isValidMetric())
                        indx = TRIKMEDS.medoid(parallel, points, data, dm, cache);
                    else
                        indx = PAM.medoid(parallel, points, data, dm, cache);
                }
                else//Lets do approx search
                    indx = MEDDIT.medoid(parallel, points, 0.2, data, dm, cache);
                return data.get(indx);
            }
        },
        /**
         * A random point will be selected as the pivot for the ball
         */
        RANDOM
        {
            @Override
            public Vec getPivot(boolean parallel, List<Integer> points, List<? extends Vec> data,  DistanceMetric dm, List<Double> cache)
            {
               int indx = RandomUtil.getLocalRandom().nextInt(points.size());
               return data.get(indx);
            }
        },;
        
        public abstract Vec getPivot(boolean parallel, List<Integer> points, List<? extends Vec> data, DistanceMetric dm, List<Double> cache);
    }

    public BallTree()
    {
        this(new EuclideanDistance(), ConstructionMethod.KD_STYLE, PivotSelection.CENTROID);
    }

    public BallTree(DistanceMetric dm, ConstructionMethod method, PivotSelection pivot_method)
    {
        setConstruction_method(method);
        setPivot_method(pivot_method);
        setDistanceMetric(dm);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public BallTree(BallTree toCopy)
    {
        this(toCopy.dm, toCopy.construction_method, toCopy.pivot_method);
        if(toCopy.allVecs != null)
            this.allVecs = new ArrayList<>(toCopy.allVecs);
        if(toCopy.cache != null)
            this.cache = new DoubleList(toCopy.cache);
        if(toCopy.root != null)
            this.root = cloneChangeContext(toCopy.root);
        this.leaf_size = toCopy.leaf_size;
    }
    
    @Override
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }
    
    @Override
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }

    /**
     * Sets the number of points stored within a leaf node of the index. Larger
     * values avoid search overhead, but reduce opportunities for pruning.
     *
     * @param leaf_size the size of a leaf node. Must be at least 2
     */
    public void setLeafSize(int leaf_size)
    {
        if (leaf_size < 2)
            throw new IllegalArgumentException("The leaf size must be >= 2 to support all splitting methods");
        this.leaf_size = leaf_size;
    }

    /**
     *
     * @return the number of points to store within a leaf node
     */
    public int getLeafSize()
    {
        return leaf_size;
    }
    
    /**
     * Computes the maximum depth of the current tree. A value of zero indicates
     * that only a root node exists or the tree is empty. Any other value is the
     * maximum number of children a node contains.
     *
     * @return the maximum current depth of this Ball Tree
     */
    public int getMaxDepth()
    {
        if(root == null)
            return 0;
        else 
            return root.findMaxDepth(0);
    }

    public void setPivot_method(PivotSelection pivot_method)
    {
        this.pivot_method = pivot_method;
    }

    public PivotSelection getPivot_method()
    {
        return pivot_method;
    }

    public void setConstruction_method(ConstructionMethod construction_method)
    {
        this.construction_method = construction_method;
    }

    public ConstructionMethod getConstruction_method()
    {
        return construction_method;
    }
    
    private Node build_far_top_down(List<Integer> points, boolean parallel)
    {
        Branch branch = new Branch();
        branch.setPivot(points);
        branch.setRadius(points);

        //Use point farthest from parent pivot for left child
        int f1 = ParallelUtils.streamP(points.stream(), parallel)
                .map(i->new IndexDistPair(i, dm.dist(i, branch.pivot, branch.pivot_qi, allVecs, cache)))
                .max(IndexDistPair::compareTo).orElse(new IndexDistPair(0, 0.0)).indx;
        //use point farhter from f1 for right child
        int f2 = ParallelUtils.streamP(points.stream(), parallel)
                .map(i->new IndexDistPair(i, dm.dist(i, f1, allVecs, cache)))
                .max(IndexDistPair::compareTo).orElse(new IndexDistPair(1, 0.0)).indx;

        //Now split children based on who is closes to f1 and f2
        IntList left_children = new IntList();
        IntList right_children = new IntList();
        for(int p : points)
        {
            double d_f1 = dm.dist(p, f1, allVecs, cache);
            double d_f2 = dm.dist(p, f2, allVecs, cache);
            if(d_f1 < d_f2)
                left_children.add(p);
            else
                right_children.add(p);
        }
        
        if(left_children.isEmpty() || right_children.isEmpty())
        {
            //This can happen if all the points have the exact same value, so all distances are zero. 
            //So we can't branch, return a leaf node instead
            left_children.addAll(right_children);
            Leaf leaf = new Leaf(left_children);
            leaf.pivot = branch.pivot;
            leaf.pivot_qi = branch.pivot_qi;
            leaf.radius = 0.0;//Weird, but correct! All dists = 0, so radius = 0
            return leaf;
        }

        //everyone has been assigned, now creat children objects
        branch.left_child = build(left_children, parallel);
        branch.right_child = build(right_children, parallel);
        branch.left_child.parent = branch;
        branch.right_child.parent = branch;

        return branch;
    }
    
    private Node build_kd(List<Integer> points, boolean parallel)
    {
        //Lets find the dimension with the maximum spread
        int D = allVecs.get(0).length();
        final boolean isSparse = allVecs.get(0).isSparse();
        
        //If sparse, keep a set of indexes we HAVE NOT SEEN
        //these have implicity zeros we need to add back at the end
        final Set<Integer> neverSeen;
        if (isSparse)
            if (parallel)
            {
                neverSeen = ConcurrentHashMap.newKeySet();
                ListUtils.addRange(neverSeen, 0, D, 1);
            }
            else
                neverSeen = new IntSet(ListUtils.range(0, D));
        else
            neverSeen = Collections.EMPTY_SET;
        
        AtomicDoubleArray mins = new AtomicDoubleArray(D);
        mins.fill(Double.POSITIVE_INFINITY);
        AtomicDoubleArray maxs = new AtomicDoubleArray(D);
        maxs.fill(Double.NEGATIVE_INFINITY);
        ParallelUtils.streamP(points.stream(), parallel).forEach(i->
        {
            for(IndexValue iv : get(i))
            {
                int d = iv.getIndex();
                mins.updateAndGet(d, (m_d)->Math.min(m_d, iv.getValue()));
                maxs.updateAndGet(d, (m_d)->Math.max(m_d, iv.getValue()));
                neverSeen.remove(d);
            }
        });
        
        IndexDistPair maxSpread = ParallelUtils.range(D, parallel)
                .mapToObj(d->
                {
                    double max_d = maxs.get(d), min_d = mins.get(d);
                    if(neverSeen != null && neverSeen.contains(d))
                    {
                        max_d = Math.max(max_d, 0);
                        min_d = Math.min(min_d, 0);
                    }
                    return new IndexDistPair(d, max_d - min_d);
                })
                .max(IndexDistPair::compareTo).get();

        
        if(maxSpread.dist == 0)//all the data is the same? Return a leaf
        {
            Leaf leaf = new Leaf(new IntList(points));
            leaf.setPivot(points);
            leaf.setRadius(points);
            return leaf;
        }
        
        //We found it! Lets sort points by this new value
        final int d = maxSpread.indx;
        points.sort((Integer o1, Integer o2) -> Double.compare(get(o1).get(d), get(o2).get(d)));
        int midPoint = points.size()/2;
        //Lets check that we don't have identical values, and adjust as needed
        while(midPoint > 1 && get(midPoint-1).get(d) == get(midPoint).get(d))
            midPoint--;
        List<Integer> left_children = points.subList(0, midPoint);
        List<Integer> right_children = points.subList(midPoint, points.size());
        
        Branch branch = new Branch();
        branch.setPivot(points);
        branch.setRadius(points);
        //everyone has been assigned, now creat children objects
        branch.left_child = build(left_children, parallel);
        branch.right_child = build(right_children, parallel);
        branch.left_child.parent = branch;
        branch.right_child.parent = branch;
        
        
        return branch;
    }
    
    private Node build_anchors(List<Integer> points, boolean parallel)
    {
        //Ceiling to avoid issues with points rounding down to k=1, causing an infinite recusion 
        int K = (int) Math.ceil(Math.sqrt(points.size()));
        
        int[] anchor_point_index = new int[K];
        int[] anchor_index = new int[K];
        IntList[] owned = new IntList[K];
        //anchor paper says sort from hight dist to low, we do reverse for convience and removal efficiancy 
        DoubleList[] ownedDist = new DoubleList[K];
        for(int k = 1; k < K; k++)
        {
            owned[k] = new IntList();
            ownedDist[k] = new DoubleList();
        }
        
        Random rand = RandomUtil.getRandom();
        
        //First case is special, select anchor at random and create list
        anchor_point_index[0] =rand.nextInt(points.size());
        anchor_index[0] = points.get(anchor_point_index[0]);
        owned[0] = IntList.range(points.size());
        ownedDist[0] = DoubleList.view(ParallelUtils.streamP(owned[0].streamInts(), parallel)
                .mapToDouble(i->dm.dist(anchor_index[0], points.get(i), allVecs, cache))
                .toArray(),
                    points.size());

        IndexTable it = new IndexTable(ownedDist[0]);
        it.apply(owned[0]);
        it.apply(ownedDist[0]);
        
        //Now lets create the other anchors
        for(int k = 1; k < K; k++)
        {
            /*
             * How is the new anchor a^new chosen? We simply find the current 
             * anchor a^maxrad with the largest radius, and choose the pivot of 
             * a^new to be the point owned by a^maxrad that is furthest from 
             * a^maxrad
             */
            int max_radius_anch = IntStream.range(0, k).mapToObj(z-> new IndexDistPair(z, ownedDist[z].get(ownedDist[z].size()-1)))
                    .max(IndexDistPair::compareTo)
                    .get().indx;
            anchor_point_index[k] = owned[max_radius_anch].getI(owned[max_radius_anch].size()-1);
            anchor_index[k] = points.get(anchor_point_index[k]);
            owned[max_radius_anch].remove(owned[max_radius_anch].size()-1);
            ownedDist[max_radius_anch].remove(ownedDist[max_radius_anch].size()-1);
            owned[k].add(anchor_point_index[k]);
            ownedDist[k].add(0.0);
            
            //lets go through other anchors and see what we can steal
            for(int j = 0; j < k; j++)
            {
                double dist_ak_aj = dm.dist(anchor_index[j], anchor_index[k], allVecs, cache);
                
                ListIterator<Integer> ownedIter = owned[j].listIterator(owned[j].size());
                ListIterator<Double> ownedDistIter = ownedDist[j].listIterator(ownedDist[j].size());
                while(ownedIter.hasPrevious())
                {
                    int point_indx = ownedIter.previous();
                    double dist_aj_x = ownedDistIter.previous();
                    double dist_ak_x = dm.dist(anchor_index[k], points.get(point_indx), allVecs, cache);
                    if(dist_ak_x < dist_aj_x)//we can steal this point! 
                    {
                        owned[k].add(point_indx);
                        ownedDist[k].add(dist_ak_x);
                        ownedIter.remove();
                        ownedDistIter.remove();
                    }
                    else if(dist_ak_x < dist_ak_aj/2)
                    {
                        //"we can deduce that the remainder of the points in ai's list cannot possibly be stolen"
                        break;
                    }
                }
            }
            //now sort our new children
            it = new IndexTable(ownedDist[k]);
            it.apply(owned[k]);
            it.apply(ownedDist[k]);
        }
        
        //Now we have sqrt(R) anchors. Lets do the middle-down first, creating Nodes for each anchor
        List<Node> anchor_nodes = new ArrayList<>();
        for (int k = 0; k < K; k++)
        {
            Node n_k = build(IntList.view(owned[k].streamInts().map(i->points.get(i)).toArray()), parallel);
            n_k.pivot = get(anchor_index[k]);
            n_k.radius = ownedDist[k].getD(ownedDist[k].size()-1);
            anchor_nodes.add(n_k);
        }
        
        //TODO below code is ugly... needs improvement
        
        //Ok, now lets go middle-up to finish the tree
        //We will store the costs of merging any pair of anchor_nodes in this map
        Map<Pair<Integer, Integer>, Double> mergeCost = new HashMap<>();
        Map<Pair<Integer, Integer>, Vec> pivotCache = new HashMap<>();
        //use a priority queue to pop of workers, and use values from mergeCost to sort
        List<PriorityQueue<Pair<Integer, Integer>>> mergeQs = new ArrayList<>();
        PriorityQueue<Integer> QQ = new PriorityQueue<>((q1, q2)-> 
        {
            double v1 = mergeCost.get(mergeQs.get(q1).peek());
            double v2 = mergeCost.get(mergeQs.get(q2).peek());
            return Double.compare(v1, v2);
        });
        
        ///Initial population of Qs and costs
        for(int k = 0; k < K; k++)
        {
            PriorityQueue<Pair<Integer, Integer>> mergeQ_k = new PriorityQueue<>((Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) ->
                Double.compare(mergeCost.get(o1), mergeCost.get(o2)));
            mergeQs.add(mergeQ_k);
            Node n_k = anchor_nodes.get(k);
            IntList owned_nk = new IntList();
            for(int i : n_k)
                owned_nk.add(i);
            int size_k = owned_nk.size();
            for(int z = k+1; z < K; z++)
            {
                Node n_z = anchor_nodes.get(z);
                Pair<Integer, Integer> p = new Pair<>(k, z);
                
                
                IntList owned_nkz = new IntList(owned_nk);
                int size_z, size_nk;
                
                for(int i : n_z)
                    owned_nkz.add(i);
                size_nk = owned_nkz.size();
                size_z = size_nk-size_k;
                
                Vec pivot_candidate;
                if(pivot_method == PivotSelection.CENTROID)
                {
                    //we can directly compute the would-be pivot
                    pivot_candidate = n_k.pivot.clone();
                    pivot_candidate.mutableMultiply(size_k/(double)size_nk);
                    pivot_candidate.mutableAdd(size_z/(double)size_nk, n_z.pivot);
                }
                else//we need to compute the pivot
                    pivot_candidate = pivot_method.getPivot(parallel, owned_nkz, allVecs, dm, cache);
                
                List<Double> pivor_candidate_qi = dm.getQueryInfo(pivot_candidate);
                //what would the radius be?
                double radius_kz = 0;
                for(int i : owned_nkz)
                    radius_kz = Math.max(radius_kz, dm.dist(i, pivot_candidate, pivor_candidate_qi, allVecs, cache));
                mergeCost.put(p, radius_kz);
                pivotCache.put(p, pivot_candidate);

                mergeQ_k.add(p);
            }
            
            if(!mergeQ_k.isEmpty())
                QQ.add(k);
        }
        
        //Now lets start merging! 
        Branch toReturn = null;
        while(!QQ.isEmpty())
        {
            int winningQ = QQ.poll();
            Pair<Integer, Integer> toMerge = mergeQs.get(winningQ).poll();
            int other = toMerge.getSecondItem();
            if(anchor_nodes.get(winningQ) == null)//leftover, its gone
                continue;
            else if(anchor_nodes.get(other) == null)
            {
                if(!mergeQs.get(winningQ).isEmpty())//stale, lets fix
                    QQ.add(winningQ);
                continue;
            }
            Branch merged = toReturn = new Branch();
            merged.pivot = pivotCache.get(toMerge);
            merged.pivot_qi = dm.getQueryInfo(merged.pivot);
            merged.radius = mergeCost.get(toMerge);
            merged.left_child = anchor_nodes.get(winningQ);
            merged.right_child = anchor_nodes.get(other);
            merged.left_child.parent = merged;
            merged.right_child.parent = merged;
            anchor_nodes.set(winningQ, merged);
            anchor_nodes.set(other, null);
            
            //OK, we have merged two points. Now book keeping. Remove all Qs 
            PriorityQueue<Pair<Integer, Integer>> mergeQ_k = new PriorityQueue<>((Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) ->
                Double.compare(mergeCost.get(o1), mergeCost.get(o2)));
            mergeQs.set(winningQ, mergeQ_k);
            
            Node n_k = merged;
            IntList owned_nk = new IntList();
            for(int i : n_k)
                owned_nk.add(i);
            int size_k = owned_nk.size();
            for(int z = 0; z < anchor_nodes.size(); z++)
            {
                if(z == winningQ)
                    continue;
                if(anchor_nodes.get(z) == null)
                    continue;
                Node n_z = anchor_nodes.get(z);
                Pair<Integer, Integer> p;
                if(winningQ < z)
                    p = new Pair<>(winningQ, z);
                else
                    p = new Pair<>(z, winningQ);
                
                
                IntList owned_nkz = new IntList(owned_nk);
                int size_z, size_nk;
                
                for(int i : n_z)
                    owned_nkz.add(i);
                size_nk = owned_nkz.size();
                size_z = size_nk-size_k;
                
                Vec pivot_candidate;
                if(pivot_method == PivotSelection.CENTROID)
                {
                    //we can directly compute the would-be pivot
                    pivot_candidate = n_k.pivot.clone();
                    pivot_candidate.mutableMultiply(size_k/(double)size_nk);
                    pivot_candidate.mutableAdd(size_z/(double)size_nk, n_z.pivot);
                }
                else//we need to compute the pivot
                    pivot_candidate = pivot_method.getPivot(parallel, owned_nkz, allVecs, dm, cache);
                
                List<Double> pivor_candidate_qi = dm.getQueryInfo(pivot_candidate);
                //what would the radius be?
                double radius_kz = 0;
                for(int i : owned_nkz)
                    radius_kz = Math.max(radius_kz, dm.dist(i, pivot_candidate, pivor_candidate_qi, allVecs, cache));
                
                pivotCache.put(p, pivot_candidate);

                if(winningQ < z)
                {
                    mergeCost.put(p, radius_kz);
                    mergeQ_k.add(p);
                }
                else
                {
                    mergeQs.get(z).remove(p);
                    mergeCost.put(p, radius_kz);
                    mergeQs.get(z).add(p);
                }
            }
            
            if(!mergeQ_k.isEmpty())
                QQ.add(winningQ);
        }
        
        return toReturn;
    }
    
    private Node build(List<Integer> points, boolean parallel)
    {
        //universal base case
        if(points.size() <= leaf_size)
        {
            Leaf leaf = new Leaf(new IntList(points));
            leaf.setPivot(points);
            leaf.setRadius(points);
            return leaf;
        }
        
        switch(construction_method)
        {
            case ANCHORS_HIERARCHY:
                return build_anchors(points, parallel);
            case KD_STYLE:
                return build_kd(points, parallel);
            case TOP_DOWN_FARTHEST:
                return build_far_top_down(points, parallel);
                
        }
        return new Leaf(new IntList(0));
    }
    
    @Override
    public void build(boolean parallel, List<V> collection, DistanceMetric dm)
    {
        this.allVecs = new ArrayList<>(collection);
        setDistanceMetric(dm);
        this.cache = dm.getAccelerationCache(allVecs, parallel);
        this.root = build(IntList.range(collection.size()), parallel);
    }

    @Override
    public void insert(V x)
    {

        if(root == null)
        {
            allVecs = new ArrayList<>();
            allVecs.add(x);
            cache = dm.getAccelerationCache(allVecs);
            
            root = new Leaf(IntList.range(1));
            root.pivot = x.clone();
            root.pivot_qi = dm.getQueryInfo(x);
            root.radius = 0;
            
            return;
        }
        int indx = allVecs.size();
        allVecs.add(x);
        if(cache != null)
            cache.addAll(dm.getQueryInfo(x));
        
        Branch parentNode = null;
        Node curNode = root;
        double dist_to_curNode = dm.dist(indx, curNode.pivot, curNode.pivot_qi, allVecs, cache);
        while(curNode != null)
        {
            curNode.radius = Math.max(curNode.radius, dist_to_curNode);
            
            if(curNode instanceof jsat.linear.vectorcollection.BallTree.Leaf)
            {
                Leaf lroot = (Leaf) curNode;
                lroot.children.add(indx);
                
                if(lroot.children.size() > leaf_size)
                {
                    Node newNode = build(lroot.children, false);
                    if(parentNode == null)//We are the root node and a leaf
                        root = newNode;
                    else if(parentNode.left_child == curNode)//YES, intentinoally checking object equality
                        parentNode.left_child = newNode;
                    else
                        parentNode.right_child = newNode;
                }
                return;
            }
            else
            {
                Branch b = (Branch) curNode;
                double left_dist = dm.dist(indx, b.left_child.pivot, b.left_child.pivot_qi, allVecs, cache);
                double right_dist = dm.dist(indx, b.right_child.pivot, b.right_child.pivot_qi, allVecs, cache);
                
                boolean goLeftBranch;
                
                goLeftBranch = left_dist < right_dist;
                
                //decend tree
                parentNode = b;
                if(goLeftBranch)
                {
                    curNode = b.left_child;
                    dist_to_curNode = left_dist;
                }
                else
                {
                    curNode = b.right_child;
                    dist_to_curNode = right_dist;
                }
            }
        }
    }

    @Override
    public BallTree<V> clone()
    {
        return new BallTree<>(this);
    }

    @Override
    public void search(Vec query, double range, List<Integer> neighbors, List<Double> distances)
    {
        neighbors.clear();
        distances.clear();
        root.search(query, dm.getQueryInfo(query), range, neighbors, distances);
        
        IndexTable it = new IndexTable(distances);
        it.apply(distances);
        it.apply(neighbors);
    }

    @Override
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances)
    {
        neighbors.clear();
        distances.clear();
        
        BoundedSortedList<IndexDistPair> knn = new BoundedSortedList<>(numNeighbors);
        root.search(query, dm.getQueryInfo(query), numNeighbors, knn, Double.POSITIVE_INFINITY);
        for(IndexDistPair p : knn)
        {
            neighbors.add(p.indx);
            distances.add(p.dist);
        }
    }

    @Override
    public V get(int indx)
    {
        return allVecs.get(indx);
    }

    @Override
    public int size()
    {
        return allVecs.size();
    }

    private abstract class Node implements Cloneable, Serializable, Iterable<Integer>, IndexNode<Node>
    {
        Vec pivot;
        List<Double> pivot_qi;
        double radius;
        Node parent;
        double parrent_dist = Double.POSITIVE_INFINITY;

        public Node()
        {
        }

        public Node(Node toCopy)
        {
            if(toCopy.pivot != null)
                this.pivot = toCopy.pivot.clone();
            if(toCopy.pivot_qi != null)
                this.pivot_qi = new DoubleList(toCopy.pivot_qi);
            this.radius = toCopy.radius;
        }
        
        public void setPivot(List<Integer> points)
        {
            if(points.size() == 1)
                pivot = get(points.get(0)).clone();
            else
                pivot = pivot_method.getPivot(false, points, allVecs, dm, cache);
            pivot_qi = dm.getQueryInfo(pivot);
        }
        
        public void setRadius(List<Integer> points)
        {
            this.radius = 0;
            for(int i : points)
                radius = Math.max(radius, dm.dist(i, pivot, pivot_qi, allVecs, cache));
        }
        
        abstract public int findMaxDepth(int curDepth);
        
        abstract public void search(Vec query, List<Double> qi, double range, List<Integer> neighbors, List<Double> distances);
        
        abstract public void search(Vec query, List<Double> qi, int numNeighbors, BoundedSortedList<IndexDistPair> knn, double pivot_to_query);

        @Override
        public double minNodeDistance(int other)
        {
            return 0;
        }

        @Override
        public double minNodeDistance(Node other)
        {
            return dm.dist(this.pivot, other.pivot) - this.radius - other.radius;
        }

        @Override
        public double furthestDescendantDistance()
        {
            return radius;
        }
        
        @Override
        public double maxNodeDistance(Node other)
        {
            return dm.dist(this.pivot, other.pivot) + this.radius + other.radius;
        }

        @Override
        public double[] minMaxDistance(Node other)
        {
            double d = dm.dist(this.pivot, other.pivot);
            return new double[]
            {
                d - this.radius - other.radius, 
                d + this.radius + other.radius
            };
        }
        
        
        
        @Override
        public double furthestPointDistance()
        {
            return 0;//don't own any points, so dist is zero
        }
        
        @Override
        public Node getParrent()
        {
            return parent;
        }

//        @Override
//        public double getParentDistance()
//        {
//            return parrent_dist;
//        }

        @Override
        public Vec getVec(int indx)
        {
            return get(indx);
        }

    }
    
    private class Leaf extends Node
    {
        IntList children;

        public Leaf(IntList children)
        {
            this.children = children;
        }

        public Leaf(Leaf toCopy)
        {
            super(toCopy);
            this.children = new IntList(toCopy.children);
        }

        @Override
        public void search(Vec query, List<Double> qi, double range, List<Integer> neighbors, List<Double> distances)
        {
            for(int indx : children)
            {
                double dist = dm.dist(indx, query, qi, allVecs, cache);
                if(dist <= range)
                {
                    neighbors.add(indx);
                    distances.add(dist);
                }
            }
        }

        @Override
        public void search(Vec query, List<Double> qi, int numNeighbors, BoundedSortedList<IndexDistPair> knn, double pivot_to_query)
        {
            for(int indx : children)
                knn.add(new IndexDistPair(indx, dm.dist(indx, query, qi, allVecs, cache)));
        }

        @Override
        public Iterator<Integer> iterator()
        {
            return children.iterator();
        }

        @Override
        public int findMaxDepth(int curDepth)
        {
            return curDepth;
        }

        @Override
        public int numChildren()
        {
            return 0;
        }

        @Override
        public IndexNode getChild(int indx)
        {
            throw new IndexOutOfBoundsException("Leaf nodes do not have children");
        }

        @Override
        public int numPoints()
        {
            return children.size();
        }

        @Override
        public int getPoint(int indx)
        {
            return children.get(indx);
        }
        
    }
    
    private class Branch extends Node
    {
        Node left_child;
        Node right_child;

        public Branch()
        {
        }
        
        public int findMaxDepth(int curDepth)
        {
            return Math.max(left_child.findMaxDepth(curDepth+1), right_child.findMaxDepth(curDepth+1));
        }

        public Branch(Branch toCopy)
        {
            super(toCopy);
            this.left_child = cloneChangeContext(toCopy.left_child);
            this.right_child = cloneChangeContext(toCopy.right_child);
        }
        
        @Override
        public void search(Vec query, List<Double> qi, double range, List<Integer> neighbors, List<Double> distances)
        {
            if(dm.dist(query, pivot) - radius >= range)
                return;//We can prune this branch!
            left_child.search(query, qi, range, neighbors, distances);
            right_child.search(query, qi, range, neighbors, distances);
        }

        @Override
        public void search(Vec query, List<Double> qi, int numNeighbors, BoundedSortedList<IndexDistPair> knn, double pivot_to_query)
        {
            if(Double.isInfinite(pivot_to_query))//can happen for first call
                pivot_to_query = dm.dist(query, pivot);
            if(knn.size() >= numNeighbors && pivot_to_query - radius >= knn.last().dist)
                return;//We can prune this branch!
            double dist_left = dm.dist(query, left_child.pivot);
            double dist_right = dm.dist(query, right_child.pivot);
            
            double close_child_dist = dist_left;
            Node close_child = left_child;
            double far_child_dist = dist_right;
            Node far_child = right_child;
            
            if(dist_right < dist_left)
            {
                close_child_dist = dist_right;
                close_child = right_child;
                far_child_dist = dist_left;
                far_child = left_child;
            }
            
            close_child.search(query, qi, numNeighbors, knn, close_child_dist);
            far_child.search(query, qi, numNeighbors, knn, far_child_dist);
        }

        @Override
        public Iterator<Integer> iterator()
        {
            Iterator<Integer> iter_left = left_child.iterator();
            if(right_child == null)
                System.out.println("AWD?");
            Iterator<Integer> iter_right = right_child.iterator();
            return new Iterator<Integer>()
            {
                @Override
                public boolean hasNext()
                {
                    return iter_left.hasNext() || iter_right.hasNext();
                }

                @Override
                public Integer next()
                {
                    if(iter_left.hasNext())
                        return iter_left.next();
                    else
                        return iter_right.next();
                }
            };
        }

        @Override
        public int numChildren()
        {
            return 2;
        }

        @Override
        public IndexNode getChild(int indx)
        {
            switch(indx)
            {
                case 0:
                    return left_child;
                case 1:
                    return right_child;
                default:
                    throw new IndexOutOfBoundsException();
            }
        }

        @Override
        public int numPoints()
        {
            return 0;
        }

        @Override
        public int getPoint(int indx)
        {
            throw new IndexOutOfBoundsException("Branching node does not contain any children");
        }
        
    }
    
    private Node cloneChangeContext(Node toClone)
    {
        if (toClone != null)
            if (toClone instanceof jsat.linear.vectorcollection.BallTree.Leaf)
                return new Leaf((Leaf) toClone);
            else
                return new Branch((Branch) toClone);
        return null;
    }
}
