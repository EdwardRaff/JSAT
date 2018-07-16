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

import static java.lang.Math.*;
import java.util.ArrayList;
import java.util.List;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.BoundedSortedList;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;
import jsat.utils.IntList;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.function.BiFunction;
import jsat.utils.ListUtils;
/**
 *
 * @author Edward Raff
 * @param <V>
 */
public interface DualTree<V extends Vec> extends VectorCollection<V>
{
    
    public IndexNode getRoot();

    @Override
    public DualTree<V> clone();
    
    default public double dist(int self_index, int other_index, DualTree<V> other)
    {
        
        return getDistanceMetric().dist(this.get(self_index), other.get(self_index));
    }

    @Override
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances);
    
    
    @Override
    default public void search(VectorCollection<V> VC, int numNeighbors, List<List<Integer>> neighbors, List<List<Double>> distances, boolean parallel )
    {
        if(!(VC instanceof DualTree))
        {
            VectorCollection.super.search(VC, numNeighbors, neighbors, distances, parallel);
            return;
        }
        DualTree<V> Q = (DualTree<V>) VC;
        
        //Mpa each node to a cached value. This is used for recursive bound updates
        Map<IndexNode, Double> query_B_cache = parallel ? new ConcurrentHashMap<>(Q.size()) : new HashMap<>(Q.size());
        
        //For each item in Q, we want to find its nearest neighbor in THIS collection. 
        //each item in Q gets a priority queue of k-nns
        List<BoundedSortedList<IndexDistPair>> allPriorities = new ArrayList<>();
        for(int i = 0; i < Q.size(); i++)
            allPriorities.add(new BoundedSortedList<>(numNeighbors));
        
        ///For simplicity and fast calculations, lets combine acceleration caches into one view
        final List<Double> this_cache = this.getAccelerationCache();
        final List<Double> other_cache = Q.getAccelerationCache();
        
        final int N_r = this.size();
        final List<Double> wholeCache = this_cache == null ? null : ListUtils.mergedView(this_cache, other_cache);
        final List<Vec> allVecs = new ArrayList<>(N_r+Q.size());
        for(int i = 0; i < N_r; i++)
            allVecs.add(this.get(i));
        for(int i = 0; i < Q.size(); i++)
            allVecs.add(Q.get(i));
        
        DistanceMetric dm = getDistanceMetric();
        
        BaseCaseDT base;
        if(!parallel)//easy serial case
            base = (int r_indx, int q_indx) ->
            {
                double d = dm.dist(r_indx, N_r+q_indx, allVecs, wholeCache);

                allPriorities.get(q_indx).add(new IndexDistPair(r_indx, d));
                return d;
            };
        else//slightly more complicated parallel case
            base = (int r_indx, int q_indx) ->
            {
                double d = dm.dist(r_indx, N_r + q_indx, allVecs, wholeCache);

                BoundedSortedList<IndexDistPair> target = allPriorities.get(q_indx);

                synchronized (target)
                {
                    target.add(new IndexDistPair(r_indx, d));
                }
                return d;
            };
        
        
        
        ScoreDTLazy score = (IndexNode ref, IndexNode query, double origScore) ->
        {
            if(origScore < 0)
                return ref.minNodeDistance(query);
            double bound_final = computeKnnBound(query, numNeighbors, allPriorities, query_B_cache);
            
            final double d_min_b = origScore;
            if(Double.isFinite(bound_final))
            {
                query_B_cache.put(query, bound_final);
                
                if(d_min_b > bound_final)//YAY we can prune!
                    return Double.NaN;
            }
            //default case, don't prune
            return d_min_b;
        };
        
        traverse(Q, base, score, true, parallel);
        
        
        neighbors.clear();
        distances.clear();
        for(int i = 0; i < Q.size(); i++)
        {
            IntList n = new IntList(numNeighbors);
            DoubleList d = new DoubleList(numNeighbors);
            
            BoundedSortedList<IndexDistPair> knn = allPriorities.get(i);
            for(int j = 0; j < knn.size(); j++)
            {
                IndexDistPair ip = knn.get(j);
                n.add(ip.getIndex());
                d.add(ip.getDist());
            }
            neighbors.add(n);
            distances.add(d);
            
        }
        
    }

    default double computeKnnBound(IndexNode query, int numNeighbors, List<BoundedSortedList<IndexDistPair>> allPriorities, Map<IndexNode, Double> query_B_cache)
    {
        double bound_1 = Double.NEGATIVE_INFINITY;
        for(int p = 0; p < query.numPoints(); p++)
        {
            BoundedSortedList<IndexDistPair> D_p = allPriorities.get(query.getPoint(p));
            synchronized(D_p)
            {
                if(D_p.size() == numNeighbors)//has enough neighbors to return a meaningful boun
                    bound_1 = max(bound_1, D_p.last().dist);
                else//can't bound
                {
                    bound_1 = Double.POSITIVE_INFINITY;
                    break;
                }
            }
        }
        if(Double.isInfinite(bound_1))//cant bound
            bound_1 = Double.POSITIVE_INFINITY;
        else//can bound, make it correct
            for(int c = 0; c < query.numChildren(); c++)
            {
                double B_nc = query_B_cache.getOrDefault(query.getChild(c), Double.POSITIVE_INFINITY);
//                if(Double.isInfinite(B_nc))//tighten by recursive search
//                    B_nc = computeKnnBound(query.getChild(c), numNeighbors, allPriorities, query_B_cache);
                bound_1 = max(bound_1, B_nc);
            }
        ///compute bound 2i. First set to infinity, and find min portion
        double bound_2i = Double.POSITIVE_INFINITY;
        for(int i = 0; i < query.numPoints(); i++)
        {
            int qi_indx = query.getPoint(i);
            BoundedSortedList<IndexDistPair> pqi = allPriorities.get(qi_indx);
            synchronized(pqi)
            {
                if(pqi.size() >= numNeighbors)
                    bound_2i = min(bound_2i, pqi.last().dist);
            }
        }
        //then add the remaining 2 terms, which are constant for a given Node Q. If no valid points, bound remains infinite
        bound_2i += query.furthestPointDistance() +  query.furthestDescendantDistance();
        //Compute 3rd bound
        double lambda_q = query.furthestDescendantDistance();
        double bound_3 = Double.POSITIVE_INFINITY;
        for(int c = 0; c < query.numChildren(); c++)
        {
            IndexNode n_c = query.getChild(c);
            double B_nc = query_B_cache.getOrDefault(n_c, Double.POSITIVE_INFINITY);
//            if(Double.isInfinite(B_nc))//tighten by recursive search
//                    B_nc = computeKnnBound(n_c, numNeighbors, allPriorities, query_B_cache);
            bound_3 = min(bound_3, B_nc + 2*(lambda_q-n_c.furthestDescendantDistance()));
        }
        IndexNode q_parrent = query.getParrent();
        double bound_4 = q_parrent == null ? Double.POSITIVE_INFINITY : query_B_cache.getOrDefault(q_parrent, Double.POSITIVE_INFINITY);
        final double bound_final = min(min(bound_1, bound_2i), min(bound_3, bound_4));
        
        //update cache with min value
        query_B_cache.compute(query, (IndexNode t, Double u) ->
        {
            if(u == null)
                u = Double.POSITIVE_INFINITY;
            return Math.min(u, bound_final);
        });
        
//        if(Double.isFinite(bound_3))
//            System.out.println(bound_3);
        return bound_final;
    }
    
    @Override
    default public void search(VectorCollection<V> VC, double r_min, double r_max, List<List<Integer>> neighbors, List<List<Double>> distances, boolean parallel )
    {
        if(!(VC instanceof DualTree))
        {
            VectorCollection.super.search(VC, r_min, r_max, neighbors, distances, parallel);
            return;
        }
        DualTree<V> Q = (DualTree<V>) VC;
            
        neighbors.clear();
        distances.clear();
        for(int i = 0; i < Q.size(); i++)
        {
            neighbors.add(new IntList());
            distances.add(new DoubleList());
        }
        
        ///For simplicity and fast calculations, lets combine acceleration caches into one view
        final List<Double> this_cache = this.getAccelerationCache();
        final List<Double> other_cache = Q.getAccelerationCache();
        
        final int N_r = this.size();
        final List<Double> wholeCache = this_cache == null ? null : ListUtils.mergedView(this_cache, other_cache);
        final List<Vec> allVecs = new ArrayList<>(N_r+Q.size());
        for(int i = 0; i < N_r; i++)
            allVecs.add(this.get(i));
        for(int i = 0; i < Q.size(); i++)
            allVecs.add(Q.get(i));
        
        DistanceMetric dm = getDistanceMetric();
        
        BaseCaseDT base = (int r_indx, int q_indx) ->
        {
            double d = dm.dist(r_indx, N_r+q_indx, allVecs, wholeCache);
            if(r_min <= d && d <= r_max)
            {
                synchronized(neighbors.get(q_indx))
                {
                    neighbors.get(q_indx).add(r_indx);
                    distances.get(q_indx).add(d);
                }
            }
            return d;
        };
        
        ScoreDT score = (IndexNode ref, IndexNode query) ->
        {
            double[] minMax = ref.minMaxDistance(query);
            double d_min = minMax[0];
            double d_max = minMax[1];
            if(d_min > r_max || d_max < r_min)//If min dist is greater than max-range, or max distance is greater than min-range, we can prune
                return Double.NaN;
            
            if(r_min < d_min && d_max < r_max)//Bound says ALL DECENDENTS BELONG, so lets do that! 
            {
                IntList r_dec = new IntList();
                for(Iterator<Integer> iter = ref.DescendantIterator(); iter.hasNext(); )
                    r_dec.add(iter.next());
                IntList q_dec = new IntList();
                for(Iterator<Integer> iter = query.DescendantIterator(); iter.hasNext(); )
                    q_dec.add(iter.next());
                for(int i : r_dec)
                {
                    for(int j : q_dec)
                    {
                        double d = dm.dist(i, N_r+j, allVecs, wholeCache);
                        synchronized(neighbors.get(j))
                        {
                            neighbors.get(j).add(i);
                            distances.get(j).add(d);
                        }
                    }
                }
                //Return NaN so that search stops, we added everyone!
                return Double.NaN;
            }
            
            return d_min;
        };
        
        //Range search dosn't benefit from improved search order. So use basic one and avoid extra overhead
        traverse(Q, base, score, false, parallel);
        
        //Now lets sort the returned lists
        for(int i = 0; i < neighbors.size(); i++)
        {
            IndexTable it = new IndexTable(distances.get(i));
            it.apply(distances.get(i));
            it.apply(neighbors.get(i));
        }
    }

    default void traverse(DualTree<V> Q, BaseCaseDT base, ScoreDT score, boolean improvedTraverse, boolean parallel)
    {
        IndexNode R_root = this.getRoot(), Q_root = Q.getRoot();
        
        if(!this.getRoot().allPointsInLeaves())//warp the roots so that we can use the same traversal for all implementations
        {
            R_root = new SelfAsChildNode<>(this.getRoot());
            Q_root = new SelfAsChildNode<>(Q.getRoot());
        }
        
        if(parallel)
            ForkJoinPool.commonPool().invoke(new DualTreeTraversalAction(R_root, Q_root, base, score, improvedTraverse));
        else
            dual_depth_first(R_root, Q_root, base, score, improvedTraverse);
    }
    
    /**
     * This class is used as a helper class to deal with Dual Trees which may
     * contain points in branching nodes. The dual tree traversal assumes all
     * points belong in leaf nodes. This fixes that by wraping an IndexNode to
     * behave as if all points owned within a branch really belong to a special
     * extra "self" child.
     *
     * @param <N>
     */
    class SelfAsChildNode<N extends IndexNode<N>> implements IndexNode<SelfAsChildNode<N>>
    {
        public boolean asLeaf;
        N wrapping;

        public SelfAsChildNode(N wrapping)
        {
            this.wrapping = wrapping;
            asLeaf = !wrapping.hasChildren();
        }

        public SelfAsChildNode(boolean asLeaf, N wrapping)
        {
            this.asLeaf = asLeaf;
            this.wrapping = wrapping;
        }
        
        
        @Override
        public double furthestPointDistance()
        {
            if(!asLeaf)//Not acting as a leaf, so you don't have children!
                return 0;
            //else, return the answer
            return wrapping.furthestPointDistance();
        }

        @Override
        public double furthestDescendantDistance()
        {
            if(asLeaf)
                return wrapping.furthestPointDistance();
            else
                return wrapping.furthestDescendantDistance();
        }

        @Override
        public int numChildren()
        {
            if(asLeaf)
                return 0;
            else
                return wrapping.numChildren() + 1;//+1 for self child
        }

        @Override
        public IndexNode getChild(int indx)
        {
            if(indx == wrapping.numChildren())
                return new SelfAsChildNode(true, wrapping);
            //else, return base children
            return new SelfAsChildNode(wrapping.getChild(indx));
        }

        @Override
        public Vec getVec(int indx)
        {
            return wrapping.getVec(indx);
        }

        @Override
        public int numPoints()
        {
            if(asLeaf)
                return wrapping.numPoints();
            else
                return 0;
        }

        @Override
        public int getPoint(int indx)
        {
            if(asLeaf)
                return wrapping.getPoint(indx);
            else//we can't have children if we aren't a leaf node!
                throw new IndexOutOfBoundsException("Leaf node does not have any children");
        }

        @Override
        public SelfAsChildNode<N> getParrent()
        {
            if(asLeaf)
                if(wrapping.hasChildren())//we are a branch node and acting as a leaf, so parrent its our non-leaf self
                    return new SelfAsChildNode<>(false, wrapping);
            //we are true leaf node, parrent is just parrent 
            // OR
            // we are not a leaf node, parrent is again just parrent
            N parrent = wrapping.getParrent();
            if(parrent == null)
                return null;
            return new SelfAsChildNode<>(false, parrent);
        }

        @Override
        public double minNodeDistance(SelfAsChildNode<N> other)
        {
            return wrapping.minNodeDistance(other.wrapping);
        }

        @Override
        public double maxNodeDistance(SelfAsChildNode<N> other)
        {
            return wrapping.maxNodeDistance(other.wrapping);
        }

        @Override
        public double minNodeDistance(int other)
        {
            return wrapping.minNodeDistance(other);
        }

        @Override
        public boolean equals(Object obj)
        {
            if(obj instanceof SelfAsChildNode)
            {
                SelfAsChildNode other = (SelfAsChildNode) obj;
                if(this.asLeaf == other.asLeaf)
                    return this.wrapping.equals(other.wrapping);
            }
            return false;
        }

        @Override
        public int hashCode()
        {
            int hash = 5;
            hash = 71 * hash + (this.asLeaf ? 1 : 0);
            if(this.wrapping == null)
                System.out.println();
            hash = 71 * hash + this.wrapping.hashCode();
            return hash;
        }

        @Override
        public double[] minMaxDistance(SelfAsChildNode<N> other)
        {
            return wrapping.minMaxDistance(other.wrapping);
        }
        
    }
    
    static final double COMP_SCORE = -1;
    
    public static void dual_depth_first(IndexNode n_r, IndexNode n_q, BaseCaseDT base, ScoreDT score, boolean improvedSearch)
    {
        //Algo 10 in Thesis

        //3: {Perform base cases for points in node combination.}
        for(int i = 0; i < n_r.numPoints(); i++)
            for(int j = 0; j < n_q.numPoints(); j++)
                base.base_case(n_r.getPoint(i), n_q.getPoint(j));
        
        //7: {Assemble list of combinations to recurse into.}
        //8: q←empty priority queue
        PriorityQueue<IndexTuple> q = new PriorityQueue<>();
        
        //9: if Nq andNr both have children then
        if(n_q.hasChildren() && n_r.hasChildren())
        {
            //the Algorithm 10 version. Simpler but not as efficent
            if(!improvedSearch)
            {
                for(int i = 0; i < n_r.numChildren(); i++)
                    for(int j = 0; j < n_q.numChildren(); j++)
                    {
                        IndexNode n_r_i = n_r.getChild(i);
                        IndexNode n_q_j = n_q.getChild(j);

                        double s = score.score(n_r_i, n_q_j, COMP_SCORE);
                        if(!Double.isNaN(s))
                            q.offer(new IndexTuple(n_r_i, n_q_j, s));
                    }
            }
            else //Below is the Algo 13 version. 
            {
                for(int c = 0; c < n_q.numChildren(); c++)
                {
                    IndexNode n_q_c = n_q.getChild(c);
                    List<IndexTuple> q_qc =new ArrayList<>();
                    boolean all_scores_same = true;
                    for(int i = 0; i < n_r.numChildren(); i++)
                    {
                        IndexNode n_r_i = n_r.getChild(i);
                        double s = score.score(n_r_i, n_q_c, COMP_SCORE);
                        //check if all scores have the same value
                        if(i > 0 && abs(q_qc.get(i-1).priority-s) < 1e-13)
                            all_scores_same = false;
                        q_qc.add(new IndexTuple(n_r_i, n_q_c, s));
                    }

                    if(all_scores_same)
                    {
                        double s = score.score(n_r, n_q_c, COMP_SCORE);
                        q.offer(new IndexTuple(n_r, n_q_c, s));
                    }
                    else
                        q.addAll(q_qc);
                }
            }
        }
        else if(n_q.hasChildren()) //implicitly n_r has not children if this check passes
        {
            for(int j = 0; j < n_q.numChildren(); j++)
            {
                IndexNode n_q_j = n_q.getChild(j);
                double s = score.score(n_r, n_q_j, COMP_SCORE);
                if (!Double.isNaN(s))
                    q.offer(new IndexTuple(n_r, n_q_j, s));
            }
        }
        else if(n_r.hasChildren())// implicitly n_q has no children if this check passes
        {
            for (int i = 0; i < n_r.numChildren(); i++)
            {
                IndexNode n_r_i = n_r.getChild(i);
                double s = score.score(n_r_i, n_q, COMP_SCORE);
                if (!Double.isNaN(s))
                    q.offer(new IndexTuple(n_r_i, n_q, s));
            }
        }
        
        
        //22: {Recurse into combinations with highest priority first.
        while(!q.isEmpty())
        {
            IndexTuple toProccess = q.poll();
            if(score instanceof ScoreDTLazy)//re-compute the score before we just go in
            {
                double s = score.score(toProccess.a, toProccess.b, toProccess.priority);
                if(Double.isNaN(s))//We might have a pruning op now
                    continue;//Good job!
            }
            dual_depth_first(toProccess.a, toProccess.b, base, score, improvedSearch);
        }
    }
    
    class DualTreeTraversalAction extends RecursiveAction implements Comparable<DualTreeTraversalAction>
    {
        IndexNode n_r;
        IndexNode n_q;
        BaseCaseDT base;
        ScoreDT score;
        boolean improvedSearch;
        double priority;

        public DualTreeTraversalAction(IndexNode n_r, IndexNode n_q, BaseCaseDT base, ScoreDT score, boolean improvedSearch)
        {
            this(n_r, n_q, base, score, improvedSearch, 0.0);
        }

        public DualTreeTraversalAction(IndexNode n_r, IndexNode n_q, BaseCaseDT base, ScoreDT score, boolean improvedSearch, double priority)
        {
            this.n_r = n_r;
            this.n_q = n_q;
            this.base = base;
            this.score = score;
            this.improvedSearch = improvedSearch;
            this.priority = priority;
        }
        
        

        @Override
        protected void compute()
        {
            /* 
             * B/c of fork-join framework, we can't do the ScoreDTLazy 
             * check before placing them into the execution que. So we will do 
             * them at the root no upon ourselves. We can do that b/c priority 
             * is the score for the pair of IndexNodes we are about to process!
             */
            
            if(score instanceof ScoreDTLazy)//re-compute the score before we do work
            {
                double s = score.score(n_r, n_q, priority);
                if(Double.isNaN(s))//We might have a pruning op now
                    return;//Good job! No more work to do
            }
            
            //Algo 10 in Thesis

            //3: {Perform base cases for points in node combination.}
            for(int i = 0; i < n_r.numPoints(); i++)
                for(int j = 0; j < n_q.numPoints(); j++)
                    base.base_case(n_r.getPoint(i), n_q.getPoint(j));

            //7: {Assemble list of combinations to recurse into.}
            //8: q←empty priority queue
            PriorityQueue<DualTreeTraversalAction> q = new PriorityQueue<>();

            //9: if Nq andNr both have children then
            if(n_q.hasChildren() && n_r.hasChildren())
            {
                //the Algorithm 10 version. Simpler but not as efficent
                if(!improvedSearch)
                {
                    for(int i = 0; i < n_r.numChildren(); i++)
                        for(int j = 0; j < n_q.numChildren(); j++)
                        {
                            IndexNode n_r_i = n_r.getChild(i);
                            IndexNode n_q_j = n_q.getChild(j);

                            double s = score.score(n_r_i, n_q_j, COMP_SCORE);
                            if(!Double.isNaN(s))
                                q.offer(new DualTreeTraversalAction(n_r_i, n_q_j, base, score, improvedSearch, s));
                        }
                }
                else //Below is the Algo 13 version. 
                {
                    for(int c = 0; c < n_q.numChildren(); c++)
                    {
                        IndexNode n_q_c = n_q.getChild(c);
                        List<DualTreeTraversalAction> q_qc =new ArrayList<>();
                        boolean all_scores_same = true;
                        for(int i = 0; i < n_r.numChildren(); i++)
                        {
                            IndexNode n_r_i = n_r.getChild(i);
                            double s = score.score(n_r_i, n_q_c, COMP_SCORE);
                            //check if all scores have the same value
                            if(i > 0 && abs(q_qc.get(i-1).priority-s) < 1e-13)
                                all_scores_same = false;
                            q_qc.add(new DualTreeTraversalAction(n_r_i, n_q_c, base, score, improvedSearch, s));
                        }

                        if(all_scores_same)
                        {
                            double s = score.score(n_r, n_q_c, COMP_SCORE);
                            q.offer(new DualTreeTraversalAction(n_r, n_q_c, base, score, improvedSearch, s));
                        }
                        else
                            q.addAll(q_qc);
                    }
                }
            }
            else if(n_q.hasChildren()) //implicitly n_r has not children if this check passes
            {
                for(int j = 0; j < n_q.numChildren(); j++)
                {
                    IndexNode n_q_j = n_q.getChild(j);
                    double s = score.score(n_r, n_q_j, COMP_SCORE);
                    if (!Double.isNaN(s))
                        q.offer(new DualTreeTraversalAction(n_r, n_q_j, base, score, improvedSearch, s));
                }
            }
            else if(n_r.hasChildren())// implicitly n_q has no children if this check passes
            {
                for (int i = 0; i < n_r.numChildren(); i++)
                {
                    IndexNode n_r_i = n_r.getChild(i);
                    double s = score.score(n_r_i, n_q, COMP_SCORE);
                    if (!Double.isNaN(s))
                        q.offer(new DualTreeTraversalAction(n_r_i, n_q, base, score, improvedSearch, s));
                }
            }


            //22: {Recurse into combinations with highest priority first.
            invokeAll(q);
        }

        @Override
        public int compareTo(DualTreeTraversalAction o)
        {
            return Double.compare(this.priority, o.priority);
        }
        
    }
}
