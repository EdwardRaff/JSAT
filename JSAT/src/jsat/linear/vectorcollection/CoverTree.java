/*
 * Copyright (C) 2017 Edward Raff
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
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.math.FastMath;
import jsat.utils.BoundedSortedList;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;
import jsat.utils.IntList;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import jsat.utils.Pair;
import jsat.utils.concurrent.AtomicDouble;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.XORWOW;

/**
 * This class implements the Cover-tree algorithm for answering nearest neighbor
 * queries. In particular, it uses the "Simplified Cover-tree" algorithm. <br>
 * Note, this implementation does not yet support parallel construction. 
 * <br>
 * 
 * See:
 * <ul>
 * <li>Beygelzimer, A., Kakade, S., & Langford, J. (2006). Cover trees for
 * nearest neighbor. In International Conference on Machine Learning (pp.
 * 97–104). New York: ACM. Retrieved from
 * <a href="http://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/covertree.pdf">here</a></li>
 * <li>Izbicki, M., & Shelton, C. R. (2015). Faster Cover Trees. In Proceedings
 * of the Thirty-Second International Conference on Machine Learning (Vol.
 * 37).</li>
 * </ul>
 *
 * @author Edward Raff
 */
public final class CoverTree<V extends Vec> implements IncrementalCollection<V>
{
    private DistanceMetric dm;
    private List<V> vecs;
    private List<Double> accell_cache = null;
    private TreeNode root = null;
    private boolean maxDistDirty = false;
//    private boolean nearest_ancestor = false;
    private boolean looseBounds = false;
    private static final int min_pow_map = -120;
    private static final int max_pow_map = 1000;
    private static final float[] pow_map = new float[max_pow_map-min_pow_map];
    private static final double base = 1.3;
    private static final double log2_base = Math.log(base)/Math.log(2);
    static
    {
        for(int pow_indx = min_pow_map; pow_indx < max_pow_map; pow_indx++)
            pow_map[pow_indx-min_pow_map] = (float) Math.pow(base, pow_indx);
    }
    
    private static double pow(int expo)
    {
        if(expo >= min_pow_map && expo < max_pow_map)
            return pow_map[expo-min_pow_map];
        else
            return Math.pow(base, expo);
    }

    public CoverTree(DistanceMetric dm)
    {
        this.dm = dm;
        vecs = new ArrayList<>();
    }

    public CoverTree(List<V> source, DistanceMetric dm)    
    {
        this(source, dm, false);
    }
    
    public CoverTree(List<V> source, DistanceMetric dm, boolean parallel)
    {
        this(source, dm, parallel, false);
    }
    
    public CoverTree(List<V> source, DistanceMetric dm, boolean parallel, boolean looseBounds)
    {
        setLooseBounds(looseBounds);
        build(parallel, source, dm);
    }

    public CoverTree(CoverTree<V> toCopy)
    {
        this.dm = toCopy.dm.clone();
        this.looseBounds = toCopy.looseBounds;
        this.vecs = new ArrayList<>(toCopy.vecs);
        if(toCopy.accell_cache != null)
            this.accell_cache = new DoubleList(toCopy.accell_cache);
        if(toCopy.root != null)
            this.root = new TreeNode(toCopy.root);
    }
    
    @Override
    public List<Double> getAccelerationCache()
    {
        return accell_cache;
    }

    @Override
    public void build(boolean parallel, List<V> collection, DistanceMetric dm)
    {
        this.dm = dm;
        setLooseBounds(looseBounds);
        this.vecs = new ArrayList<>(collection);
        this.accell_cache = dm.getAccelerationCache(vecs, parallel);
        //Cover Tree is sensative to insertion order, so lets make sure its random
        IntList order = new IntList(this.vecs.size());
        ListUtils.addRange(order, 0, this.vecs.size(), 1);
        
//        Set<Integer> S = getSet(parallel);
//        S.addAll(order);
//        int p = S.stream().findAny().get();
//        S.remove(p);
//        
//        this.root = new TreeNode(p);
//        construct(parallel, root, S, getSet(parallel), Integer.MAX_VALUE);
        
        
        Collections.shuffle(order, new XORWOW(54321));
        int pos = 0;
        for(int i : order)
        {
            root = simpleInsert(root, i);
            pos++;
//            System.out.println("\t" + pos + " vs " + this.root.magnitude());
        }
//        System.out.println(this.vecs.size() + " vs " + this.root.magnitude());
        if(!this.looseBounds)//pre-compute all max-dist bounds used during search
        {
            this.root.maxdist();
            Iterator<TreeNode> iter = this.root.descendants();
            while(iter.hasNext())
            {
                iter.next().maxdist();
            }
        }
    }
    
    /**
     * 
     * @param p the point p 
     * @return 
     */
    private Set<Integer> construct(boolean parallel, TreeNode p, Set<Integer> near, Set<Integer> far, int level)
    {
        if(near.isEmpty())
            return far;
        
        Set<Integer> workingNearSet;
        if(level == Integer.MAX_VALUE)//We need to figure out the correct level and do the split at once to avoid duplicate work
        {
            int[] points = near.stream().mapToInt(i->i).toArray();
            double[] dists = new double[points.length];
            
            
            double maxDist = ParallelUtils.run(parallel, points.length, (start, end)->
            {
                double max_ = 0;
                for(int i = start; i < end; i++)
                {
                    dists[i] = dm.dist(p.vec_indx, points[i], vecs, accell_cache);
                    max_ = Math.max(max_, dists[i]);
                }
                
                return max_;
            }, (a, b) -> Math.max(a, b));
            
            
            level = p.level = (int) Math.ceil(FastMath.log2(maxDist)/log2_base+1e-4);
            p.maxdist = maxDist;
            double r_split = pow(p.level-1);
            
            near.clear();
            Set<Integer> newNear = getSet(parallel);
            Set<Integer> newFar = getSet(parallel);
            ParallelUtils.run(parallel, points.length, (start, end)->
            {
                
                for(int i = start; i < end; i++)
                {
                    double d_i = dists[i];
                    if(d_i <= r_split)
                        newNear.add(points[i]);
                    else if (d_i < 2 * r_split)
                        newFar.add(points[i]);
                    else
                        near.add(points[i]);
                }
            });
            workingNearSet = construct(parallel, p, newNear, newFar, p.level-1);
        }
        else
        {
            Pair<Set<Integer>, Set<Integer>> pairRet = split(parallel, p.vec_indx, pow(level-1), near);
            workingNearSet = construct(parallel, p, pairRet.getFirstItem(), pairRet.getSecondItem(), level-1);
        }
        
        while(!workingNearSet.isEmpty())
        {
            //(i) pick q in NEAR
            int q_indx = workingNearSet.stream().findAny().get();
            workingNearSet.remove(q_indx);
            TreeNode q = new TreeNode(q_indx, level-1);
            //(ii) <CHILD, UNUSED> = Construct (q, SPLIT(d(q, ·), 2^(i−1),NEAR,FAR), i−1)
            Set<Integer> unused = construct(parallel, q, workingNearSet, far, level-1);
            //(iii) add CHILD to Children(pi) 
            p.addChild(q);
            //(iv) let <NEW-NEAR, NEW-FAR> =SPLIT(d(p, ·), 2^i,UNUSED)
            Pair<Set<Integer>, Set<Integer>> newPiar = split(parallel, p.vec_indx, pow(level), unused);
            Set<Integer> newNear = newPiar.getFirstItem();
            Set<Integer> newFar = newPiar.getSecondItem();
            //(v) add NEW-FAR to FAR, and NEW-NEAR to NEAR.
            far.addAll(newFar);
            workingNearSet.addAll(newNear);
        }
        
        return far;
    }
    
    private Pair<Set<Integer>, Set<Integer>> split(boolean parallel, int p, double r, Set<Integer>... S)
    {
        Set<Integer> newNear = getSet(parallel);
        Set<Integer> newFar = getSet(parallel);
        
        for(Set<Integer> S_i : S)
        {
            int[] toRemove = ParallelUtils.streamP(S_i.stream(), parallel).mapToInt(i->
            {
                double d_i = dm.dist(p, i, vecs, accell_cache);
                
                if(d_i <= r)
                    newNear.add(i);
                else if(d_i < 2*r)
                    newFar.add(i);
                else
                    return -1;//-1 will be 'removed' from the set S_i. but -1 isn't a valid index. So no impact
                
                return i;
            }).distinct().toArray();
            
            S_i.removeAll(IntList.view(toRemove));
        }
        
        return new Pair<>(newNear, newFar);
    }

    private Set<Integer> getSet(boolean parallel)
    {
        Set<Integer> newNear;
        if(parallel)
            newNear = ConcurrentHashMap.newKeySet();
        else
            newNear = new IntSet();
        return newNear;
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

    public void setLooseBounds(boolean looseBounds)
    {
        this.looseBounds = looseBounds;
    }

    @Override
    public void search(Vec query, double range, List<Integer> neighbors, List<Double> distances)
    {
        neighbors.clear();
        distances.clear();
        
        this.root.findNN(range, query, dm.getQueryInfo(query), neighbors, distances, -1.0);
        
        IndexTable it = new IndexTable(distances);
        it.apply(distances);
        it.apply(neighbors);
    }

    @Override
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances)
    {
//        if(maxDistDirty && ! looseBounds)
//        {
//            this.root.invalidateMaxDist();
//            maxDistDirty = false;
//        }
        BoundedSortedList<IndexDistPair> bsl = new BoundedSortedList<>(numNeighbors);
        this.root.findNN(numNeighbors, query, dm.getQueryInfo(query), bsl);
        neighbors.clear();
        distances.clear();
        for(IndexDistPair a : bsl)
        {
            neighbors.add(a.getIndex());
            distances.add(a.getDist());
        }
    }
    
    @Override
    public int size()
    {
        return vecs.size();
    }

    @Override
    public V get(int indx)
    {
        return vecs.get(indx);
    }

    @Override
    public CoverTree<V> clone()
    {
        return new CoverTree<>(this);
    }
    
    protected void simpleInsert(V x)
    {
        int x_indx = this.vecs.size();
        this.vecs.add(x);
        if(this.accell_cache == null && dm.supportsAcceleration())
            this.accell_cache = new DoubleList();
        if(this.accell_cache != null)
            this.accell_cache.addAll(dm.getQueryInfo(x));
        
        if(this.root == null)//start the tree
        {
            this.root = new TreeNode(x_indx);
//            this.root.setLevel(0);
        }
        else//actually insert
            this.root = simpleInsert(root, x_indx);
    }
    
    protected TreeNode simpleInsert(TreeNode p, int x_indx)
    {
        if(this.root == null)//start the tree
        {
            this.root = new TreeNode(x_indx);
//            this.root.setLevel(0);
            return this.root;
        }
        
        double p_x_dist = p.dist(x_indx);
        if(p_x_dist > p.covdist())//line 1
        {
            /*
             * If the insetion point x has a distance that is SUPER far away, 
             * the below bound may never hold. Thus, lets detect loops and short
             * circuit
             */
            final int start_indx = p.vec_indx;
            if(p_x_dist - pow(p.level+1) < base*p.covdist())//if this is true, the condition will be true for p AND ALL CHILDREN OF P
                while(p_x_dist > base*p.covdist() && !p.isLeaf())//line 2
                {
                    //3: remove any leaf q from p 
                    TreeNode q;
                    q = p.removeAnyLeaf();
                    //4: p' = tree with root q and p as only child
                    TreeNode p_prime = q;
                    p_prime.addChild(p);
                    p_prime.fixLevel();
                    p = p_prime;//5: p = p'
                    p_x_dist = p.dist(x_indx);
                    if(p.vec_indx == start_indx)//WE HAVE DONE THIS BEFORE
                        break;
                }
            //6: return tree with x as root and p as only child
            TreeNode X = new TreeNode(x_indx);
            X.addChild(p);
            X.fixLevel();
            return X;
        }
        //return insert_(p,x)
        return simpleInsert_(p, x_indx);
    }
    
    /**
     * prerequisites: d(p,x) ≤ covdist(p)
     * @param p
     * @param x_indx 
     * @return  
     */
    protected TreeNode simpleInsert_(TreeNode p, int x_indx)
    {
//        if(nearest_ancestor)
//        {
//            double[] dist_to_x = new double[p.numChildren()];
//            for(int i = 0; i < dist_to_x.length; i++)
//                dist_to_x[i] = p.getChild(i).dist(x_indx);
//            IndexTable it = new IndexTable(dist_to_x);
//            
//            for(int order = 0; order < p.numChildren(); order++) //Line 1:
//            {
//                int q_indx = it.index(order);
//                TreeNode q = p.getChild(q_indx);
//                if(q.dist(x_indx) <= q.covdist()) //line 2: d(q,x)≤covdist(q)
//                {
//                    //3: q' ← insert_(q,x)
//                    TreeNode q_prime = simpleInsert_(q, x_indx);
//                    //4: p' ← p with child q replaced with q'
//                    p.replaceChild(q_indx, q_prime);
//                    //5: return p'
//                    return p;
//                }
//            }
//            //6: return rebalance(p, x)
//            return rebalance(p, x_indx);
//        }
//        else
        {
            for(int q_indx = 0; q_indx < p.numChildren(); q_indx++) //Line 1:
            {
                TreeNode q = p.getChild(q_indx);
                if(q.dist(x_indx) <= q.covdist()) //line 2: d(q,x)≤covdist(q)
                {
                    //3: q' ← insert_(q,x)
                    TreeNode q_prime = simpleInsert_(q, x_indx);
                    //4: p' ← p with child q replaced with q'
                    p.replaceChild(q_indx, q_prime);
                    //5: return p'
                    return p;
                }
            }
            //6: return p with x added as a child
            p.addChild(new TreeNode(x_indx, p.level-1));
            return p;
        }
    }
    
    
    @Override
    public void insert(V x)
    {
//        maxDistDirty = true;
        simpleInsert(x);
    }
    
    private class TreeNode implements Cloneable, Serializable
    {
        TreeNode parent = null;
        int level;
        int vec_indx;
        DoubleList children_dists;
        List<TreeNode> children;
        boolean is_sorted = true;
        double maxdist = -1;

        public TreeNode(int vec_indx)
        {
            this(vec_indx, -110);
        }
        
        public TreeNode(int vec_indx, int level)
        {
            this.vec_indx = vec_indx;
            this.level = level;
            children = new ArrayList<>();
            children_dists = new DoubleList();
        }

        /**
         * Copy constructor. Will not copy the parent node. But will copy children nodes and set their parents appropriately. 
         * @param toCopy 
         */
        public TreeNode(TreeNode toCopy)
        {
            this.level = toCopy.level;
            this.vec_indx = toCopy.vec_indx;
            if(toCopy.children != null)
            {
                this.children = new ArrayList<>(toCopy.children.size());
                this.children_dists = new DoubleList(toCopy.children_dists);
                for(TreeNode childToCopy : toCopy.children)
                {
                    TreeNode child = new TreeNode(childToCopy);
                    child.parent = this;
                    this.children.add(child);
                }
            }
            this.is_sorted = toCopy.is_sorted;
            this.maxdist = toCopy.maxdist;
        }

        @Override
        protected TreeNode clone()
        {
            return new TreeNode(this);
        }
        
        public void invalidateMaxDist()
        {
            this.maxdist = -1;
            for(TreeNode c : children)
                c.invalidateMaxDist();
        }
        
        public void invalParentMaxdist()
        {
            this.maxdist = -2;
            if(this.parent != null)
                this.parent.invalParentMaxdist();
        }
        
        public void findNN(int k, Vec query, List<Double> x_qi, BoundedSortedList<IndexDistPair> knn)
        {
            Stack<TreeNode> toEval_stack = new Stack<>();
            DoubleList dist_to_q_stack = new DoubleList();
            {//Quick, add root info to stack for search & prime search Q
                double p_x_dist = this.dist(query, x_qi);
                dist_to_q_stack.push(p_x_dist);
                toEval_stack.push(this);
            }
            
            //Search loop
            while(!toEval_stack.isEmpty())
            {
                TreeNode p = toEval_stack.pop();
                double p_to_q_dist = dist_to_q_stack.pop();
                knn.add(new IndexDistPair(p.vec_indx, p_to_q_dist));
                
                double[] child_query_dist = new double[p.numChildren()];
                for(int child_indx = 0; child_indx < p.numChildren(); child_indx++)//compute dists and add to knn while we are at it
                {
                    TreeNode q = p.getChild(child_indx);
                    child_query_dist[child_indx] = q.dist(query, x_qi);
                }
                
                //get them in sorted order
                IndexTable it = new IndexTable(child_query_dist);
                for(int i_oder = it.length()-1; i_oder >= 0; i_oder--)//reverse order so stack goes in sorted order
                {
                    final int i = it.index(i_oder);
                    TreeNode q = p.getChild(i);
                    
                    //4:  if d(y,x)>d(y,q)−maxdist(q) then
                    if(knn.size() < k || knn.last().getDist() > child_query_dist[i] - q.maxdist())
                    {//Add to the search Q
                        toEval_stack.push(q);
                        dist_to_q_stack.push(child_query_dist[i]);
                    }
                }
            }
        }
        
        //This is the old search code, new code (above) avoids recursion and makes explicit stack
        private void findNN_recurse(int k, Vec x, List<Double> x_qi, BoundedSortedList<IndexDistPair> knn, double my_dist_to_x)
        {
            TreeNode p = this;
            
            double p_x_dist;
            if(my_dist_to_x < 0)
            {
                p_x_dist = p.dist(x, x_qi);
                
            }
            else
                p_x_dist = my_dist_to_x;
            knn.add(new IndexDistPair(p.vec_indx, p_x_dist));
            //1: if d(p,x)<d(y,x) then, handled implicitly by knn object
//            if(knn.size() < k || p_x_dist < knn.last().getDist())
//            knn.add(new ProbailityMatch<V>(p_x_dist, vecs.get(p.vec_indx)));//2: y <= p
            //3: for each child q of p sorted by *distance to x* do
            double[] q_x_dist = new double[p.numChildren()];
            for(int q_indx = 0; q_indx < p.numChildren(); q_indx++)//compute dists and add to knn while we are at it
            {
                TreeNode q = p.getChild(q_indx);
                q_x_dist[q_indx] = q.dist(x, x_qi);
                //DO NOT ADD DISTANCE TO KNN YET, we will do it recursively
                //need to avoid it so bound check below will work propertly
                //and otherwise we would double count
//                knn.add(new ProbailityMatch<V>(q_x_dist[q_indx], vecs.get(q.vec_indx)));
//                q.findNN(k, x, x_qi, knn);
            }
            //get them in sorted order
            IndexTable it = new IndexTable(q_x_dist);
            for(int i_oder = 0; i_oder < it.length(); i_oder++)
            {
                final int i = it.index(i_oder);
                TreeNode q = p.getChild(i);
//                knn.add(new ProbailityMatch<V>(q_x_dist[i], vecs.get(q.vec_indx)));
                //4:  if d(y,x)>d(y,q)−maxdist(q) then
//                if(knn.size() < k || knn.last().getDist() > q.dist(y_vec, dm.getQueryInfo(y_vec)) - q.maxdist())
                if(knn.size() < k || knn.last().getDist() > q_x_dist[i] - q.maxdist())
                    q.findNN_recurse(k, x, x_qi, knn, q_x_dist[i]);//Line 5:
//                else if(q.isLeaf())
//                {
//                    knn.add(new ProbailityMatch<V>(q.dist(x, x_qi), vecs.get(q.vec_indx)));
//                }
            }
        }
        
        public void findNN(double radius, Vec x, List<Double> x_qi, List<Integer> neighbors, List<Double> distances, double my_dist_to_x)
        {
            TreeNode p = this;
            
            double p_x_dist;
            if(my_dist_to_x < 0)
            {
                p_x_dist = p.dist(x, x_qi);
            }
            else
                p_x_dist = my_dist_to_x;
            if(p_x_dist <= radius)
            {
                neighbors.add(p.vec_indx);
                distances.add(p_x_dist);
            }
            //3: for each child q of p , no need to sort b/c radius search
            double[] q_x_dist = new double[p.numChildren()];
            for(int q_indx = 0; q_indx < p.numChildren(); q_indx++)//compute dists and add to knn while we are at it
            {
                TreeNode q = p.getChild(q_indx);
                q_x_dist[q_indx] = q.dist(x, x_qi);
                //DO NOT ADD DISTANCE TO KNN YET, we will do it on recursion
            }
            //get them in sorted order
            for(int i = 0; i < q_x_dist.length; i++)
            {
                TreeNode q = p.getChild(i);
                //4:  if d(y,x)>d(y,q)−maxdist(q) then
                if(radius > q_x_dist[i] - q.maxdist())
                    q.findNN(radius, x, x_qi, neighbors, distances, q_x_dist[i]);//Line 5:
            }
        }
        
        public int magnitude()
        {
            int count = 1;
            for(int i = 0; i < numChildren(); i++)
                count += getChild(i).magnitude();
            return count;
        }
        
        public boolean isLeaf()
        {
            return this.children == null || this.children.isEmpty();
        }
        
        public int numChildren()
        {
            return this.children.size();
        }
        
        public TreeNode getChild(int indx)
        {
            return this.children.get(indx);
        }
        
        public void addChild(TreeNode child)
        {
//            int new_level = Math.max(this.level, child.level+1);
            double dist_to_c = this.dist(child.vec_indx);
//            int insert_indx = Collections.binarySearch(children_dists, dist_to_c);
//            if(insert_indx < 0)//no exact match, convert to insertion index
//                insert_indx = -insert_indx-1;
            int insert_indx = this.children.size();
            this.children.add(insert_indx, child);
            this.children_dists.add(insert_indx, dist_to_c);
//            child.setLevel(new_level-1);
//            this.setLevel(new_level);
            this.fixChildrenLevel();
//            this.maxdist = -1;//no logner valid, so clear it
            this.invalParentMaxdist();
        }
        
        public void replaceChild(int orig_index, TreeNode child)
        {
            
            double dist_to_c = this.dist(child.vec_indx);

            this.children.set(orig_index, child);
            this.children_dists.set(orig_index, dist_to_c);
//            child.setLevel(new_level-1);
//            this.setLevel(new_level);
//            this.maxdist = -1;//no logner valid, so clear it
            this.fixChildrenLevel();
            this.invalParentMaxdist();
            
//            this.children.remove(orig_index);
//            this.children_dists.removeD(orig_index);
//            this.addChild(child);
            
            
            
//            int new_level = Math.max(this.level, child.level+1);
//            this.children.set(orig_index, child);
//            this.children_dists.set(orig_index, this.dist(child.vec_indx));
//            child.setLevel(new_level-1);
//            this.setLevel(new_level);
        }
        
        public void removeChild(int orig_index)
        {
            this.children.remove(orig_index);
            this.children_dists.remove(orig_index);
//            this.fixChildrenLevel();
            this.invalParentMaxdist();
        }
        
        /**
         * Removes a descendant of this node that is a leaf node. 
         * @return the descendant that was removed
         */
        public TreeNode removeAnyLeaf()
        {
            if(this.isLeaf())
                throw new RuntimeException("BUG: node has no children to rmeove");
            //lets just grab the furthest child? 

            TreeNode child = children.get(children.size()-1);
            if(child.isLeaf())
            {
                child.invalParentMaxdist();
                children.remove(children.size()-1);
                children_dists.remove(children_dists.size()-1);
                return child;
            }
            else//need to remove one of child's descentants to get a leaf
            {
                return child.removeAnyLeaf();
            }
        }
        
        public double dist(TreeNode q)
        {
            return dm.dist(this.vec_indx, q.vec_indx, vecs, accell_cache);
        }
        
        public double dist(int x_indx)
        {
            return dm.dist(this.vec_indx, x_indx, vecs, accell_cache);
        }
        
        public double dist(Vec x, List<Double> qi)
        {
            return dm.dist(this.vec_indx, x, qi, vecs, accell_cache);
        }

        public void setLevel(int level)
        {
//            if(this.level == level)
//                return;//levels are already set correctly
            this.level = level;
//            for(TreeNode q : children)
//                q.setLevel(level-1);
        }
        
        public void fixLevel()
        {
//            double maxDist = Math.pow(1.3, -110);
            double maxDist = pow(-110);
            for(int i = 0; i < numChildren(); i++)
                maxDist = Math.max(maxDist, this.children_dists.getD(i));
//            this.level = (int) Math.ceil(Math.log(maxDist)/Math.log(1.3));
            this.level = (int) Math.ceil(FastMath.log2(maxDist)/log2_base+1e-4);
            fixChildrenLevel();
        }
        
        public void fixChildrenLevel()
        {
            for(int i = 0; i < numChildren(); i++)
            {
                TreeNode c = getChild(i);
                if(this.level-1 != c.level)
                {
                    c.level = this.level-1;
                    c.fixChildrenLevel();
                }
            }
        }
        
        public double covdist()
        {
//            return Math.pow(1.3, level);
            return pow(level);
        }
        
        public double sepdist()
        {
//            return Math.pow(1.3, level-1);
            return pow(level-1);
        }

        private double maxdist()
        {
            if(isLeaf())
                return 0;
            if(looseBounds)
                return pow(level+1);
            if (this.maxdist >= 0)
                return maxdist;
            //else, maxdist = -1, indicating we need to compute it
            Stack<TreeNode> toGetChildrenFrom = new Stack<>();
            toGetChildrenFrom.add(this);
            
            while(!toGetChildrenFrom.empty())
            {
                TreeNode runner = toGetChildrenFrom.pop();
                
                for(int q_indx = 0; q_indx < runner.numChildren(); q_indx++)
                {
                    TreeNode q = runner.getChild(q_indx);
                    maxdist = Math.max(maxdist, this.dist(q.vec_indx));//TODO can optimize for first set of childern, we already have that
                    toGetChildrenFrom.add(q);
                }
            }
            
            return maxdist;
        }
        
        private Iterator<TreeNode> descendants()
        {
            final Stack<TreeNode> toIterate = new Stack<>();
            toIterate.addAll(children);
            Iterator<TreeNode> iter = new Iterator<TreeNode>()
            {
                @Override
                public boolean hasNext()
                {
                    return !toIterate.isEmpty();
                }

                @Override
                public TreeNode next()
                {
                    TreeNode next = toIterate.pop();
                    toIterate.addAll(next.children);
                    return next;
                }

                @Override
                public void remove()
                {
                    throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
                }
            };
            return iter;
        }
    }
}
