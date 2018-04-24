package jsat.linear.vectorcollection;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Random;
import java.util.Stack;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.BooleanList;
import jsat.utils.BoundedSortedList;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;
import jsat.utils.IntList;
import jsat.utils.ModifiableCountDownLatch;
import jsat.utils.Pair;
import jsat.utils.SimpleList;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 * Provides a simplified implementation of Vantage Point Trees, as described in 
 * "Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces" 
 * by Peter N. Yianilos 
 * <br>
 * VPTrees are more expensive to create, requiring O(n log n) distance computations. However,
 * they work well for high dimensional data sets, and provide O( log n ) query time for 
 * {@link #search(jsat.linear.Vec, int) }
 * 
 * 
 * @author Edward Raff
 * @param <V>
 */
public class SVPTree<V extends Vec> implements IncrementalCollection<V>, DualTree<V>
{

    private static final long serialVersionUID = -7271540108746353762L;
    private DistanceMetric dm;
    private List<Double> distCache;
    private List<V> allVecs;
    protected volatile TreeNode root;
    private int size;
    private int maxLeafSize = 5;

    @Override
    public IndexNode getRoot()
    {
        return root;
    }
    
    public SVPTree(List<V> list, DistanceMetric dm, boolean parallel)
    {
        build(parallel, list, dm);
    }
    
    
    public SVPTree(List<V> list, DistanceMetric dm)
    {
        this(list, dm, false);
    }
    
    public SVPTree()
    {
        this(new EuclideanDistance());
    }
    
    
    public SVPTree(DistanceMetric dm)
    {
        this.dm = dm;
        if(!dm.isSubadditive())
            throw new RuntimeException("VPTree only supports metrics that support the triangle inequality");
        this.size = 0;
        this.allVecs = new ArrayList<>();
        if(dm.supportsAcceleration())
            this.distCache = new DoubleList();
    }
    
    /**
     * Copy constructor
     * @param toClone the object to copy
     */
    protected SVPTree(SVPTree<V> toClone)
    {
        this.dm = toClone.dm.clone();
        this.root = cloneChangeContext(toClone.root);
        this.size = toClone.size;
        this.maxLeafSize = toClone.maxLeafSize;
        if(toClone.allVecs != null)
            this.allVecs = new ArrayList<>(toClone.allVecs);
        if(toClone.distCache != null)
            this.distCache = new DoubleList(toClone.distCache);
    }
    
    @Override
    public List<Double> getAccelerationCache()
    {
        return distCache;
    }

    @Override
    public double dist(int self_index, int other_index, DualTree<V> other)
    {
        return DualTree.super.dist(self_index, other_index, other); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public void build(boolean parallel, List<V> list, DistanceMetric dm)
    {
        setDistanceMetric(dm);
        if(!dm.isSubadditive())
            throw new RuntimeException("VPTree only supports metrics that support the triangle inequality");
        Random rand = RandomUtil.getRandom();

        this.size = list.size();
        this.allVecs = list;
        distCache = dm.getAccelerationCache(allVecs, parallel);
        //Use simple list so both halves can be modified simultaniously
        List<Pair<Double, Integer>> tmpList = new SimpleList<>(list.size());
        for(int i = 0; i < allVecs.size(); i++)
            tmpList.add(new Pair<>(-1.0, i));
        if(!parallel)
            this.root = makeVPTree(tmpList);
        else
        {
            ExecutorService threadpool = ParallelUtils.getNewExecutor(parallel);
            ModifiableCountDownLatch mcdl = new ModifiableCountDownLatch(1);
            this.root = makeVPTree(tmpList, threadpool, mcdl);
            mcdl.countDown();
            try
            {
                mcdl.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(SVPTree.class.getName()).log(Level.SEVERE, null, ex);
                System.err.println("Falling back to single threaded VPTree constructor");
                tmpList.clear();
                for(int i = 0; i < list.size(); i++)
                    tmpList.add(new Pair<>(-1.0, i));
                this.root = makeVPTree(tmpList);
            }
            finally
            {
                threadpool.shutdownNow();
            }
        }
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

    private TreeNode cloneChangeContext(TreeNode toClone)
    {
        if (toClone != null)
            if (toClone instanceof jsat.linear.vectorcollection.SVPTree.VPLeaf)
                return new VPLeaf((VPLeaf) toClone);
            else
                return new VPNode((VPNode) toClone);
        return null;
    }
    
    @Override
    public int size()
    {
        return size;
    }

    @Override
    public V get(int indx)
    {
        return allVecs.get(indx);
    }

    @Override
    public void insert(V x)
    {
        int indx = size++;
        allVecs.add(x);
        if(distCache != null)
            distCache.addAll(dm.getQueryInfo(x));
        
        
        if(root == null)
        {
            ArrayList<Pair<Double, Integer>> list = new ArrayList<>();
            list.add(new Pair<>(Double.MAX_VALUE, indx));
            root = new VPLeaf(list);
            return;
        }
        ///else, do a normal insert
        root.insert(indx, Double.MAX_VALUE);
        if(root instanceof jsat.linear.vectorcollection.SVPTree.VPLeaf)//is root a leaf?
        {
            VPLeaf leaf = (VPLeaf) root;
            if(leaf.points.size() > maxLeafSize*maxLeafSize)//check to expand
            {
                //hacky, but works
                int orig_leaf_isze = maxLeafSize;
                maxLeafSize = maxLeafSize*maxLeafSize;//call normal construct with adjusted leaf size to stop expansion
                ArrayList<Pair<Double, Integer>> S = new ArrayList<>();
                for(int i = 0; i < leaf.points.size(); i++)
                    S.add(new Pair<>(Double.MAX_VALUE, leaf.points.getI(i)));
                root = makeVPTree(S);
                maxLeafSize = orig_leaf_isze;//restor
            }
        }
        //else, normal non-leaf root insert handles expansion when needed
    }

    @Override
    public void search(Vec query, double range, List<Integer> neighbors, List<Double> distances)
    {
        List<Double> qi = dm.getQueryInfo(query);
        root.searchRange(VecPaired.extractTrueVec(query), range, neighbors, distances, 0.0, qi);
        
        IndexTable it = new IndexTable(distances);
        it.apply(neighbors);
        it.apply(distances);
    }

    @Override
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances)
    {
        BoundedSortedList<IndexDistPair> boundedList= new BoundedSortedList<>(numNeighbors, numNeighbors);

        List<Double> qi = dm.getQueryInfo(query);
        root.searchKNN(VecPaired.extractTrueVec(query), numNeighbors, boundedList, 0.0, qi);
        
        for(IndexDistPair pm : boundedList)
        {
            neighbors.add(pm.getIndex());
            distances.add(pm.getDist());
        }
    }
    
    /**
     * Computes the distances to the vantage point, 
     * Sorts the list by distance to the vantage point, 
     * finds the splitting index, and sets up the parent node. 
     * @param S the list
     * @param node the parent node
     * @return the index that was used to split on. 
     */
    private int sortSplitSet(final List<Pair<Double, Integer>> S, final VPNode node)
    {
        for (Pair<Double, Integer> S1 : S)
            S1.setFirstItem(dm.dist(node.p, S1.getSecondItem(), allVecs, distCache)); //Each point gets its distance to the vantage point
        Collections.sort(S, (Pair<Double, Integer> o1, Pair<Double, Integer> o2) -> Double.compare(o1.getFirstItem(), o2.getFirstItem()));
        int splitIndex = splitListIndex(S); 
        node.left_low = S.get(0).getFirstItem();
        node.left_high = S.get(splitIndex).getFirstItem();
        node.right_low = S.get(splitIndex+1).getFirstItem();
        node.right_high = S.get(S.size()-1).getFirstItem();
        return splitIndex;
    }

    
    /**
     * Determines which index to use as the splitting index for the VP radius 
     * @param S the non empty list of elements 
     * @return the index that should be used to split on [0, index] belonging to the left, and (index, S.size() ) belonging to the right. 
     */
    protected int splitListIndex(List<Pair<Double, Integer>> S)
    {
        return S.size()/2;
    }

    /**
     * Returns the maximum leaf node size. Leaf nodes are used to reduce inefficiency of splitting small lists. 
     * If a sublist will fit into a leaf node, a leaf node will be created instead of splitting. This is the 
     * maximum number of points that may be used to construct a leaf node. 
     * 
     * @return the maximum leaf node size in the tree
     */
    public int getMaxLeafSize()
    {
        return maxLeafSize;
    }

    /**
     * Sets  the maximum leaf node size. Leaf nodes are used to reduce inefficiency of splitting small lists. 
     * If a sublist will fit into a leaf node, a leaf node will be created instead of splitting. This is the 
     * maximum number of points that may be used to construct a leaf node. <br>
     * The minimum leaf size is 5 for implementation reasons. If a value less than 5 is given, 5 will be used isntead. 
     * 
     * @param maxLeafSize the new maximum leaf node size. 
     */
    public void setMaxLeafSize(int maxLeafSize)
    {
        this.maxLeafSize = Math.max(5, maxLeafSize);
    }
    
    
    //The probability match is used to store and sort by median distances. 
    private TreeNode makeVPTree(List<Pair<Double, Integer>> S)
    {
        if(S.isEmpty())
            return null;
        else if(S.size() <= maxLeafSize)
        {
            VPLeaf leaf = new VPLeaf(S);
            return leaf;
        }
        
        int vpIndex = selectVantagePointIndex(S);
        final VPNode node = new VPNode(S.get(vpIndex).getSecondItem());
        node.parent_dist = S.get(vpIndex).getFirstItem();
        
        int splitIndex = sortSplitSet(S, node);
        
        /*
         * Re use the list and let it get altered. We must compute the right side first. 
         * If we altered the left side, the median would move left, and the right side 
         * would get thrown off or require aditonal book keeping. 
         */
        node.right = makeVPTree(S.subList(splitIndex+1, S.size()));
        if(node.right != null)
            node.right.parent = node;
        node.left  = makeVPTree(S.subList(0, splitIndex+1));
        if(node.left != null)
            node.left.parent = node;
        
        return node;
    }
    
    private TreeNode makeVPTree(final List<Pair<Double, Integer>> S, final ExecutorService threadpool, final ModifiableCountDownLatch mcdl)
    {
        if(S.isEmpty())
        {
            return null;
        }
        else if(S.size() <= maxLeafSize)
        {
            VPLeaf leaf = new VPLeaf(S);
            return leaf;
        }
        
        int vpIndex = selectVantagePointIndex(S);
        final VPNode node = new VPNode(S.get(vpIndex).getSecondItem());
        node.parent_dist = S.get(vpIndex).getFirstItem();
        
        int splitIndex = sortSplitSet(S, node);
        
        
        //Start 2 threads, but only 1 of them is "new" 
        mcdl.countUp();
        

        final List<Pair<Double, Integer>> rightS = S.subList(splitIndex+1, S.size());
        final List<Pair<Double, Integer>> leftS = S.subList(0, splitIndex+1);
        
        threadpool.submit(() -> 
        {
            node.right = makeVPTree(rightS, threadpool, mcdl);
            if(node.right != null)
                node.right.parent = node;
            mcdl.countDown();
        });
        node.left  = makeVPTree(leftS, threadpool, mcdl);
        if(node.left != null)
            node.left.parent = node;

        return node;
    }
    
    
    private int selectVantagePointIndex(List<Pair<Double, Integer>> S)
    {
        int vpIndex;
        vpIndex = RandomUtil.getLocalRandom().nextInt(S.size());
        return vpIndex;
    }

    @Override
    public SVPTree<V> clone()
    {
        return new SVPTree<>(this);
    }
    
    private abstract class TreeNode implements Cloneable, Serializable, IndexNode
    {
        VPNode parent;
        
        /**
         * Inserts the given data point into the tree structure. The vector
         * should have already been added to {@link #allVecs}.
         *
         * @param x_indx the index of the vector to insert
         * @param dist_to_parent the distance of the current point to the parent
         * node's vantage point. May be {@link Double#MAX_VALUE} if root node.
         */
        public abstract void insert(int x_indx, double dist_to_parent);
        /**
         * Performs a KNN query on this node.
         * 
         * @param query the query vector
         * @param k the number of neighbors to consider
         * @param list the storage location on the nearest neighbors
         * @param x the distance between this node's parent vantage point to the query vector.
         * Though not all nodes will use this value, the leaf nodes will - so it should always be given.
         * Initial calls from the root node may choose to us zero. 
         * @param qi the value of qi
         */
        
        public abstract void searchKNN(Vec query, int k, BoundedSortedList<IndexDistPair> list, double x, List<Double> qi);
        
        /**
         * Performs a range query on this node
         * 
         * @param query the query vector
         * @param range the maximal distance a point can be from the query point
         * to be added to the return list
         * @param neighbors the storage location on the data points within the
         * range of the query vector
         * @param distances the value of distances to each neighbor
         * @param x the distance between this node's parent vantage point to the
         * query vector. Though not all nodes will use this value, the leaf
         * nodes will - so it should always be given. Initial calls from the
         * root node may choose to us zero.
         * @param qi the value of qi
         */
        
        public abstract void searchRange(Vec query, double range, List<Integer> neighbors, List<Double> distances, double x, List<Double> qi);
        
        public abstract boolean isLeaf();
        
        @Override
        public abstract TreeNode clone();
        
    }
    
    private class VPNode extends TreeNode
    {
        int p;
        double left_low, left_high, right_low, right_high;
        TreeNode right, left;
        double parent_dist;

        public VPNode(int p)
        {
            this.p = p;
        }

        public VPNode(VPNode toCopy)
        {
            this(toCopy.p);
            this.left_low  = toCopy.left_low;
            this.left_high = toCopy.left_high;
            this.right_low = toCopy.right_low;
            this.right_high = toCopy.right_high;
            this.left = cloneChangeContext(toCopy.left);
            this.right = cloneChangeContext(toCopy.right);
        }

        @Override
        public boolean isLeaf()
        {
            return false;
        }
        
        @Override
        public void insert(int x_indx, double dist_to_parent)
        {
            double dist = dm.dist(p, x_indx, allVecs, distCache);
            TreeNode child;
            if(dist*2 < left_high+right_low)
            {
                left_high = Math.max(left_high, dist);
                left_low = Math.min(left_low, dist);
                child = left = maybeExpandChild(left);
            }
            else
            {
                right_high = Math.max(right_high, dist);
                right_low = Math.min(right_low, dist);
                child = right = maybeExpandChild(right);
            }
            child.insert(x_indx, dist);
        }

        /**
         * If the given node is a leaf node, this will check if it is time to
         * expand the leaf, and return the new non-leaf child. Otherwise, it
         * will return the original node.
         *
         * @param child the child node to potentially expand
         * @return the node that should be used as the child node
         */
        private TreeNode maybeExpandChild(TreeNode child)
        {
            //have to use fully qualified path b/c non-static child member
            if(child instanceof jsat.linear.vectorcollection.SVPTree.VPLeaf)
            {
                IntList childs_children = ((VPLeaf) child).points;
                if(childs_children.size() <= maxLeafSize*maxLeafSize)
                    return child;
                List<Pair<Double, Integer>> S = new ArrayList<>(childs_children.size());
                for(int indx : childs_children)
                    S.add(new Pair<>(Double.MAX_VALUE, indx));//double value will be set apprioatly later
                int vpIndex = selectVantagePointIndex(S);
                
                final VPNode node = new VPNode(S.get(vpIndex).getSecondItem());
                node.parent_dist = S.get(vpIndex).getFirstItem();
                node.parent = ((VPLeaf) child).parent;
                
                //move VP to front, its self dist is zero and we dont want it used in computing bounds. 
                Collections.swap(S, 0, vpIndex);
                int splitIndex = sortSplitSet(S.subList(1, S.size()), node)+1;//ofset by 1 b/c we sckipped the VP, which was moved to the front
                
                node.right = new VPLeaf(S.subList(splitIndex+1, S.size()));
                node.right.parent = node;
                node.left = new VPLeaf(S.subList(1, splitIndex+1));
                node.left.parent = node;
                return node;
            }
            else
                return child;
        }
        
        private boolean searchInLeft(double x, double tau)
        {
            if(left == null)
                return false;
            return left_low-tau <= x && x <= left_high+tau;
        }
        
        private boolean searchInRight(double x, double tau)
        {
            if(right == null)
                return false;
            return right_low-tau <= x && x <= right_high+tau;
        }
        
        @Override
        public void searchKNN(Vec query, int k, BoundedSortedList<IndexDistPair> list, double x, List<Double> qi)
        {
            Deque<VPNode> curNode_stack = new ArrayDeque<VPNode>();
            
            DoubleList distToParrent_stack = new DoubleList();
            BooleanList search_left_stack = new BooleanList();
            
            curNode_stack.add(this);
            
            while(!curNode_stack.isEmpty())
            {
                if(curNode_stack.size() > search_left_stack.size())//we are decending the tree
                {
                    VPNode node = curNode_stack.peek();
                    x = dm.dist(node.p, query, qi, allVecs, distCache);
                    distToParrent_stack.push(x);
                    if(list.size() < k || x < list.get(k-1).getDist())
                        list.add(new IndexDistPair(node.p, x));
                    double tau = list.get(list.size()-1).getDist();
                    double middle = (node.left_high+node.right_low)*0.5;
                    boolean leftFirst =  x < middle;

                    //If we search left now, on pop we need to search right
                    search_left_stack.add(!leftFirst);
                    if(leftFirst)
                    {
                        if(node.searchInLeft(x, tau) || list.size() < k)
                        {
                            if(node.left.isLeaf())
                                node.left.searchKNN(query, k, list, x, qi);
                            else
                            {
                                curNode_stack.push((VPNode) node.left);
                                continue;//CurNode will now have a size 1 greater than the search_left_stach
                            }
                        }
                    }
                    else
                    {
                        if(node.searchInRight(x, tau) || list.size() < k)
                        {
                            if(node.right.isLeaf())
                                node.right.searchKNN(query, k, list, x, qi);
                            else
                            {
                                curNode_stack.push((VPNode) node.right);
                                continue;//CurNode will now have a size 1 greater than the search_left_stach
                            }
                        }
                    }
                }
                else//we are poping up the search patch
                {
                    VPNode node = curNode_stack.pop();//pop, we are defintly done with this node after
                    x = distToParrent_stack.pop();
                    double tau = list.get(list.size()-1).getDist();
                    Boolean finishLeft = search_left_stack.pop();
                    
                    
                    if(finishLeft)
                    {
                        if(node.searchInLeft(x, tau) || list.size() < k)
                        {
                            if(node.left.isLeaf())
                                node.left.searchKNN(query, k, list, x, qi);
                            else
                            {
                                curNode_stack.push((VPNode) node.left);
                                continue;//CurNode will now have a size 1 greater than the search_left_stach
                            }
                        }
                        //else, branch was pruned. Loop back and keep popping
                    }
                    else
                    {
                        if(node.searchInRight(x, tau) || list.size() < k)
                        {
                            if(node.right.isLeaf())
                                node.right.searchKNN(query, k, list, x, qi);
                            else
                            {
                                curNode_stack.push((VPNode) node.right);
                                continue;//CurNode will now have a size 1 greater than the search_left_stach
                            }
                        }
                        //else, branch was pruned. Loop back and keep popping
                    }
                }
                
            }
            
        }
        
        public void searchKNN_recurse(Vec query, int k, BoundedSortedList<IndexDistPair> list, double x, List<Double> qi)
        {
            x = dm.dist(p, query, qi, allVecs, distCache);
            if(list.size() < k || x < list.get(k-1).getDist())
                list.add(new IndexDistPair(this.p, x));
            double tau = list.get(list.size()-1).getDist();
            double middle = (this.left_high+this.right_low)*0.5;
            
//            if(this.left instanceof VPNode && this.right in)
            

            if( x < middle)
            {
                if(searchInLeft(x, tau) || list.size() < k)
                    this.left.searchKNN(query, k, list, x, qi);
                tau = list.get(list.size()-1).getDist();
                if(searchInRight(x, tau) || list.size() < k)
                    this.right.searchKNN(query, k, list, x, qi);
            }
            else
            {
                if(searchInRight(x, tau) || list.size() < k)
                    this.right.searchKNN(query, k, list, x, qi);
                tau = list.get(list.size()-1).getDist();
                if(searchInLeft(x, tau) || list.size() < k)
                    this.left.searchKNN(query, k, list, x, qi);
            }
        }

        @Override
        public void searchRange(Vec query, double range, List<Integer> neighbors, List<Double> distances, double x, List<Double> qi)
        {
            x = dm.dist(this.p, query, qi, allVecs, distCache);
            if(x <= range)
            {
                neighbors.add(this.p);
                distances.add(x);
            }

            if (searchInLeft(x, range))
                this.left.searchRange(query, range, neighbors, distances, x, qi);
            if (searchInRight(x, range))
                this.right.searchRange(query, range, neighbors, distances, x, qi);
        }

        @Override
        public TreeNode clone()
        {
            return new VPNode(this);
        }

        @Override
        public VPNode getParrent()
        {
            return parent;
        }

        @Override
        public double maxNodeDistance(IndexNode other)
        {
            if(other instanceof jsat.linear.vectorcollection.SVPTree.VPNode)
            {
                jsat.linear.vectorcollection.SVPTree.VPNode o = (jsat.linear.vectorcollection.SVPTree.VPNode) other;
                Vec ov = o.getVec(o.p);
                List<Double> qi = dm.getQueryInfo(ov);
                return dm.dist(this.p, ov, qi, allVecs, distCache) + this.right_high + o.right_high;
            }
            else
            {
//                VPLeaf c = (jsat.linear.vectorcollection.SVPTree.VPLeaf) other;
//                VPNode o = c.getParrent();
//                Vec ov = o.getVec(o.p);
//                List<Double> qi = dm.getQueryInfo(ov);
//                return dm.dist(this.p, ov, qi, allVecs, distCache) + this.right_high + c.getParentDistance();
                return Double.POSITIVE_INFINITY;
            }
        }

        @Override
        public double minNodeDistance(IndexNode other)
        {
            if(other instanceof jsat.linear.vectorcollection.SVPTree.VPNode)
            {
                jsat.linear.vectorcollection.SVPTree.VPNode o = (jsat.linear.vectorcollection.SVPTree.VPNode) other;

                Vec ov = o.getVec(o.p);
                List<Double> qi = dm.getQueryInfo(ov);
                return dm.dist(this.p, ov, qi, allVecs, distCache) - this.right_high - o.right_high;
            }
            else
            {
//                VPLeaf c = (jsat.linear.vectorcollection.SVPTree.VPLeaf) other;
//                VPNode o = c.getParrent();
//                Vec ov = o.getVec(o.p);
//                List<Double> qi = dm.getQueryInfo(ov);
//                return dm.dist(this.p, ov, qi, allVecs, distCache) - this.right_high - c.getParentDistance();
                return 0;
            }
        }

        @Override
        public double[] minMaxDistance(IndexNode other)
        {
            if(other instanceof jsat.linear.vectorcollection.SVPTree.VPNode)
            {
                jsat.linear.vectorcollection.SVPTree.VPNode o = (jsat.linear.vectorcollection.SVPTree.VPNode) other;

                Vec ov = o.getVec(o.p);
                List<Double> qi = dm.getQueryInfo(ov);
                double d = dm.dist(this.p, ov, qi, allVecs, distCache);
                return new double[]
                {
                    d - this.right_high - o.right_high,
                    d + this.right_high + o.right_high
                };
            }
            else
            {
                return new double[]{0, Double.POSITIVE_INFINITY};
            }
        }

        @Override
        public double minNodeDistance(int other)
        {
            return dm.dist(p, other, allVecs, distCache) - right_low;
        }

        @Override
        public double getParentDistance()
        {
            return parent_dist;
        }

        @Override
        public double furthestPointDistance()
        {
            return 0;//WE have one point which is the centroid, so distance is 0. 
        }

        @Override
        public double furthestDescendantDistance()
        {
            return right_high;
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
                    return left;
                case 1:
                    return right;
                default:
                    throw new IndexOutOfBoundsException();
            }
        }

        @Override
        public Vec getVec(int indx)
        {
            return get(indx);
        }

        @Override
        public int numPoints()
        {
            return 0;
        }

        @Override
        public int getPoint(int indx)
        {
            throw new IndexOutOfBoundsException("VPNode has only one point, can't access index " + indx);
        }
        
    }
    
    private class VPLeaf extends TreeNode
    {
        /**
         * The index in {@link #allVecs} for each data point stored in this Leaf node
         */
        IntList points;
        /**
         * The distance of each point in this leaf to the parent node we came from. 
         */
        DoubleList bounds;
        
        public VPLeaf(List<Pair<Double, Integer>> points)
        {
            this.points = new IntList(points.size());
            this.bounds = new DoubleList(points.size());
            for(int i = 0; i < points.size(); i++)
            {
                this.points.add(points.get(i).getSecondItem());
                this.bounds.add(points.get(i).getFirstItem());
            }
        }
        
        public VPLeaf(VPLeaf toCopy)
        {
            this.bounds = new DoubleList(toCopy.bounds);
            this.points = new IntList(toCopy.points);
        }

        @Override
        public void insert(int x_indx, double dist_to_parent)
        {
            this.points.add(x_indx);
            this.bounds.add(dist_to_parent);
        }

        @Override
        public void searchKNN(Vec query, int k, BoundedSortedList<IndexDistPair> list, double x, List<Double> qi)
        {
            double dist = -1;
            
            //The zero check, for the case that the leaf is the ONLY node, x will be passed as 0.0 <= Max value will be true 
            double tau = list.isEmpty() ? Double.MAX_VALUE : list.get(list.size()-1).getDist();
            for (int i = 0; i < points.size(); i++)
            {
                int point_i = points.getI(i);
                double bound_i = bounds.getD(i);
                if (list.size() < k)
                {
                    
                    list.add(new IndexDistPair(point_i, dm.dist(point_i, query, qi, allVecs, distCache)));
                    tau = list.get(list.size() - 1).getDist();
                }
                else if (bound_i - tau <= x && x <= bound_i + tau)//Bound check agains the distance to our parrent node, provided by x
                    if ((dist = dm.dist(point_i, query, qi, allVecs, distCache)) < tau)
                    {
                        list.add(new IndexDistPair(point_i, dist));
                        tau = list.get(list.size() - 1).getDist();
                    }
            }
        }

        @Override
        public void searchRange(Vec query, double range, List<Integer> neighbors, List<Double> distances, double x, List<Double> qi)
        {
            double dist = Double.MAX_VALUE;
            
            for (int i = 0; i < points.size(); i++)
            {
                int point_i = points.getI(i);
                double bound_i = bounds.getD(i);
                if (bound_i - range <= x && x <= bound_i + range)//Bound check agains the distance to our parrent node, provided by x
                    if ((dist = dm.dist(point_i, query, qi, allVecs, distCache)) < range)
                    {
                        neighbors.add(point_i);
                        distances.add(dist);
                    }
            }
        }

        @Override
        public boolean isLeaf()
        {
            return true;
        }

        @Override
        public TreeNode clone()
        {
            return new VPLeaf(this);
        }

        @Override
        public VPNode getParrent()
        {
            return parent;
        }

        @Override
        public double maxNodeDistance(IndexNode other)
        {
            return Double.POSITIVE_INFINITY;
//            if(other instanceof jsat.linear.vectorcollection.SVPTree.VPNode)
//            {
//                return other.maxNodeDistance(this);
//            }
//            else
//            {
//                VPLeaf c = (jsat.linear.vectorcollection.SVPTree.VPLeaf) other;
//                VPNode o = c.getParrent();
//                Vec ov = o.getVec(o.p);
//                List<Double> qi = dm.getQueryInfo(ov);
//                return dm.dist(this.getParrent().p, ov, qi, allVecs, distCache) + this.getParentDistance() + c.getParentDistance();
//            }
        }
        

        @Override
        public double minNodeDistance(IndexNode other)
        {
            return 0;
//            if(other instanceof jsat.linear.vectorcollection.SVPTree.VPNode)
//            {
//                return other.minNodeDistance(this);
//            }
//            else
//            {
//                VPLeaf c = (jsat.linear.vectorcollection.SVPTree.VPLeaf) other;
//                VPNode o = c.getParrent();
//                Vec ov = o.getVec(o.p);
//                List<Double> qi = dm.getQueryInfo(ov);
//                return dm.dist(this.getParrent().p, ov, qi, allVecs, distCache) - this.getParentDistance() - c.getParentDistance();
//            }
        }

        @Override
        public double minNodeDistance(int other)
        {
            //Leaf node, return a value that makes caller go brute-force
            return 0.0;
        }

        @Override
        public double getParentDistance()
        {
            return bounds.stream().mapToDouble(d->d).max().orElse(Double.POSITIVE_INFINITY);
        }

        @Override
        public double furthestPointDistance()
        {
            return bounds.stream().mapToDouble(d->d).max().orElse(Double.POSITIVE_INFINITY);
        }

        @Override
        public double furthestDescendantDistance()
        {
            return bounds.stream().mapToDouble(d->d).max().orElse(Double.POSITIVE_INFINITY);
        }

        @Override
        public int numChildren()
        {
            return 0;
        }

        @Override
        public IndexNode getChild(int indx)
        {
            throw new IndexOutOfBoundsException("Leaf nodes have no children");
        }

        @Override
        public Vec getVec(int indx)
        {
            return get(indx);
        }

        @Override
        public int numPoints()
        {
            return points.size();
        }

        @Override
        public int getPoint(int indx)
        {
            return points.getI(indx);
        }
    }
}
