package jsat.linear.vectorcollection;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.VecPairedComparable;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.BoundedSortedList;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;
import jsat.utils.IntList;
import jsat.utils.ModifiableCountDownLatch;
import jsat.utils.Pair;
import jsat.utils.ProbailityMatch;
import jsat.utils.SimpleList;
import jsat.utils.random.RandomUtil;

/**
 * Provides an implementation of Vantage Point Trees, as described in 
 * "Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces" 
 * by Peter N. Yianilos 
 * <br>
 * VPTrees are more expensive to create, requiring O(n log n) distance computations. However,
 * they work well for high dimensional data sets, and provide O( log n ) query time for 
 * {@link #search(jsat.linear.Vec, int) }
 * <br> 
 * Note: In the original paper, the VP-tree is detailed, and then enhanced to the VPs-tree, 
 * and the VPsb-tree, which each add additional optimizations. This implementation is equivalent
 * to the VPsb-tree presented in the original paper. 
 * 
 * @author Edward Raff
 */
public class VPTree<V extends Vec> implements IncrementalCollection<V>
{

    private static final long serialVersionUID = -7271540108746353762L;
    private DistanceMetric dm;
    private List<Double> distCache;
    private List<V> allVecs;
    private Random rand;
    private int sampleSize;
    private int searchIterations;
    private volatile TreeNode root;
    private VPSelection vpSelection;
    private int size;
    private int maxLeafSize = 5;

    public enum VPSelection
    {
        /**
         * Uses the sampling method described in the original paper
         */
        Sampling, 
        /**
         * Randomly selects a new point to be the Vantage Point
         */
        Random
    }

    public VPTree(List<V> list, DistanceMetric dm, VPSelection vpSelection, Random rand, int sampleSize, int searchIterations, ExecutorService threadpool)
    {
        this.dm = dm;
        if(!dm.isSubadditive())
            throw new RuntimeException("VPTree only supports metrics that support the triangle inequality");
        this.rand = rand;
        this.sampleSize = sampleSize;
        this.searchIterations = searchIterations;
        this.size = list.size();
        this.vpSelection = vpSelection;
        this.allVecs = list;
        if(threadpool == null || threadpool instanceof FakeExecutor)
            distCache = dm.getAccelerationCache(allVecs);
        else
            distCache = dm.getAccelerationCache(allVecs, threadpool);
        //Use simple list so both halves can be modified simultaniously
        List<Pair<Double, Integer>> tmpList = new SimpleList<Pair<Double, Integer>>(list.size());
        for(int i = 0; i < allVecs.size(); i++)
            tmpList.add(new Pair<Double, Integer>(-1.0, i));
        if(threadpool == null)
            this.root = makeVPTree(tmpList);
        else
        {
            ModifiableCountDownLatch mcdl = new ModifiableCountDownLatch(1);
            this.root = makeVPTree(tmpList, threadpool, mcdl);
            mcdl.countDown();
            try
            {
                mcdl.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(VPTree.class.getName()).log(Level.SEVERE, null, ex);
                System.err.println("Falling back to single threaded VPTree constructor");
                tmpList.clear();
                for(int i = 0; i < list.size(); i++)
                    tmpList.add(new Pair<Double, Integer>(-1.0, i));
                this.root = makeVPTree(tmpList);
            }
        }
    }
    
    public VPTree(List<V> list, DistanceMetric dm, VPSelection vpSelection, Random rand, int sampleSize, int searchIterations)
    {
        this(list, dm, vpSelection, rand, sampleSize, searchIterations, null);
    }

    public VPTree(List<V> list, DistanceMetric dm, VPSelection vpSelection)
    {
        this(list, dm, vpSelection, RandomUtil.getRandom(), 80, 40);
    }
    
    public VPTree(List<V> list, DistanceMetric dm)
    {
        this(list, dm, VPSelection.Random);
    }
    
    public VPTree(DistanceMetric dm)
    {
        this.dm = dm;
        if(!dm.isSubadditive())
            throw new RuntimeException("VPTree only supports metrics that support the triangle inequality");
        this.rand = RandomUtil.getRandom();
        this.sampleSize = 80;
        this.searchIterations = 40;
        this.size = 0;
        this.vpSelection = VPSelection.Random;
        this.allVecs = new ArrayList<V>();
        if(dm.supportsAcceleration())
            this.distCache = new DoubleList();
    }
    
    /**
     * Copy constructor
     * @param toClone the object to copy
     */
    protected VPTree(VPTree<V> toClone)
    {
        this.dm = toClone.dm.clone();
        this.rand = toClone.rand == null ? null : new Random(toClone.rand.nextInt());
        this.sampleSize = toClone.sampleSize;
        this.searchIterations = toClone.searchIterations;
        this.root = cloneChangeContext(toClone.root);
        this.vpSelection = toClone.vpSelection;
        this.size = toClone.size;
        this.maxLeafSize = toClone.maxLeafSize;
        if(toClone.allVecs != null)
            this.allVecs = new ArrayList<V>(toClone.allVecs);
        if(toClone.distCache != null)
            this.distCache = new DoubleList(toClone.distCache);
    }

    private TreeNode cloneChangeContext(TreeNode toClone)
    {
        if (toClone != null)
            if (toClone instanceof jsat.linear.vectorcollection.VPTree.VPLeaf)
                return new VPLeaf((VPLeaf) toClone);
            else
                return new VPNode((VPNode) toClone);
        return null;
    }

    /**
     * no-arg constructor for serialization
     */
    protected VPTree()
    {
    }
     
    
    
    @Override
    public int size()
    {
        return size;
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
            ArrayList<Pair<Double, Integer>> list = new ArrayList<Pair<Double, Integer>>();
            list.add(new Pair<Double, Integer>(Double.MAX_VALUE, indx));
            root = new VPLeaf(list);
            return;
        }
        ///else, do a normal insert
        root.insert(indx, Double.MAX_VALUE);
        if(root instanceof jsat.linear.vectorcollection.VPTree.VPLeaf)//is root a leaf?
        {
            VPLeaf leaf = (VPLeaf) root;
            if(leaf.points.size() > maxLeafSize*maxLeafSize)//check to expand
            {
                //hacky, but works
                int orig_leaf_isze = maxLeafSize;
                maxLeafSize = maxLeafSize*maxLeafSize;//call normal construct with adjusted leaf size to stop expansion
                ArrayList<Pair<Double, Integer>> S = new ArrayList<Pair<Double, Integer>>();
                for(int i = 0; i < leaf.points.size(); i++)
                    S.add(new Pair<Double, Integer>(Double.MAX_VALUE, leaf.points.getI(i)));
                root = makeVPTree(S);
                maxLeafSize = orig_leaf_isze;//restor
            }
        }
        //else, normal non-leaf root insert handles expansion when needed
    }
        
    @Override
    @SuppressWarnings("unchecked")
    public List<? extends VecPaired<V, Double>> search(Vec query, double range)
    {
        if(range <= 0)
            throw new RuntimeException("Range must be a positive number");
        List<VecPairedComparable<V, Double>> returnList = new ArrayList<VecPairedComparable<V, Double>>();
        
        List<Double> qi = dm.getQueryInfo(query);
        root.searchRange(VecPaired.extractTrueVec(query), range, (List)returnList, 0.0, qi);
        
        Collections.sort(returnList);
        
        return returnList;
    }
    
    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, int neighbors)
    {
        BoundedSortedList<ProbailityMatch<V>> boundedList= new BoundedSortedList<ProbailityMatch<V>>(neighbors, neighbors);

        List<Double> qi = dm.getQueryInfo(query);
        root.searchKNN(VecPaired.extractTrueVec(query), neighbors, boundedList, 0.0, qi);
        
        List<VecPaired<V, Double>> list = new ArrayList<VecPaired<V, Double>>(boundedList.size());
        for(ProbailityMatch<V> pm : boundedList)
            list.add(new VecPaired<V, Double>(pm.getMatch(), pm.getProbability()));
        return list;
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
        Collections.sort(S, new Comparator<Pair<Double, Integer>>() 
        {
            @Override
            public int compare(Pair<Double, Integer> o1, Pair<Double, Integer> o2)
            {
                return Double.compare(o1.getFirstItem(), o2.getFirstItem());
            }
        });
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
        
        //move VP to front, its self dist is zero and we dont want it used in computing bounds. 
        Collections.swap(S, 0, vpIndex);
        int splitIndex = sortSplitSet(S.subList(1, S.size()), node)+1;//ofset by 1 b/c we sckipped the VP, which was moved to the front
        
        /*
         * Re use the list and let it get altered. We must compute the right side first. 
         * If we altered the left side, the median would move left, and the right side 
         * would get thrown off or require aditonal book keeping. 
         */
        node.right = makeVPTree(S.subList(splitIndex+1, S.size()));
        node.left  = makeVPTree(S.subList(1, splitIndex+1));
        
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
        
        //move VP to front, its self dist is zero and we dont want it used in computing bounds. 
        Collections.swap(S, 0, vpIndex);
        int splitIndex = sortSplitSet(S.subList(1, S.size()), node)+1;//ofset by 1 b/c we sckipped the VP, which was moved to the front
        
        
        //Start 2 threads, but only 1 of them is "new" 
        mcdl.countUp();
        

        final List<Pair<Double, Integer>> rightS = S.subList(splitIndex+1, S.size());
        final List<Pair<Double, Integer>> leftS = S.subList(1, splitIndex+1);
        
        threadpool.submit(new Runnable() 
        {
            @Override
            public void run()
            {
                node.right = makeVPTree(rightS, threadpool, mcdl);
                mcdl.countDown();
            }
        });
        node.left  = makeVPTree(leftS, threadpool, mcdl);

        return node;
    }
    
    
    private int selectVantagePointIndex(List<Pair<Double, Integer>> S)
    {
        int vpIndex;
        if (vpSelection == VPSelection.Random)
            vpIndex = rand.nextInt(S.size());
        else//Sampling
        {
            List<Integer> samples = new IntList(sampleSize);
            if (sampleSize <= S.size())
                for (int i = 0; i < sampleSize; i++)
                    samples.add(S.get(i).getSecondItem());
            else
                for (int i = 0; i < sampleSize; i++)
                    samples.add(S.get(rand.nextInt(S.size())).getSecondItem());

            double[] distances = new double[sampleSize];

            int bestVP = -1;
            double bestSpread = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < Math.min(searchIterations, S.size()); i++)
            {
                //When low on samples, just brute force!
                int candIndx = searchIterations <= S.size() ? i : rand.nextInt(S.size());
                int candV = S.get(candIndx).getSecondItem();

                for (int j = 0; j < samples.size(); j++)
                    distances[j] = dm.dist(candV, samples.get(j), allVecs, distCache);

                Arrays.sort(distances);
                double median = distances[distances.length / 2];
                double spread = 0;
                for (double distance : distances)
                    spread += Math.abs(distance - median);
                if (spread > bestSpread)
                {
                    bestSpread = spread;
                    bestVP = candIndx;
                }
            }

            vpIndex = bestVP;
        }
        return vpIndex;
    }
    
    /**
     * Determines what point from the data set will become a vantage point, and removes it from the list
     * @param S the set to select a vantage point from
     * @return the index of thevantage point removed from the set
     */
    private int selectVantagePoint(List<Pair<Double, Integer>> S)
    {
        int vpIndex = selectVantagePointIndex(S);
        
        return S.get(vpIndex).getSecondItem();
    }

    @Override
    public VPTree<V> clone()
    {
        return new VPTree<V>(this);
    }
    
    private abstract class TreeNode implements Cloneable, Serializable
    {
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
        
        public abstract void searchKNN(Vec query, int k, BoundedSortedList<ProbailityMatch<V>> list, double x, List<Double> qi);
        
        /**
         * Performs a range query on this node
         * 
         * @param query the query vector
         * @param range the maximal distance a point can be from the query point to be added to the return list
         * @param list the storage location on the data points within the range of the query vector
         * @param x the distance between this node's parent vantage point to the query vector.
         * Though not all nodes will use this value, the leaf nodes will - so it should always be given.
         * Initial calls from the root node may choose to us zero. 
         * @param qi the value of qi
         */
        
        public abstract void searchRange(Vec query, double range, List<VecPaired<V, Double>> list, double x, List<Double> qi);
        
        @Override
        public abstract TreeNode clone();
    }
    
    private class VPNode extends TreeNode
    {
        int p;
        double left_low, left_high, right_low, right_high;
        TreeNode right, left;

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
            if(child instanceof jsat.linear.vectorcollection.VPTree.VPLeaf)
            {
                IntList childs_children = ((VPLeaf) child).points;
                if(childs_children.size() <= maxLeafSize*maxLeafSize)
                    return child;
                List<Pair<Double, Integer>> S = new ArrayList<Pair<Double, Integer>>(childs_children.size());
                for(int indx : childs_children)
                    S.add(new Pair<Double, Integer>(Double.MAX_VALUE, indx));//double value will be set apprioatly later
                int vpIndex = selectVantagePointIndex(S);
                
                final VPNode node = new VPNode(S.get(vpIndex).getSecondItem());
                
                //move VP to front, its self dist is zero and we dont want it used in computing bounds. 
                Collections.swap(S, 0, vpIndex);
                int splitIndex = sortSplitSet(S.subList(1, S.size()), node)+1;//ofset by 1 b/c we sckipped the VP, which was moved to the front
                
                node.right = new VPLeaf(S.subList(splitIndex+1, S.size()));
                node.left = new VPLeaf(S.subList(1, splitIndex+1));
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
        public void searchKNN(Vec query, int k, BoundedSortedList<ProbailityMatch<V>> list, double x, List<Double> qi)
        {
            x = dm.dist(p, query, qi, allVecs, distCache);
            if(list.size() < k || x < list.get(k-1).getProbability())
                list.add(new ProbailityMatch<V>(x, allVecs.get(this.p)));
            double tau = list.get(list.size()-1).getProbability();
            double middle = (this.left_high+this.right_low)*0.5;

            if( x < middle)
            {
                if(searchInLeft(x, tau) || list.size() < k)
                    this.left.searchKNN(query, k, list, x, qi);
                tau = list.get(list.size()-1).getProbability();
                if(searchInRight(x, tau) || list.size() < k)
                    this.right.searchKNN(query, k, list, x, qi);
            }
            else
            {
                if(searchInRight(x, tau) || list.size() < k)
                    this.right.searchKNN(query, k, list, x, qi);
                tau = list.get(list.size()-1).getProbability();
                if(searchInLeft(x, tau) || list.size() < k)
                    this.left.searchKNN(query, k, list, x, qi);
            }
        }

        @Override
        public void searchRange(Vec query, double range, List<VecPaired<V, Double>> list, double x, List<Double> qi)
        {
            x = dm.dist(this.p, query, qi, allVecs, distCache);
            if(x <= range)
                list.add(new VecPairedComparable<V, Double>(allVecs.get(this.p), x));

            if (searchInLeft(x, range))
                this.left.searchRange(query, range, list, x, qi);
            if (searchInRight(x, range))
                this.right.searchRange(query, range, list, x, qi);
        }

        @Override
        public TreeNode clone()
        {
            return new VPNode(this);
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
        public void searchKNN(Vec query, int k, BoundedSortedList<ProbailityMatch<V>> list, double x, List<Double> qi)
        {
            double dist = -1;
            
            //The zero check, for the case that the leaf is the ONLY node, x will be passed as 0.0 <= Max value will be true 
            double tau = list.size() == 0 ? Double.MAX_VALUE : list.get(list.size()-1).getProbability();
            for (int i = 0; i < points.size(); i++)
            {
                int point_i = points.getI(i);
                double bound_i = bounds.getD(i);
                if (list.size() < k)
                {
                    
                    list.add(new ProbailityMatch<V>(dm.dist(point_i, query, qi, allVecs, distCache), allVecs.get(point_i)));
                    tau = list.get(list.size() - 1).getProbability();
                }
                else if (bound_i - tau <= x && x <= bound_i + tau)//Bound check agains the distance to our parrent node, provided by x
                    if ((dist = dm.dist(point_i, query, qi, allVecs, distCache)) < tau)
                    {
                        list.add(new ProbailityMatch<V>(dist, allVecs.get(point_i)));
                        tau = list.get(list.size() - 1).getProbability();
                    }
            }
        }

        @Override
        public void searchRange(Vec query, double range, List<VecPaired<V, Double>> list, double x, List<Double> qi)
        {
            double dist = Double.MAX_VALUE;
            
            for (int i = 0; i < points.size(); i++)
            {
                int point_i = points.getI(i);
                double bound_i = bounds.getD(i);
                if (bound_i - range <= x && x <= bound_i + range)//Bound check agains the distance to our parrent node, provided by x
                    if ((dist = dm.dist(point_i, query, qi, allVecs, distCache)) < range)
                        list.add(new VecPairedComparable<V, Double>(allVecs.get(point_i), dist));
            }
        }

        @Override
        public TreeNode clone()
        {
            return new VPLeaf(this);
        }
    }
    
    public static class VPTreeFactory<V extends Vec> implements VectorCollectionFactory<V>
    {

        private static final long serialVersionUID = -2739851193676265510L;
        private VPSelection vpSelectionMethod;

        public VPTreeFactory(VPSelection vpSelectionMethod)
        {
            this.vpSelectionMethod = vpSelectionMethod;
        }

        public VPTreeFactory()
        {
            this(VPSelection.Random);
        }
        
        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return new VPTree<V>(source, distanceMetric, vpSelectionMethod);
        }

        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool)
        {
            return new VPTree<V>(source, distanceMetric, vpSelectionMethod, new Random(10), 80, 40, threadpool);
        }

        @Override
        public VectorCollectionFactory<V> clone()
        {
            return new VPTreeFactory<V>(vpSelectionMethod);
        }
    }
}
