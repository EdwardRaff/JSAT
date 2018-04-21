
package jsat.linear.vectorcollection;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.linear.IndexValue;

import jsat.linear.Vec;
import jsat.linear.distancemetrics.*;
import jsat.math.FastMath;
import jsat.math.OnLineStatistics;
import jsat.utils.*;
import jsat.utils.concurrent.ParallelUtils;

/**
 * Standard KDTree implementation. KDTrees are fast to create with no distance computations needed.
 * Though KDTrees can be constructed in O(n) time, this
 * implementation is O(n log n). KDTrees can be very fast for low dimensional
 * data queries, but degrade as the dimensions increases. For very high
 * dimensions or pathologically bad data, O(n<sup>2</sup>) performance worse
 * then {@link VectorArray} can occur.
 * <br>
 * <br>
 * Note: KD trees are only usable with Distance Metrics based off of the pNorm
 * between two vectors. The valid distance metrics are
 * {@link EuclideanDistance}, {@link ChebyshevDistance}, {@link ManhattanDistance}, {@link MinkowskiDistance}<br>
 * <br>
 * See:
 * <ul>
 * <li>Bentley, J. L. (1975). Multidimensional Binary Search Trees Used for
 * Associative Searching. Commun. ACM, 18(9), 509–517.
 * http://doi.org/10.1145/361002.361007</li>
 * <li>Moore, A. (1991). A tutorial on kd-trees (No. Technical Report No.
 * 209).</li>
 * </ul>
 *
 * @author Edward Raff
 * @param <V> The vector type
 */
public class KDTree<V extends Vec> implements IncrementalCollection<V>
{

    private static final long serialVersionUID = -7401342201406776463L;
    private DistanceMetric distanceMetric;
    private KDNode root;
    private PivotSelection pvSelection;
    private int size;
    private int leaf_node_size = 20;
    private List<V> allVecs;
    private List<Double> distCache;
    
    /**
     * KDTree uses an index of the vector at each stage to use as a pivot, 
     * dividing the remaining elements into two sets. These control the 
     * method used to determine the pivot at each step. 
     */
    public enum PivotSelection
    {
        /**
         * The next pivot will be selected by iteratively going through each possible pivot. 
         * This method has no additional overhead. 
         */
        INCREMENTAL, 
        /**
         * The next pivot will be selected by determining which pivot index contains the most variance. 
         * This method requires an additional O(n d) work per step. Where n is the number of data points
         * being split, and d is the dimension of the data set. 
         */
        VARIANCE,
        /**
         * The next pivot dimension will be selected as the dimension with the
         * maximum spread, with the value selected as the point closest to the
         * median value of the spread (i.e., the medoid)
         * See: Moore, A. (1991). A tutorial on kd-trees (No. Technical Report
         * No. 209).
         */
        SPREAD_MEDOID,
    }

    /**
     * Creates a new KDTree with the given data and methods. 
     * 
     * @param vecs the list of vectors to place in this structure
     * @param distanceMetric the metric to use for the space
     * @param pvSelection the method of selection to use for determining what
     * pivot to use.
     * @param parallel {@code true} if multiple threads should be used for
     * construction, {@code false} otherwise.
     */
    public KDTree(List<V> vecs, DistanceMetric distanceMetric, PivotSelection pvSelection, boolean parallel)
    {
        this.distanceMetric = distanceMetric;
        this.pvSelection = pvSelection;
        build(parallel, vecs, distanceMetric);
    }
    
    /**
     * Creates a new KDTree with the given data and methods. 
     * 
     * @param vecs the list of vectors to place in this structure
     * @param distanceMetric the metric to use for the space
     * @param pvSelection the method of selection to use for determining what pivot to use. 
     */
    public KDTree(List<V> vecs, DistanceMetric distanceMetric, PivotSelection pvSelection)
    {
        this(vecs, distanceMetric, pvSelection, false);
    }
    
    /**
     * Creates a new KDTree with the given data and methods. <br>
     * 
     * @param vecs the list of vectors to place in this structure
     * @param distanceMetric the metric to use for the space
     */
    public KDTree(List<V> vecs, DistanceMetric distanceMetric)
    {
        this(vecs, distanceMetric, PivotSelection.SPREAD_MEDOID);
    }
    
    private KDTree(DistanceMetric distanceMetric, PivotSelection pvSelection)
    {
        setDistanceMetric(distanceMetric);
        this.pvSelection = pvSelection;
    }

    public KDTree(PivotSelection pivotSelection)
    {
        this(new EuclideanDistance(), pivotSelection);
    }
    
    public KDTree()
    {
        this(PivotSelection.SPREAD_MEDOID);
    }
    
    @Override
    public List<Double> getAccelerationCache()
    {
        return distCache;
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
        this.leaf_node_size = leaf_size;
    }

    /**
     *
     * @return the number of points to store within a leaf node
     */
    public int getLeafSize()
    {
        return leaf_node_size;
    }

    @Override
    public void setDistanceMetric(DistanceMetric dm)
    {
        if(!( dm instanceof EuclideanDistance || dm instanceof ChebyshevDistance || 
              dm instanceof ManhattanDistance || dm instanceof MinkowskiDistance) )
            throw new ArithmeticException("KD Trees are not compatible with the given distance metric.");
        this.distanceMetric = dm;
    }

    @Override
    public DistanceMetric getDistanceMetric()
    {
        return distanceMetric;
    }

    @Override
    public void build(boolean parallel, List<V> vecs, DistanceMetric dm)
    {
        setDistanceMetric(dm);
        this.size = vecs.size();
        allVecs = vecs = new ArrayList<>(vecs);//copy to avoid altering the input set
        distCache = distanceMetric.getAccelerationCache(vecs, parallel);
        List<Integer> vecIndices = new IntList(size);
        ListUtils.addRange(vecIndices, 0, size, 1);
        
        if(!parallel)
            this.root = buildTree(vecIndices, 0, null, null);
        else
        {
            ModifiableCountDownLatch mcdl = new ModifiableCountDownLatch(1);
            this.root = buildTree(vecIndices, 0, ParallelUtils.CACHED_THREAD_POOL, mcdl);
            try
            {
                mcdl.await();
            }
            catch (InterruptedException ex)
            {
                //Failure, fall back to single threaded version
                this.root = buildTree(vecIndices, 0, null, null);
            }
        }
    }

    @Override
    public void insert(V x)
    {
        if(allVecs == null)//init
        {
            allVecs = new ArrayList<>();
            distCache = distanceMetric.getAccelerationCache(allVecs);
            this.size = 0;
            this.root = new KDLeaf(0, new IntList());
        }
        int indx = size++;
        allVecs.add(x);
        if(distCache != null)
            distCache.addAll(distanceMetric.getQueryInfo(x));

        if(root.insert(indx))
            root = buildTree(IntList.range(size), 0, null, null);
    }
    
    private class KDNode implements Cloneable, Serializable
    {
        protected int axis;
        /**
         * The splitting value along the axis
         */
        protected double pivot_s;
        
        protected KDNode left;
        protected KDNode right;
        
        public KDNode(int axis)
        {
            this.axis = axis;
        }

        public KDNode(KDNode toCopy)
        {
            this(toCopy.axis);
            this.pivot_s = toCopy.pivot_s;
            if(toCopy.left != null)
                this.left = toCopy.left.clone();
            if(toCopy.left != null)
                this.right = toCopy.right.clone();
        }
        
        @SuppressWarnings("unused")
        public void setAxis(int axis)
        {
            this.axis = axis;
        }

        public void setLeft(KDNode left)
        {
            this.left = left;
        }

        public void setRight(KDNode right)
        {
            this.right = right;
        }
        @SuppressWarnings("unused")
        public int getAxis()
        {
            return axis;
        }
        @SuppressWarnings("unused")
        public KDNode getLeft()
        {
            return left;
        }
        
        @SuppressWarnings("unused")
        public KDNode getRight()
        {
            return right;
        }

        @Override
        protected KDNode clone() 
        {
            return new KDNode(this);
        }
        
        protected void searchK(int k, BoundedSortedList<IndexDistPair> knn, Vec target, List<Double> qi)
        {
            double target_s = target.get(axis);
            boolean target_in_left = target_s <= pivot_s;
            
            KDNode nearKD, farKD;

            if(target_in_left)
            {
                nearKD = left;
                farKD = right;
            }
            else
            {
                nearKD = right;
                farKD = left;
            }
            
            nearKD.searchK(k, knn, target, qi);
            
            double maxDistSoFar = Double.MAX_VALUE;
            if(knn.size() >= k)
                maxDistSoFar = knn.get(k-1).getDist();
            if(maxDistSoFar > Math.abs(target_s-pivot_s))
                farKD.searchK(k, knn, target, qi);
        }
        
        protected void searchR(double radius, List<Integer> vecsInRage, List<Double> distVecsInRange, Vec target, List<Double> qi)
        {
            double target_s = target.get(axis);

            if(radius > target_s-pivot_s)
                left.searchR(radius, vecsInRage, distVecsInRange, target, qi);
            
            if(radius > pivot_s-target_s)
                right.searchR(radius, vecsInRage, distVecsInRange, target, qi);
        }
        
        /**
         * 
         * @param x_indx
         * @return {@code true} if this node should be replaced using its children after insertion
         */
        protected boolean insert(int x_indx)
        {
            double target_s = get(x_indx).get(axis);
            boolean target_in_left = target_s <= pivot_s;

            if (target_in_left)
            {
                if (left.insert(x_indx))
                    left = buildTree(((KDLeaf) left).owned, axis + 1, null, null);
            }
            else
            {
                if (right.insert(x_indx))
                    right = buildTree(((KDLeaf) right).owned, axis + 1, null, null);
            }
            return false;
        }
    }
    
    private class KDLeaf extends KDNode
    {
        protected IntList owned;
        
        public KDLeaf(int axis, List<Integer> toOwn)
        {
            super(axis);
            this.owned = new IntList(toOwn);
        }

        public KDLeaf(KDLeaf toCopy)
        {
            super(toCopy);
            this.owned = new IntList(toCopy.owned);
        }

        @Override
        protected void searchK(int k, BoundedSortedList<IndexDistPair> knn, Vec target, List<Double> qi)
        {
            for(int i : owned)
            {
                double dist = distanceMetric.dist(i, target, qi, allVecs, distCache);
                knn.add(new IndexDistPair(i, dist));
            }
        }
        
        @Override
        protected void searchR(double radius, List<Integer> vecsInRage, List<Double> distVecsInRange, Vec target, List<Double> qi)
        {
            for(int i : owned)
            {
                double dist = distanceMetric.dist(i, target, qi, allVecs, distCache);
                if(dist <= radius)
                {
                    vecsInRage.add(i);
                    distVecsInRange.add(dist);
                }
            }
        }

        @Override
        protected boolean insert(int x_indx)
        {
            this.owned.add(x_indx);
            return owned.size() >= leaf_node_size*2;
        }

        @Override
        protected KDLeaf clone()
        {
            return new KDLeaf(this);
        }
    }
    
    private class VecIndexComparator implements Comparator<Integer>
    {
        private final int index;

        public VecIndexComparator(int index)
        {
            this.index = index;
        }
        
        @Override
        public int compare(Integer o1, Integer o2)
        {
            return Double.compare( allVecs.get(o1).get(index), allVecs.get(o2).get(index));
        }
        
    }
    
    /**
     * 
     * @param data subset of data to work on
     * @param depth recursion depth
     * @param threadpool threadpool source. Null is accepted, and means it will be done immediately 
     * @param mcdl used to wait on for the original caller, only needed when threadpool is non null
     * @return the root tree node for the given set of data
     */
    private KDNode buildTree(final List<Integer> data, final int depth, final ExecutorService threadpool, final ModifiableCountDownLatch mcdl)
    {
        if(data == null || data.isEmpty())
        {
            if(threadpool != null)//Threadpool null checks since no thread pool means do single threaded
                mcdl.countDown();
            return null;
        }
        
        int mod = allVecs.get(0).length();
        
        if(data.size() <= leaf_node_size)
        {
            if(threadpool != null)
                mcdl.countDown();
//            return new KDNode(data.get(0), depth % mod);
            return new KDLeaf(depth % mod, data);
        }
        
        final boolean isSparse = get(data.get(0)).isSparse();
        int pivot = -1;
        //Some pivot methods will select the value they want, and so overwrite NaN. Otherwise, use NaN to flag that a median search is needed
        double pivot_val = Double.NaN;
        switch (pvSelection)
        {
            case VARIANCE:
                OnLineStatistics[] allStats = new OnLineStatistics[mod];
                for (int j = 0; j < allStats.length; j++)
                    allStats[j] = new OnLineStatistics();
                for (int i : data)//For each data point
                {
                    V vec = get(i);
                    for (int j = 0; j < allStats.length; j++)//For each dimension
                        allStats[j].add(vec.get(j));
                }
                double maxVariance = -1;
                for (int j = 0; j < allStats.length; j++)
                {
                    if (allStats[j].getVarance() > maxVariance)
                    {

                        maxVariance = allStats[j].getVarance();
                        pivot = j;
                    }
                }
                if (pivot < 0)//All dims had NaN as variance? Fall back to incremental selection
                    pivot = depth % mod;
                break;
            case SPREAD_MEDOID:
                //Find the spread of each dimension
                double[] mins = new double[mod];
                double[] maxs = new double[mod];
                Arrays.fill(mins, Double.POSITIVE_INFINITY);
                Arrays.fill(maxs, Double.NEGATIVE_INFINITY);
                //If sparse, keep a set of indexes we HAVE NOT SEEN
                //these have implicity zeros we need to add back at the end
                final Set<Integer> neverSeen = isSparse ? new IntSet(ListUtils.range(0, get(0).length())) : Collections.EMPTY_SET;
                for(int i : data)
                {
                    V v = get(i);
                    for(IndexValue iv : v)
                    {
                        int d = iv.getIndex();
                        double val = iv.getValue();
                        mins[d] = Math.min(mins[d], val);
                        maxs[d] = Math.max(maxs[d], val);
                        neverSeen.remove(d);
                    }
                }
                //find the dimension of maximum spread
                int maxSpreadDim = 0;
                double maxSpreadVal = 0;
                
                for(int d = 0; d < mod; d++)
                {
                    if(neverSeen != null && neverSeen.contains(d))
                    {
                        maxs[d] = Math.max(maxs[d], 0);
                        mins[d] = Math.min(mins[d], 0);
                    }
                    double v = maxs[d]-mins[d];
                    if(v > maxSpreadVal)
                    {
                        maxSpreadDim = d;
                        maxSpreadVal = v;
                    }
                }
                pivot = maxSpreadDim;
                //find the value cloesest to the midpoint of the spread
                double midPoint = (maxs[maxSpreadDim]-mins[maxSpreadDim])/2 + mins[maxSpreadDim];
                double closestVal = maxs[maxSpreadDim];
                for (int i = 0; i < data.size(); i++)
                {
                    V v = get(i);
                    double val = v.get(maxSpreadDim);
                    if (Math.abs(midPoint - val) < Math.abs(midPoint - closestVal))
                        closestVal = val;
                }
                pivot_val = closestVal;
                break;
            default:
            case INCREMENTAL:
                pivot = depth % mod;
                break;
        }
        
        final KDNode node = new KDNode(pivot);
        
        //split index is the point in the array data that splits it into the left and right child branches
        int splitIndex = -1;
        //Looks like we have a pivot value? lets check it!
        if(!Double.isNaN(pivot_val))
        {
            //lets go through and push the data around the pivot value
            int front = 0;
            for(int i = 0; i < data.size(); i++)
                if(get(data.get(i)).get(pivot) <= pivot_val)
                    ListUtils.swap(data, front++, i);
            //How deep would we go if the tree was balanced?
            int balanced_depth = FastMath.floor_log2(allVecs.size());
            if(balanced_depth*3/2 < depth 
                    &&  (front < leaf_node_size/3 || data.size()-front < leaf_node_size/3) 
                    || balanced_depth*3 < depth)//too lopsided, fall back to medain spliting!
                pivot_val = Double.NaN;
            else
            {
                splitIndex = front-1;
                node.pivot_s = pivot_val;
            }
        }
        
        if(splitIndex <= 0 || splitIndex >= data.size()-1)//Split turned bad
            pivot_val = Double.NaN;//Set to NaN so that we fall back to median-based split selection
        
        //INTENTIONALLY NOT AN ELSE IF
        //pivot_val might be set to NaN if pivot looked bad
        if(Double.isNaN(pivot_val))
        {
            Collections.sort(data, new VecIndexComparator(pivot));

            splitIndex = getMedianIndex(data, pivot);
            if(splitIndex == data.size()-1)//Everyone has the same value? OK, leaf node then
                return new KDLeaf(depth % mod, data);
            node.pivot_s = pivot_val = get(data.get(splitIndex)).get(pivot);
        }
        
        if(splitIndex == 0 || splitIndex >= data.size()-1)
        {
            System.out.println("Adsas");
        }
                
        //We could save code lines by making only one path threadpool dependent. 
        //But this order has better locality for single threaded, while the 
        //reverse call order workes better for multi core
        if(threadpool == null)
        {
            node.setLeft(buildTree(data.subList(0, splitIndex+1), depth+1, threadpool, mcdl));
            node.setRight(buildTree(data.subList(splitIndex+1, data.size()), depth+1, threadpool, mcdl));
        }
        else//multi threaded
        {
            mcdl.countUp();
            IntList data_l = new IntList(data.subList(0, splitIndex+1));
            IntList data_r = new IntList(data.subList(splitIndex+1, data.size()));
            //Right side first, it will start running on a different core
            threadpool.submit(() ->
            {
                node.setRight(buildTree(data_r, depth+1, threadpool, mcdl));
            });
            
            //now do the left here, 
            node.setLeft(buildTree(data_l, depth+1, threadpool, mcdl));
        }
        
        return node;
    }

    /**
     * Returns the index for the median, adjusted incase multiple features have the same value. 
     * @param data the dataset to get the median index of 
     * @param pivot the dimension to pivot on, and ensure the median index has a different value on the left side
     * @return 
     */
    public int getMedianIndex(final List<Integer> data, int pivot)
    {
        int medianIndex = data.size()/2;
        //What if more than one point have the samve value? Keep incrementing until that dosn't happen
        while(medianIndex < data.size()-1 && allVecs.get(data.get(medianIndex)).get(pivot) == allVecs.get(data.get(medianIndex+1)).get(pivot))
            medianIndex++;
        return medianIndex;
    }

    @Override
    public void search(Vec query, int numNeighbors, List<Integer> neighbors, List<Double> distances)
    {
        if (numNeighbors < 1)
            throw new RuntimeException("Invalid number of neighbors to search for");

        BoundedSortedList<IndexDistPair> knns = new BoundedSortedList<>(numNeighbors);

//        knnKDSearch(query, knns);
        root.searchK(numNeighbors, knns, query, distanceMetric.getQueryInfo(query));

        neighbors.clear();
        distances.clear();
        for (int i = 0; i < knns.size(); i++)
        {
            IndexDistPair pm = knns.get(i);
            neighbors.add(pm.getIndex());
            distances.add(pm.getDist());
        }
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
    public void search(Vec query, double range, List<Integer> neighbors, List<Double> distances)
    {
        if (range <= 0)
            throw new RuntimeException("Range must be a positive number");
        neighbors.clear();
        distances.clear();

        
        List<Double> qi = distanceMetric.getQueryInfo(query);

        root.searchR(range, neighbors, distances, query, qi);

        IndexTable it = new IndexTable(distances);
        it.apply(neighbors);
        it.apply(distances);
    }
    
    
    @Override
    public KDTree<V> clone()
    {
        KDTree<V> clone = new KDTree<>(distanceMetric, pvSelection);
        if(this.distCache != null)
            clone.distCache = new DoubleList(this.distCache);
        if(this.allVecs != null)
            clone.allVecs = new ArrayList<>(this.allVecs);
        clone.size = this.size;
        if(this.root != null)
            clone.root = this.root.clone();
        return clone;
    }
    
}
