
package jsat.linear.vectorcollection;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ExecutorService;

import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.VecPairedComparable;
import jsat.linear.distancemetrics.*;
import jsat.math.OnLineStatistics;
import jsat.utils.*;

/**
 * Standard KDTree implementation. KDTrees are fast to create with no distance computations needed.
 * Though KDTrees can be constructed in O(n) time, this implementation is O(n log n). KDTrees can be very 
 * fast for low dimensional data queries, but degrade as the dimensions increases. For very high dimensions 
 * or pathologically bad data, O(n<sup>2</sup>) performance worse then {@link VectorArray} can occur. 
 * <br>
 * <br>
 * Note: KD trees are only usable with Distance Metrics based off of the pNorm between two vectors. The valid distance metrics are 
 * {@link EuclideanDistance}, {@link ChebyshevDistance}, {@link ManhattanDistance}, {@link MinkowskiDistance}<br>
 * <br>
 * See:
 * <ul>
 * <li>Bentley, J. L. (1975). Multidimensional Binary Search Trees Used for Associative Searching. Commun. ACM, 18(9), 509–517. http://doi.org/10.1145/361002.361007</li>
 * <li>Moore, A. (1991). A tutorial on kd-trees (No. Technical Report No. 209).</li>
 * </ul>
 * @author Edward Raff
 * @param <V> The vector type
 */
public class KDTree<V extends Vec> implements VectorCollection<V>
{

    private static final long serialVersionUID = -7401342201406776463L;
    private DistanceMetric distanceMetric;
    private KDNode root;
    private PivotSelection pvSelection;
    private int size;
    private int leaf_node_size = 15;
    private List<V> allVecs;
    private List<Double> distCache;
    private double[] hr_hi, hr_low;
    
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
        Incremental, 
        /**
         * The next pivot will be selected by determining which pivot index contains the most variance. 
         * This method requires an additional O(n d) work per step. Where n is the number of data points
         * being split, and d is the dimension of the data set. 
         */
        Variance
    }

    /**
     * Creates a new KDTree with the given data and methods. 
     * 
     * @param vecs the list of vectors to place in this structure
     * @param distanceMetric the metric to use for the space
     * @param pvSelection the method of selection to use for determining what pivot to use. 
     * @param threadpool the source of threads to use when constructing. Null is permitted,
     * in which case a serial construction will occur. 
     */
    public KDTree(List<V> vecs, DistanceMetric distanceMetric, PivotSelection pvSelection, ExecutorService threadpool)
    {
        if(!( distanceMetric instanceof EuclideanDistance || distanceMetric instanceof ChebyshevDistance || 
              distanceMetric instanceof ManhattanDistance || distanceMetric instanceof MinkowskiDistance) )
            throw new ArithmeticException("KD Trees are not compatible with the given distance metric.");
        this.distanceMetric = distanceMetric;
        this.pvSelection = pvSelection;
        this.size = vecs.size();
        allVecs = vecs = new ArrayList<V>(vecs);//copy to avoid altering the input set
        if(threadpool == null || threadpool instanceof FakeExecutor)
            distCache = distanceMetric.getAccelerationCache(allVecs);
        else
            distCache = distanceMetric.getAccelerationCache(vecs, threadpool);
        List<Integer> vecIndices = new IntList(size);
        ListUtils.addRange(vecIndices, 0, size, 1);
        threadpool = null;
        if(threadpool == null)
            this.root = buildTree(vecIndices, 0, null, null);
        else
        {
            ModifiableCountDownLatch mcdl = new ModifiableCountDownLatch(1);
            this.root = buildTree(vecIndices, 0, threadpool, mcdl);
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
    
    /**
     * Creates a new KDTree with the given data and methods. 
     * 
     * @param vecs the list of vectors to place in this structure
     * @param distanceMetric the metric to use for the space
     * @param pvSelection the method of selection to use for determining what pivot to use. 
     */
    public KDTree(List<V> vecs, DistanceMetric distanceMetric, PivotSelection pvSelection)
    {
        this(vecs, distanceMetric, pvSelection, null);
    }
    
    /**
     * Creates a new KDTree with the given data and methods. <br>
     * 
     * @param vecs the list of vectors to place in this structure
     * @param distanceMetric the metric to use for the space
     */
    public KDTree(List<V> vecs, DistanceMetric distanceMetric)
    {
        this(vecs, distanceMetric, PivotSelection.Variance);
    }
    
    private KDTree(DistanceMetric distanceMetric, PivotSelection pvSelection)
    {
        this.distanceMetric = distanceMetric;
        this.pvSelection = pvSelection;
    }

    /**
     * no-arg constructor for serialization
     */
    public KDTree()
    {
        this(new EuclideanDistance(), PivotSelection.Variance);
    }
    
    private class KDNode implements Cloneable, Serializable
    {
        /**
         * Also called the "dom-elt"
         */
        protected int locatin;
        protected int axis;
        protected double pivot_s;
        
        protected KDNode left;
        protected KDNode right;
        
        public KDNode(int locatin, int axis)
        {
            this.locatin = locatin;
            this.axis = axis;
        }

        public KDNode(KDNode toCopy)
        {
            this(toCopy.locatin, toCopy.axis);
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
        @SuppressWarnings("unused")
        public void setLocatin(int locatin)
        {
            this.locatin = locatin;
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
        public int getLocatin()
        {
            return locatin;
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
        
        protected void searchK(int k, BoundedSortedList<ProbailityMatch<V>> knn, Vec target, List<Double> qi)
        {
            double pivot_s = this.pivot_s;
            //Cut hr in to two sub-hyperrectangles left-hr and right-hr
//            double[] left_hi = Arrays.copyOf(hr_hi, hr_hi.length);
//            left_hi[axis] = pivot_s;
//            double[] right_low = Arrays.copyOf(hr_low, hr_low.length);
//            right_low[axis] = pivot_s;
            
            double target_s = target.get(axis);
            boolean target_in_left = target_s <= pivot_s;
            
            KDNode nearKD, farKD;
//            double[] near_hr_hi, near_hr_low, far_hr_hi, far_hr_low;
            
            if(target_in_left)
            {
                nearKD = left;
                farKD = right;
//                near_hr_hi = left_hi;
//                near_hr_low = hr_low;
//                far_hr_hi = hr_hi;
//                far_hr_low = right_low;
            }
            else
            {
                nearKD = right;
                farKD = left;
//                near_hr_hi = hr_hi;
//                near_hr_low = right_low;
//                far_hr_hi = left_hi;
//                far_hr_low = hr_low;
            }
            
            nearKD.searchK(k, knn, target, qi);
            
            double maxDistSoFar = Double.MAX_VALUE;
            if(knn.size() >= k)
                maxDistSoFar = knn.get(k-1).getProbability();
            if(maxDistSoFar > Math.abs(target_s-pivot_s))
                farKD.searchK(k, knn, target, qi);
        }
        
        protected void searchR(double radius, List<VecPairedComparable<V, Double>> rnn, Vec target, List<Double> qi)
        {
            double pivot_s = this.pivot_s;
            //Cut hr in to two sub-hyperrectangles left-hr and right-hr
//            double[] left_hi = Arrays.copyOf(hr_hi, hr_hi.length);
//            left_hi[axis] = pivot_s;
//            double[] right_low = Arrays.copyOf(hr_low, hr_low.length);
//            right_low[axis] = pivot_s;
            
            double target_s = target.get(axis);

            if(radius > target_s-pivot_s)
                left.searchR(radius, rnn, target, qi);
            
            if(radius > pivot_s-target_s)
                right.searchR(radius, rnn, target, qi);
        }
    }
    
    private class KDLeaf extends KDNode
    {
        protected IntList owned;
        
        public KDLeaf(int locatin, int axis, List<Integer> toOwn)
        {
            super(locatin, axis);
            this.owned = new IntList(toOwn);
        }

        public KDLeaf(KDLeaf toCopy)
        {
            super(toCopy);
            this.owned = new IntList(toCopy.owned);
        }

        @Override
        protected void searchK(int k, BoundedSortedList<ProbailityMatch<V>> knn, Vec target, List<Double> qi)
        {
            for(int i : owned)
            {
                double dist = distanceMetric.dist(i, target, qi, allVecs, distCache);
                knn.add(new ProbailityMatch<V>(dist, allVecs.get(i)));
            }
        }
        
        @Override
        protected void searchR(double radius, List<VecPairedComparable<V, Double>> rnn, Vec target, List<Double> qi)
        {
            for(int i : owned)
            {
                double dist = distanceMetric.dist(i, target, qi, allVecs, distCache);
                if(dist <= radius)
                    rnn.add(new VecPairedComparable<V, Double>(allVecs.get(i), dist));
            }
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
        else if(depth == 0)
        {
            hr_hi = new double[allVecs.get(0).length()];
            hr_low = new double[allVecs.get(0).length()];
            Arrays.fill(hr_hi, -Double.MAX_VALUE);
            Arrays.fill(hr_low, Double.MAX_VALUE);
            for(Vec v : allVecs)
                for(int i = 0; i < v.length(); i++)
                {
                    hr_hi[i] = Math.max(hr_hi[i], v.get(i));
                    hr_low[i] = Math.min(hr_low[i], v.get(i));
                }
        }
        int mod = allVecs.get(0).length();
        
        if(data.size() <= leaf_node_size)
        {
            if(threadpool != null)
                mcdl.countDown();
//            return new KDNode(data.get(0), depth % mod);
            return new KDLeaf(-1, depth % mod, data);
        }
        
        int pivot = -1;
        if(pvSelection == PivotSelection.Incremental)
            pivot = depth % mod;
        else//Variance 
        {
            OnLineStatistics[] allStats = new OnLineStatistics[mod];
            for(int j = 0; j < allStats.length; j++)
                allStats[j] = new OnLineStatistics();
            
            for(int i = 0; i < data.size(); i++)//For each data point
            {
                V vec = allVecs.get(data.get(i));
                for(int j = 0; j < allStats.length; j++)//For each dimension 
                    allStats[j].add(vec.get(j));
            }
            
            double maxVariance = -1;
            for(int j = 0; j < allStats.length; j++)
            {
                if(allStats[j].getVarance() > maxVariance)
                {
                    
                    maxVariance = allStats[j].getVarance();
                    pivot = j;
                }
            }
            if(pivot < 0)//All dims had NaN as variance? Fall back to incremental selection
                pivot = depth % mod;
        }
        
        Collections.sort(data, new VecIndexComparator(pivot));
        
        final int medianIndex = getSplitIndex(data, pivot);
        if(medianIndex == data.size()-1)//Everyone has the same value? OK, leaf node then
            return new KDLeaf(data.get(0), depth % mod, data);
        //else, continue as planned
        
        final KDNode node = new KDNode(data.get(medianIndex), pivot);
        node.pivot_s = allVecs.get(data.get(medianIndex)).get(pivot);
        
        //We could save code lines by making only one path threadpool dependent. 
        //But this order has better locality for single threaded, while the 
        //reverse call order workes better for multi core
        if(threadpool == null)
        {
            node.setLeft(buildTree(data.subList(0, medianIndex+1), depth+1, threadpool, mcdl));
            node.setRight(buildTree(data.subList(medianIndex+1, data.size()), depth+1, threadpool, mcdl));
        }
        else//multi threaded
        {
            mcdl.countUp();
            //Right side first, it will start running on a different core
            threadpool.submit(new Runnable() {

                @Override
                public void run()
                {
                    node.setRight(buildTree(data.subList(medianIndex+1, data.size()), depth+1, threadpool, mcdl));
                }
            });
            
            //now do the left here, 
            node.setLeft(buildTree(data.subList(0, medianIndex), depth+1, threadpool, mcdl));
        }
        
        return node;
    }

    public int getSplitIndex(final List<Integer> data, int pivot)
    {
        int medianIndex = data.size()/2;
        //What if more than one point have the samve value? Keep incrementing until that dosn't happen
        while(medianIndex < data.size()-1 && allVecs.get(data.get(medianIndex)).get(pivot) == allVecs.get(data.get(medianIndex+1)).get(pivot))
            medianIndex++;
        return medianIndex;
    }
    
    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, int neighbors)
    {
        if(neighbors < 1)
            throw new RuntimeException("Invalid number of neighbors to search for");
        
        BoundedSortedList<ProbailityMatch<V>> knns = new BoundedSortedList<ProbailityMatch<V>>(neighbors);
        
//        knnKDSearch(query, knns);
        root.searchK(neighbors, knns, query, distanceMetric.getQueryInfo(query));
        
        List<VecPaired<V, Double>> knnsList = new ArrayList<VecPaired<V, Double>>(knns.size());
        for(int i = 0; i < knns.size(); i++)
        {
            ProbailityMatch<V> pm = knns.get(i);
            knnsList.add(new VecPaired<V, Double>(pm.getMatch(), pm.getProbability()));
        }
        
        return knnsList;
    }
    
    @Override
    public int size()
    {
        return size;
    }
    
    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, double range)
    {
        if(range <= 0)
            throw new RuntimeException("Range must be a positive number");
        ArrayList<VecPairedComparable<V, Double>> vecs = new ArrayList<VecPairedComparable<V, Double>>();
        
        List<Double> qi = distanceMetric.getQueryInfo(query);
        
        root.searchR(range, vecs, query, qi);
        
        Collections.sort(vecs);
        
        return vecs;
        
    }

    @Override
    public KDTree<V> clone()
    {
        KDTree<V> clone = new KDTree<V>(distanceMetric, pvSelection);
        if(this.distCache != null)
            clone.distCache = new DoubleList(this.distCache);
        if(this.allVecs != null)
            clone.allVecs = new ArrayList<V>(this.allVecs);
        clone.size = this.size;
        if(this.root != null)
            clone.root = this.root.clone();
        if(this.hr_hi != null)
            clone.hr_hi = Arrays.copyOf(hr_hi, hr_hi.length);
        if(this.hr_low != null)
            clone.hr_low = Arrays.copyOf(hr_low, hr_low.length);
        return clone;
    }
    
    public static class KDTreeFactory<V extends Vec> implements VectorCollectionFactory<V>
    {

        private static final long serialVersionUID = 3508731608962277804L;
        private PivotSelection pivotSelectionMethod;

        public KDTreeFactory(PivotSelection pvSelectionMethod)
        {
            this.pivotSelectionMethod = pvSelectionMethod;
        }

        public KDTreeFactory()
        {
            this(PivotSelection.Variance);
        }

        public PivotSelection getPivotSelectionMethod()
        {
            return pivotSelectionMethod;
        }

        public void setPivotSelectionMethod(PivotSelection pivotSelectionMethod)
        {
            this.pivotSelectionMethod = pivotSelectionMethod;
        }
        
        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return getVectorCollection(source, distanceMetric, null);
        }

        @Override
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric, ExecutorService threadpool)
        {
            return new KDTree<V>(source, distanceMetric, pivotSelectionMethod, threadpool);
        }

        @Override
        public KDTreeFactory<V> clone() 
        {
            return new KDTreeFactory<V>(pivotSelectionMethod);
        }
    }
}
