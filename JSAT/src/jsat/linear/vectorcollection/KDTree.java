
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
 * {@link EuclideanDistance}, {@link ChebyshevDistance}, {@link ManhattanDistance}, {@link MinkowskiDistance}
 * 
 * @author Edward Raff
 */
public class KDTree<V extends Vec> implements VectorCollection<V>
{

	private static final long serialVersionUID = -7401342201406776463L;
	private DistanceMetric distanceMetric;
    private KDNode root;
    private PivotSelection pvSelection;
    private int size;
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
         * The index in {@link #allVecs} of the vector that this node contains
         */
        int locatin;
        int axis;

        KDNode left;
        KDNode right;
        
        public KDNode(int locatin, int axis)
        {
            this.locatin = locatin;
            this.axis = axis;
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
            KDNode clone = new KDNode(locatin, axis);
            if(this.left != null)
                clone.left = this.left.clone();
            if(this.right != null)
                clone.right = this.right.clone();
            return clone;
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
        
        if(data.size() == 1)
        {
            if(threadpool != null)
                mcdl.countDown();
            return new KDNode(data.get(0), depth % mod);
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
        
        final int medianIndex = data.size()/2;
        
        final KDNode node = new KDNode(data.get(medianIndex), pivot);
        
        //We could save code lines by making only one path threadpool dependent. 
        //But this order has better locality for single threaded, while the 
        //reverse call order workes better for multi core
        if(threadpool == null)
        {
            node.setLeft(buildTree(data.subList(0, medianIndex), depth+1, threadpool, mcdl));
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
    
    //Use the Probaility match to pair a distance with the vector
    private void knnKDSearch(Vec query, BoundedSortedList<ProbailityMatch<V>> knns)
    {
        Stack<KDNode> stack = new Stack<KDNode>();
        stack.push(root);
        
        List<Double> qi = distanceMetric.supportsAcceleration() ? distanceMetric.getQueryInfo(query) : null;
        
        while(!stack.isEmpty())
        {
            KDNode node = stack.pop();
            if(node == null)
                continue;
            V curData = allVecs.get(node.locatin);
            double distance = distanceMetric.dist(node.locatin, query, qi, allVecs, distCache);
            
            knns.add( new ProbailityMatch<V>(distance, curData));
            
            double qVal, cVal;
            double diff = (qVal = query.get(node.axis)) - (cVal = curData.get(node.axis));

            if(diff <= 0)
            {
                if(qVal - knns.last().getProbability() <= cVal || knns.size() < knns.maxSize())
                    stack.push(node.left);
                if(qVal + knns.last().getProbability() > cVal || knns.size() < knns.maxSize())
                    stack.push(node.right);
            }
            else
            {
                if(qVal + knns.last().getProbability() > cVal || knns.size() < knns.maxSize())
                    stack.push(node.right);
                if(qVal - knns.last().getProbability() <= cVal || knns.size() < knns.maxSize())
                    stack.push(node.left);
            }
                        
        }
    }
    
    @Override
    public List<? extends VecPaired<V, Double>> search(Vec query, int neighbors)
    {
        if(neighbors < 1)
            throw new RuntimeException("Invalid number of neighbors to search for");
        
        BoundedSortedList<ProbailityMatch<V>> knns = new BoundedSortedList<ProbailityMatch<V>>(neighbors);
        
        knnKDSearch(query, knns);
        
        List<VecPaired<V, Double>> knnsList = new ArrayList<VecPaired<V, Double>>(knns.size());
        for(int i = 0; i < knns.size(); i++)
        {
            ProbailityMatch<V> pm = knns.get(i);
            knnsList.add(new VecPaired<V, Double>(pm.getMatch(), pm.getProbability()));
        }
        
        return knnsList;
    }
    
    private void distanceSearch(Vec query, List<Double> qi, KDNode node, List<VecPairedComparable<V, Double>> knns, double range)
    {
        if(node == null)
            return;
        V curData = allVecs.get(node.locatin);
        double distance = distanceMetric.dist(node.locatin, query, qi, allVecs, distCache);
        
        if(distance <= range)
            knns.add( new VecPairedComparable<V, Double>(curData, distance) );
        
        double diff = query.get(node.axis) - curData.get(node.axis);
        
        KDNode close = node.left, far = node.right;
        if(diff > 0)
        {
            close = node.right;
            far = node.left;
        }
        
        distanceSearch(query, qi, close, knns, range);
        if(diff*diff <= range)
            distanceSearch(query, qi, far, knns, range);
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
        
        List<Double> qi = distanceMetric.supportsAcceleration() ? distanceMetric.getQueryInfo(query) : null;
        
        distanceSearch(query, qi, root, vecs, range);
        
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
        return clone;
    }
    
    public static class KDTreeFactory<V extends Vec> implements VectorCollectionFactory<V>
    {
        /**
		 * 
		 */
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
