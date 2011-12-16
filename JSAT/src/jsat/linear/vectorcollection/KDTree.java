
package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import jsat.classifiers.DataPointPair;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.ChebyshevDistance;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.distancemetrics.ManhattanDistance;
import jsat.linear.distancemetrics.MinkowskiDistance;
import jsat.math.OnLineStatistics;
import jsat.utils.BoundedSortedList;
import jsat.utils.BoundedSortedSet;
import jsat.utils.PairedReturn;
import jsat.utils.ProbailityMatch;
import static jsat.linear.VecPaired.*;

/**
 * Note: KD trees are only usable with Distance Metrics based off of the pNorm between two vectors. The valid distance metrics are 
 * {@link EuclideanDistance}, {@link ChebyshevDistance}, {@link ManhattanDistance}, {@link MinkowskiDistance}
 * 
 * @author Edward Raff
 */
public class KDTree<V extends Vec> implements VectorCollection<V>
{
    private DistanceMetric distanceMetric;
    private KDNode root;
    private PivotSelection pvSelection;
    
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

    public KDTree(List<V> vecs, DistanceMetric distanceMetric, PivotSelection pvSelection)
    {
        if(!( distanceMetric instanceof EuclideanDistance || distanceMetric instanceof ChebyshevDistance || 
              distanceMetric instanceof ManhattanDistance || distanceMetric instanceof MinkowskiDistance) )
            throw new ArithmeticException("KD Trees are not compatible with the given distance metric.");
        this.distanceMetric = distanceMetric;
        this.pvSelection = pvSelection;
        this.root = buildTree(vecs, 0);
    }
    
    public KDTree(List<V> vecs, DistanceMetric distanceMetric)
    {
        this(vecs, distanceMetric, PivotSelection.Variance);
    }
    
    private class KDNode 
    {
        V locatin;
        int axis;

        KDNode left;
        KDNode right;
        
        public KDNode(V locatin, int axis)
        {
            this.locatin = locatin;
            this.axis = axis;
        }
        
        public void setAxis(int axis)
        {
            this.axis = axis;
        }

        public void setLeft(KDNode left)
        {
            this.left = left;
        }

        public void setLocatin(V locatin)
        {
            this.locatin = locatin;
        }

        public void setRight(KDNode right)
        {
            this.right = right;
        }

        public int getAxis()
        {
            return axis;
        }

        public KDNode getLeft()
        {
            return left;
        }

        public V getLocatin()
        {
            return locatin;
        }

        public KDNode getRight()
        {
            return right;
        }
        
    }
    
    private class VecIndexComparator implements Comparator<Vec>
    {
        int index;

        public VecIndexComparator(int index)
        {
            this.index = index;
        }
        
        public int compare(Vec o1, Vec o2)
        {
            return Double.compare( o1.get(index), o2.get(index));
        }
        
    }
    
    private KDNode buildTree(List<V> data, int depth)
    {
        if(data == null || data.isEmpty())
            return null;
        int mod = data.get(0).length();
        
        if(data.size() == 1)
            return new KDNode(data.get(0), depth % mod);
        
        int pivot = -1;
        if(pvSelection == PivotSelection.Incremental)
            pivot = depth % mod;
        else//Variance 
        {
            OnLineStatistics[] allStats = new OnLineStatistics[data.get(0).length()];
            for(int j = 0; j < allStats.length; j++)
                allStats[j] = new OnLineStatistics();
            
            for(int i = 0; i < data.size(); i++)//For each data point
            {
                V vec = data.get(i);
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
        }
        
        Collections.sort(data, new VecIndexComparator(pivot));
        
        int medianIndex = data.size()/2;
        V median = data.get(medianIndex);
        
        KDNode node = new KDNode(median, pivot);
        
        node.setLeft(buildTree(data.subList(0, medianIndex), depth+1));
        node.setRight(buildTree(data.subList(medianIndex+1, data.size()), depth+1));
        
        return node;
    }
    
    //Use the Probaility match to pair a distance with the vector
    private void knnKDSearch(Vec query, KDNode node, BoundedSortedList<ProbailityMatch<V>> knns)
    {
        if(node == null)
            return;
        V curData = node.locatin;
        double distance = distanceMetric.dist(query, extractTrueVec(curData));
        
        knns.add( new ProbailityMatch<V>(distance, curData));
        
        double diff = query.get(node.axis) - curData.get(node.axis);
        
        KDNode close = node.left, far = node.right;
        if(diff > 0)
        {
            close = node.right;
            far = node.left;
        }
        
        knnKDSearch(query, close, knns);
        if(diff*diff < knns.first().getProbability())
            knnKDSearch(query, far, knns);
    }
    
    public List<VecPaired<Double,V>> search(Vec query, int neighbors)
    {
        if(neighbors < 1)
            throw new RuntimeException("Invalid number of neighbors to search for");
        
        BoundedSortedList<ProbailityMatch<V>> knns = new BoundedSortedList<ProbailityMatch<V>>(neighbors);
        
        knnKDSearch(query, root, knns);
        
        List<VecPaired<Double,V>> knnsList = new ArrayList<VecPaired<Double,V>>(knns.size());
        for(int i = 0; i < knns.size(); i++)
        {
            ProbailityMatch<V> pm = knns.get(i);
            knnsList.add(new VecPaired<Double, V>(pm.getMatch(), pm.getProbability()));
        }
        
        return knnsList;
    }
    
    private void distanceSearch(Vec query, KDNode node, List<VecPaired<Double,V>> knns, double range)
    {
        if(node == null)
            return;
        V curData = node.locatin;
        double distance = distanceMetric.dist(query, extractTrueVec(curData));
        
        if(distance <= range)
            knns.add( new VecPaired<Double, V>(curData, distance) );
        
        double diff = query.get(node.axis) - curData.get(node.axis);
        
        KDNode close = node.left, far = node.right;
        if(diff > 0)
        {
            close = node.right;
            far = node.left;
        }
        
        distanceSearch(query, close, knns, range);
        if(diff*diff <= range)
            distanceSearch(query, far, knns, range);
    }
    
    public List<VecPaired<Double,V>> search(Vec query, double range)
    {
        if(range <= 0)
            throw new RuntimeException("Range must be a positive number");
        ArrayList<VecPaired<Double,V>> vecs = new ArrayList<VecPaired<Double,V>>();
        
        distanceSearch(query, root, vecs, range);
        
        Collections.sort(vecs, new Comparator<VecPaired<Double, V>>() {

            public int compare(VecPaired<Double, V> o1, VecPaired<Double, V> o2)
            {
                return Double.compare(o1.getPair(), o2.getPair());
            }
        });
        
        return vecs;
        
    }
    
    public static class KDTreeFactory<V extends Vec> implements VectorCollectionFactory<V>
    {
        private PivotSelection pvSelectionMethod;

        public KDTreeFactory(PivotSelection pvSelectionMethod)
        {
            this.pvSelectionMethod = pvSelectionMethod;
        }

        public KDTreeFactory()
        {
            this(PivotSelection.Variance);
        }
        
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return new KDTree<V>(source, distanceMetric, pvSelectionMethod);
        }
    }
}
