package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.utils.BoundedSortedList;
import jsat.utils.ProbailityMatch;

/**
 * Provides an implementation of Vantage Point Trees, as described in 
 * "Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces" 
 * by Peter N. Yianilos 
 * 
 * @author Edward Raff
 */
public class VPTree<V extends Vec> implements VectorCollection<V>
{
    private DistanceMetric dm;
    private Random rand;
    private int sampleSize;
    private int searchIterations;
    private TreeNode root;
    private VPSelection vpSelection;
    
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

    public VPTree(List<V> list, DistanceMetric dm, VPSelection vpSelection, Random rand, int sampleSize, int searchIterations)
    {
        this.dm = dm;
        if(!dm.isSubadditive())
            throw new RuntimeException("VPTree only supports metrics that support the triangle inequality");
        this.rand = rand;
        this.sampleSize = sampleSize;
        this.searchIterations = searchIterations;
        List<ProbailityMatch<V>> tmpList = new ArrayList<ProbailityMatch<V>>(list.size());
        for(V v : list)
            tmpList.add(new ProbailityMatch<V>(-1, v));
        this.root = makeVPTree(tmpList);
    }

    public VPTree(List<V> list, DistanceMetric dm, VPSelection vpSelection)
    {
        this(list, dm, vpSelection, new Random(), 80, 40);
    }
    public VPTree(List<V> list, DistanceMetric dm)
    {
        this(list, dm, VPSelection.Random);
    }
    
    public List<VecPaired<Double, V>> search(Vec query, double range)
    {
        if(range <= 0)
            throw new RuntimeException("Range must be a positive number");
        List<VecPaired<Double, V>> returnList = new ArrayList<VecPaired<Double, V>>();
        
        root.searchRange(VecPaired.extractTrueVec(query), range, returnList);
        
        Collections.sort(returnList, new Comparator<VecPaired<Double, V>>() {

            public int compare(VecPaired<Double, V> o1, VecPaired<Double, V> o2)
            {
                return Double.compare(o1.getPair(), o2.getPair());
            }
        });
        
        return returnList;
    }
    
    public List<VecPaired<Double, V>> search(Vec query, int neighbors)
    {
        BoundedSortedList<ProbailityMatch<V>> boundedList= new BoundedSortedList<ProbailityMatch<V>>(neighbors, neighbors);

        root.searchKNN(VecPaired.extractTrueVec(query), neighbors, boundedList);
        
        List<VecPaired<Double, V>> list = new ArrayList<VecPaired<Double, V>>(boundedList.size());
        for(ProbailityMatch<V> pm : boundedList)
            list.add(new VecPaired<Double, V>(pm.getMatch(), pm.getProbability()));
        return list;
    }
    
    //The probability match is used to store and sort by median distances. 
    private TreeNode makeVPTree(List<ProbailityMatch<V>> S)
    {
        if(S.isEmpty())
            return null;
        else if(S.size() <= 5)
        {
            List<V> leafs = new ArrayList<V>(S.size());
            for(ProbailityMatch<V> pm : S)
                leafs.add(pm.getMatch());
            S.clear();
            return new VPLeaf(leafs);
        }
        
        VPNode node = new VPNode(selectVantagePoint(S));
        
        //Compute distance to each point
        for(int i = 0; i < S.size(); i++)
            S.get(i).setProbability(dm.dist(node.p, S.get(i).getMatch()));//Each point gets its distance to the vantage point
        Collections.sort(S);//Get median and split lists into 2 groups
        int medianIndex = S.size() / 2;
        node.mu = S.get(medianIndex).getProbability();
        node.right_high = S.get(S.size()-1).getProbability();
        node.right_low = S.get(medianIndex+1).getProbability();
        node.left_high = S.get(medianIndex).getProbability();
        node.left_low = S.get(0).getProbability();
        
        /*
         * Re use the list and let it get altered. We must compute the right side first. 
         * If we altered the left side, the median would move left, and the right side 
         * would get thrown off or require aditonal book keeping. 
         */
        node.right = makeVPTree(S.subList(medianIndex+1, S.size()));
        node.left  = makeVPTree(S.subList(0, medianIndex+1));
        
        return node;
    }
    
    /**
     * Determines what point from the data set will become a vantage point, and removes it from the list
     * @param S the set to select a vantage point from
     * @return the vantage point removed from the set
     */
    private V selectVantagePoint(List<ProbailityMatch<V>> S)
    {
        if (vpSelection == VPSelection.Random)
            return S.remove(rand.nextInt(S.size())).getMatch();
        else//Sampling
        {
            List<V> samples = new ArrayList<V>(sampleSize);
            if (sampleSize <= S.size())
                for (int i = 0; i < sampleSize; i++)
                    samples.add(S.get(i).getMatch());
            else
                for (int i = 0; i < sampleSize; i++)
                    samples.add(S.get(rand.nextInt(S.size())).getMatch());

            double[] distances = new double[sampleSize];

            int bestVP = -1;
            double bestSpread = 0;

            for (int i = 0; i < Math.min(searchIterations, S.size()); i++)
            {
                //When low on samples, just brute force!
                int candIndx = searchIterations <= S.size() ? i : rand.nextInt(S.size());
                V candV = S.get(candIndx).getMatch();

                for (int j = 0; j < samples.size(); j++)
                    distances[j] = dm.dist(candV, samples.get(j));

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

            return S.remove(bestVP).getMatch();
        }
    }
    
    private abstract class TreeNode
    {
        public abstract void searchKNN(Vec query, int k, BoundedSortedList<ProbailityMatch<V>> list);
        public abstract void searchRange(Vec query, double range, List<VecPaired<Double, V>> list);
    }
    
    private class VPNode extends TreeNode
    {
        V p;
        double mu, left_low, left_high, right_low, right_high;
        TreeNode right, left;

        public VPNode(V p)
        {
            this.p = p;
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
        
        public void searchKNN(Vec query, int k, BoundedSortedList<ProbailityMatch<V>> list)
        {
            double x = dm.dist(query, this.p);
            if(list.size() < k || x < list.get(k-1).getProbability())
                list.add(new ProbailityMatch<V>(x, this.p));
            double tau = list.get(list.size()-1).getProbability();
            double middle = (this.left_high+this.right_low)*0.5;

            if( x < middle)
            {
                if(searchInLeft(x, tau) || list.size() < k)
                    this.left.searchKNN(query, k, list);
                tau = list.get(list.size()-1).getProbability();
                if(searchInRight(x, tau) || list.size() < k)
                    this.right.searchKNN(query, k, list);
            }
            else
            {
                if(searchInRight(x, tau) || list.size() < k)
                    this.right.searchKNN(query, k, list);
                tau = list.get(list.size()-1).getProbability();
                if(searchInLeft(x, tau) || list.size() < k)
                    this.left.searchKNN(query, k, list);
            }
        }

        @Override
        public void searchRange(Vec query, double range, List<VecPaired<Double, V>> list)
        {
            double x = dm.dist(query, this.p);
            if(x <= range)
                list.add(new VecPaired<Double, V>(this.p, x));

            if (searchInLeft(x, range))
                this.left.searchRange(query, range, list);
            if (searchInRight(x, range))
                this.right.searchRange(query, range, list);
        }
    }
    
    private class VPLeaf extends TreeNode
    {
        //TODO add code to keep track of the distance from these leaf points to their parent, perform pruning anyway b/c distance comps are expensive. 
        List<V> points;
        
        public VPLeaf(List<V> points)
        {
            this.points = points;
        }

        @Override
        public void searchKNN(Vec query, int k, BoundedSortedList<ProbailityMatch<V>> list)
        {
            double dist = -1;
            for(V v : points)
                if(list.size() < k || (dist = dm.dist(query, v)) < list.get(list.size()-1).getProbability())
                    list.add(new ProbailityMatch<V>(dist, v));
        }

        @Override
        public void searchRange(Vec query, double range, List<VecPaired<Double, V>> list)
        {
            double dist = Double.MAX_VALUE;
            for(int i = 0; i < points.size(); i++)
                if( (dist = dm.dist(query, points.get(i))) < range )
                    list.add(new VecPaired<Double, V>(points.get(i), dist));
        }
        
    }
    
    public static class VPTreeFactory<V extends Vec> implements VectorCollectionFactory<V>
    {
        private VPSelection vpSelectionMethod;

        public VPTreeFactory(VPSelection vpSelectionMethod)
        {
            this.vpSelectionMethod = vpSelectionMethod;
        }

        public VPTreeFactory()
        {
            this(VPSelection.Random);
        }
        
        public VectorCollection<V> getVectorCollection(List<V> source, DistanceMetric distanceMetric)
        {
            return new VPTree<V>(source, distanceMetric, vpSelectionMethod);
        }
        
    }
}
