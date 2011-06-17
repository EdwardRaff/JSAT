
package jsat.classifiers.knn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.DenseVector;
import jsat.linear.SparceVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.BoundedSortedSet;
import jsat.utils.ProbailityMatch;




/**
 *
 * @author Edward Raff
 */
public class NearestNeighbourKDTree implements Classifier
{
    private int k;
    private boolean weighted ;
    private DistanceMetric distanceMetric;
    private CategoricalData predicting;
    
    List<DataPointPair<Double>> dataPoints;

    private enum Mode {REGRESSION, CLASSIFICATION};
    /**
     * If we are in classification mode, the double is an integer that indicates class.
     */
    Mode mode;
    KDNode root;
    
    public NearestNeighbourKDTree(int k)
    {
        this(k, false);
    }

    public NearestNeighbourKDTree(int k, boolean weighted)
    {
        this(k, weighted, new EuclideanDistance());
    }
    
    /**
     * Constructs a new Nearest Neighbor Classifier
     * @param k the number of neighbors to examine
     * @param weighted whether or not to weight the influence of neighbors by their distance
     * @param distanceMetric the method of computing distance between two vectors. 
     */
    public NearestNeighbourKDTree(int k, boolean weighted, DistanceMetric distanceMetric )
    {
        this.mode = null;
        this.k = k;
        this.weighted = weighted;
        this.distanceMetric = distanceMetric;
    }
    
    private class KDNode 
    {
        DataPointPair<Double> locatin;
        int axis;

        KDNode left;
        KDNode right;
        
        public KDNode(DataPointPair<Double> locatin, int axis)
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

        public void setLocatin(DataPointPair<Double> locatin)
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

        public DataPointPair<Double> getLocatin()
        {
            return locatin;
        }

        public KDNode getRight()
        {
            return right;
        }
        
    }

    public CategoricalResults classify(DataPoint data)
    {
        if(root == null || mode != Mode.CLASSIFICATION)
            throw new RuntimeException("Classifier has not been trained");
        //We are using the probability to store distance, lower is better, so it all works out
        BoundedSortedSet<ProbailityMatch<DataPointPair<Double>>> knns =
                new BoundedSortedSet<ProbailityMatch<DataPointPair<Double>>>(k);
        Vec query  = data.getNumericalValues();

        knnKDSearch(query, root, knns);
        
        CategoricalResults results = new CategoricalResults(predicting.getNumOfCategories());
        double divisor = 0;
        
        for(ProbailityMatch<DataPointPair<Double>> pm : knns)
        {
            int index =  (int) Math.round(pm.getMatch().getPair());
            if(weighted)
            {
                double prob = -Math.exp(-pm.getProbability());
                divisor += prob;
                results.setProb(index, results.getProb(index) + prob);//Sum weights
            }
            else
                results.setProb(index, results.getProb(index) + 1.0);//all weights are 1
        }
        
        if(!weighted)
            divisor = knns.size();
                
        
        results.divideConst(divisor);
        
        return results;
    }
    
    private void knnKDSearch(Vec query, KDNode node, BoundedSortedSet<ProbailityMatch<DataPointPair<Double>>> knns)
    {
        if(node == null)
            return;
        DataPointPair<Double> curData = node.locatin;
        double distance = distanceMetric.dist(query, curData.getDataPoint().getNumericalValues());
        
        knns.add( new ProbailityMatch<DataPointPair<Double>>(distance, curData));
        
        double diff = query.get(node.axis) - curData.getDataPoint().getNumericalValues().get(node.axis);
        
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

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet); 
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        if(dataSet.getNumCategoricalVars() != 0)
            throw new RuntimeException("KNN requires vector data only");
        
        mode = Mode.CLASSIFICATION;
        this.predicting = dataSet.getPredicting();
        dataPoints = new ArrayList<DataPointPair<Double>>(dataSet.getSampleSize());
        
        //Add all the data points
        for(int i = 0; i < dataSet.getPredicting().getNumOfCategories(); i++)
        {
            for(DataPoint dp : dataSet.getSamples(i))
            {
                //We want to include the category in this case, so we will add it to the vector
                dataPoints.add(new DataPointPair(dp, (double)i));//bug? why isnt this auto boxed to double w/o a cast?
            }
        }
        
        root = buildTree(dataPoints, 0);
    }
    
    private class VecIndexComparator implements Comparator<DataPointPair>
    {
        int index;

        public VecIndexComparator(int index)
        {
            this.index = index;
        }
        
        public int compare(DataPointPair o1, DataPointPair o2)
        {
            return Double.compare(
                    o1.getDataPoint().getNumericalValues().get(index), 
                    o2.getDataPoint().getNumericalValues().get(index));
        }
        
    }
    
    KDNode buildTree(List<DataPointPair<Double>> data, int depth)
    {
        if(data == null || data.isEmpty())
            return null;
        int mod = data.get(0).getDataPoint().numNominalValues();
        
        int axi = depth % mod;
        
        Collections.sort(data, new VecIndexComparator(axi));
        
        int medianIndex = data.size()/2;
        DataPointPair<Double> median = data.get(medianIndex);
        
        KDNode node = new KDNode(median, axi);
        
        node.setLeft(buildTree(data.subList(0, medianIndex), depth+1));
        node.setRight(buildTree(data.subList(medianIndex+1, data.size()), depth+1));
        
        return node;
    }

    public Classifier copy()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
    public boolean supportsWeightedData()
    {
        return false;
    }
}
