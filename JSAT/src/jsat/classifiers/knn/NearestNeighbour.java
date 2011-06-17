
package jsat.classifiers.knn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.BoundedSortedSet;
import jsat.utils.ProbailityMatch;

/**
 *
 * @author Edward Raff
 */
public class NearestNeighbour implements  Classifier
{
    private int k;
    private boolean weighted ;
    private DistanceMetric distanceMetric;
    private CategoricalData predicting;
    
    int[] classification;
    List<DataPoint> dataPoints;
    
    public NearestNeighbour(int k)
    {
        this(k, false);
    }

    public NearestNeighbour(int k, boolean weighted)
    {
        this(k, weighted, new EuclideanDistance());
    }
    
    /**
     * Constructs a new Nearest Neighbor Classifier
     * @param k the number of neighbors to examine
     * @param weighted whether or not to weight the influence of neighbors by their distance
     * @param distanceMetric the method of computing distance between two vectors. 
     */
    public NearestNeighbour(int k, boolean weighted, DistanceMetric distanceMetric )
    {
        this.k = k;
        this.weighted = weighted;
        this.distanceMetric = distanceMetric;
    }
     
    public CategoricalResults classify(DataPoint data)
    {
        if(dataPoints == null)
            throw new RuntimeException("Classifier has not yet been trained");
        
        BoundedSortedSet<ProbailityMatch<Integer>> closestMatches = 
                new BoundedSortedSet<ProbailityMatch<Integer>>(k);
        
        //Divides all the result probabilities so they sum to one
        //if not weighted, divosior = k
        //if weigthed, it is the sum of the weithed distances will have to be set at the end 
        
        double divisor = 0;
        
        for(int i = 0; i < dataPoints.size(); i++)
        {
            Vec v = dataPoints.get(i).getNumericalValues();
            double distance = distanceMetric.dist(v, data.getNumericalValues());
            
            
            
            closestMatches.add(new ProbailityMatch<Integer>(distance, classification[i]));
        }
        
        CategoricalResults results = new CategoricalResults(predicting.getNumOfCategories());
        
        for(ProbailityMatch<Integer> pm : closestMatches)
        {
            
            if(weighted)
            {
                double prob = pm.getProbability();
                
                //Normaly the weigth by this method we choose the highest value isntead of the lowest
                //But we dont want to change our BoundedSOrtedSet
                //So we change the signs, so the |largest| will be at the front of the list
                if(weighted)
                    prob = -Math.exp(-prob);
                
                divisor += prob;
                results.setProb(pm.getMatch(), results.getProb(pm.getMatch()) + prob);//Sum weights
            }
            else
                results.setProb(pm.getMatch(), results.getProb(pm.getMatch()) + 1.0);//all weights are 1
        }
        
        if(!weighted)
            divisor = closestMatches.size();
                
        
        results.divideConst(divisor);
        
        return results;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        this.predicting = dataSet.getPredicting();
        dataPoints = new ArrayList<DataPoint>(dataSet.getSampleSize());
        classification = new int[dataSet.getSampleSize()];
        
        //Add all the data points
        for(int i = 0; i < dataSet.getPredicting().getNumOfCategories(); i++)
        {
            List<DataPoint> some = dataSet.getSamples(i);
            for(int z = 0; z < some.size(); z++)//include a matching category
                classification[dataPoints.size()+z] = i;
            dataPoints.addAll(some); 
        }
    }

    public Classifier copy()
    {
        NearestNeighbour copy = new NearestNeighbour(k, weighted, distanceMetric);
        
        copy.classification = Arrays.copyOf(classification, classification.length);
        copy.dataPoints = new ArrayList<DataPoint>(dataPoints.size());
        
        copy.dataPoints.addAll(this.dataPoints);
        copy.predicting = this.predicting;
        
        return copy;
    }

    public boolean supportsWeightedData()
    {
        return false;
    }
}
