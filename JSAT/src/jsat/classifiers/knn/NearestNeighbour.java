
package jsat.classifiers.knn;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.KDTree;
import jsat.linear.vectorcollection.VectorCollection;
import jsat.linear.vectorcollection.VectorCollectionFactory;

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
    
    private VectorCollectionFactory<VecPaired<Double, Vec>> vcf;
    private VectorCollection<VecPaired<Double, Vec>> vecCollection;

    private enum Mode {REGRESSION, CLASSIFICATION};
    /**
     * If we are in classification mode, the double is an integer that indicates class.
     */
    Mode mode;
    
    public NearestNeighbour(int k)
    {
        this(k, false);
    }
    
    public NearestNeighbour(int k, VectorCollectionFactory<VecPaired<Double, Vec>> vcf)
    {
        this(k, false, new EuclideanDistance(), vcf);
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
        this(k, weighted, distanceMetric, new KDTree.KDTreeFactory<VecPaired<Double, Vec>>());
    }
    
    public NearestNeighbour(int k, boolean weighted, DistanceMetric distanceMetric, VectorCollectionFactory<VecPaired<Double, Vec>> vcf )
    {
        this.mode = null;
        this.vecCollection = null;
        this.vcf = vcf;
        this.k = k;
        this.weighted = weighted;
        this.distanceMetric = distanceMetric;
    }

    public CategoricalResults classify(DataPoint data)
    {
        if(vecCollection == null || mode != Mode.CLASSIFICATION)
            throw new RuntimeException("Classifier has not been trained");
        Vec query  = data.getNumericalValues();
        
        List<VecPaired<Double,VecPaired<Double, Vec>>> knns = vecCollection.search(query, k);
        
        CategoricalResults results = new CategoricalResults(predicting.getNumOfCategories());
        
        for(int i = 0; i < knns.size(); i++)
        {
            double distance = knns.get(i).getPair();
            VecPaired<Double, Vec> pm = knns.get(i).getVector();
            int index =  (int) Math.round(pm.getPair());
            if(weighted)
            {
                double prob = -Math.exp(-distance);
                results.setProb(index, results.getProb(index) + prob);//Sum weights
            }
            else
                results.setProb(index, results.getProb(index) + 1.0);//all weights are 1
        }
        
        results.normalize();
        
        return results;
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
        List<VecPaired<Double, Vec>> dataPoints = new ArrayList<VecPaired<Double, Vec>>(dataSet.getSampleSize());
                
        //Add all the data points
        for(int i = 0; i < dataSet.getPredicting().getNumOfCategories(); i++)
        {
            for(DataPoint dp : dataSet.getSamples(i))
            {
                //We want to include the category in this case, so we will add it to the vector
                dataPoints.add(new VecPaired(dp.getNumericalValues(), (double)i));//bug? why isnt this auto boxed to double w/o a cast?
            }
        }
        
        vecCollection = vcf.getVectorCollection(dataPoints, distanceMetric);
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
