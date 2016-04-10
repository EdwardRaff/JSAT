
package jsat.classifiers.knn;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.distributions.Distribution;
import jsat.distributions.discrete.UniformDiscrete;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.*;
import jsat.linear.vectorcollection.*;
import jsat.math.MathTricks;
import jsat.math.SpecialMath;
import jsat.parameters.*;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;

/**
 * An implementation of the Nearest Neighbor algorithm, but with a 
 * British spelling! How fancy. 
 * @author Edward Raff
 */
public class NearestNeighbour implements  Classifier, Regressor, Parameterized
{

    private static final long serialVersionUID = 4239569189624285932L;
    private int k;
    private boolean weighted;
    private DistanceMetric distanceMetric;
    private CategoricalData predicting;
    
    private VectorCollectionFactory<VecPaired<Vec, Double>> vcf;
    private VectorCollection<VecPaired<Vec, Double>> vecCollection;

    /**
     * Returns the number of neighbors currently consulted to make decisions
     * @return the number of neighbors
     */
    public int getNeighbors()
    {
        return k;
    }

    /**
     * Sets the number of neighbors to consult when making decisions
     * @param k the number of neighbors to use
     */
    public void setNeighbors(int k)
    {
        if(k < 1)
            throw new ArithmeticException("Must be a positive number of neighbors");
        this.k = k;
    }
    
    public int getNeighbors(int k)
    {
        return k;
    }

    public DistanceMetric getDistanceMetric()
    {
        return distanceMetric;
    }

    public void setDistanceMetric(DistanceMetric distanceMetric)
    {
        if(distanceMetric == null)
            throw new NullPointerException("given metric was null");
        this.distanceMetric = distanceMetric;
    }
    
    

    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }

    private enum Mode {REGRESSION, CLASSIFICATION};
    /**
     * If we are in classification mode, the double is an integer that indicates class.
     */
    Mode mode;
    
    /**
     * Constructs a new Nearest Neighbor Classifier
     * @param k the number of neighbors to use
     */
    public NearestNeighbour(int k)
    {
        this(k, false);
    }
    
    /**
     * Constructs a new Nearest Neighbor Classifier
     * @param k the number of neighbors to use
     * @param vcf the vector collection factory to use for storing and querying 
     */
    public NearestNeighbour(int k, VectorCollectionFactory<VecPaired<Vec, Double>> vcf)
    {
        this(k, false, new EuclideanDistance(), vcf);
    }

    /**
     * Constructs a new Nearest Neighbor Classifier
     * @param k the number of neighbors to use
     * @param weighted whether or not to weight the influence of neighbors by their distance
     */
    public NearestNeighbour(int k, boolean weighted)
    {
        this(k, weighted, new EuclideanDistance());
    }
    
    /**
     * Constructs a new Nearest Neighbor Classifier
     * @param k the number of neighbors to use
     * @param weighted whether or not to weight the influence of neighbors by their distance
     * @param distanceMetric the method of computing distance between two vectors. 
     */
    public NearestNeighbour(int k, boolean weighted, DistanceMetric distanceMetric )
    {
        this(k, weighted, distanceMetric, new DefaultVectorCollectionFactory<VecPaired<Vec, Double>>());
    }
    
    /**
     * Constructs a new Nearest Neighbor Classifier
     * @param k the number of neighbors to use
     * @param weighted whether or not to weight the influence of neighbors by their distance
     * @param distanceMetric the method of computing distance between two vectors. 
     * @param vcf the vector collection factory to use for storing and querying 
     */
    public NearestNeighbour(int k, boolean weighted, DistanceMetric distanceMetric, VectorCollectionFactory<VecPaired<Vec, Double>> vcf )
    {
        this.mode = null;
        this.vcf = vcf;
        this.k = k;
        this.weighted = weighted;
        this.distanceMetric = distanceMetric;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(vecCollection == null || mode != Mode.CLASSIFICATION)
            throw new UntrainedModelException("Classifier has not been trained for classification");
        Vec query  = data.getNumericalValues();
        
        List<? extends VecPaired<VecPaired<Vec, Double>, Double>> knns = vecCollection.search(query, k);
        
        if(weighted)
        {
            double[] dists = new double[knns.size()];
            for(int i = 0; i < knns.size(); i++)
                dists[i] = knns.get(i).getPair();
            //we want something like 1/dist to make closer values weighted higher
            double offset = MathTricks.min(dists)*0.1+1e-15;
            for(int i = 0; i < knns.size(); i++)
                dists[i] = 1/(offset+dists[i]);
            
            MathTricks.softmax(dists, false);
            
            CategoricalResults results = new CategoricalResults(predicting.getNumOfCategories());
            for(int i = 0; i < knns.size(); i++)
            {
                VecPaired<Vec, Double> pm = knns.get(i).getVector();
                int index =  (int) Math.round(pm.getPair());

                results.setProb(index, results.getProb(index) + dists[i]);//all weights are 1
            }
            
            return results;
        }
        else
        {
            CategoricalResults results = new CategoricalResults(predicting.getNumOfCategories());
        
            for(int i = 0; i < knns.size(); i++)
            {
                VecPaired<Vec, Double> pm = knns.get(i).getVector();
                int index =  (int) Math.round(pm.getPair());

                results.setProb(index, results.getProb(index) + 1.0);//all weights are 1
            }

            results.normalize();

            return results;
        }
    }
    
    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null); 
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if(dataSet.getNumCategoricalVars() != 0)
            throw new FailedToFitException("KNN requires vector data only");
        
        mode = Mode.CLASSIFICATION;
        this.predicting = dataSet.getPredicting();
        List<VecPaired<Vec, Double>> dataPoints = new ArrayList<VecPaired<Vec, Double>>(dataSet.getSampleSize());
                
        //Add all the data points
        for(int i = 0; i < dataSet.getClassSize(); i++)
        {
            for(DataPoint dp : dataSet.getSamples(i))
            {
                //We want to include the category in this case, so we will add it to the vector
                dataPoints.add(new VecPaired(dp.getNumericalValues(), (double)i));//bug? why isnt this auto boxed to double w/o a cast?
            }
        }
        
        TrainableDistanceMetric.trainIfNeeded(distanceMetric, dataSet, threadPool);
        
        if(threadPool == null)
            vecCollection = vcf.getVectorCollection(dataPoints, distanceMetric);
        else
            vecCollection = vcf.getVectorCollection(dataPoints, distanceMetric, threadPool);
    }
    
    @Override
    public double regress(DataPoint data)
    {
        if(vecCollection == null || mode != Mode.REGRESSION)
            throw new UntrainedModelException("Classifier has not been trained for regression");
        Vec query  = data.getNumericalValues();
        
        List<? extends VecPaired<VecPaired<Vec, Double>, Double>> knns = vecCollection.search(query, k);
        
        double result = 0, weightSum = 0;
        
        for(int i = 0; i < knns.size(); i++)
        {
            double distance = knns.get(i).getPair();
            VecPaired<Vec, Double> pm = knns.get(i).getVector();
            
            double value = pm.getPair();
            
            
            if(weighted)
            {
                distance = Math.max(1e-8, distance);//Avoiding zero distances which will result in Infinty getting propigated around
                double weight = 1.0/Math.pow(distance, 2);
                weightSum += weight;
                result += value*weight;
            }
            else
            {
                result += value;
                weightSum += 1;
            }
        }
        
        return result/weightSum;
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, null);
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        if(dataSet.getNumCategoricalVars() != 0)
            throw new FailedToFitException("KNN requires vector data only");
        
        mode = Mode.REGRESSION;

        List<VecPaired<Vec, Double>> dataPoints = new ArrayList<VecPaired<Vec, Double>>(dataSet.getSampleSize());
                
        //Add all the data points
        for (int i = 0; i < dataSet.getSampleSize(); i++)
        {
            DataPointPair<Double> dpp = dataSet.getDataPointPair(i);

            dataPoints.add(new VecPaired(dpp.getVector(), dpp.getPair()));
        }
        
        TrainableDistanceMetric.trainIfNeeded(distanceMetric, dataSet, threadPool);

        if(threadPool == null)
            vecCollection = vcf.getVectorCollection(dataPoints, distanceMetric);
        else
            vecCollection = vcf.getVectorCollection(dataPoints, distanceMetric, threadPool);
    }
    
    @Override
    public NearestNeighbour clone()
    {
        NearestNeighbour clone = new NearestNeighbour(k, weighted, distanceMetric.clone(), vcf.clone());
        
        if(this.predicting != null)
            clone.predicting = this.predicting.clone();
        clone.mode = this.mode;
        
        if(this.vecCollection != null)
            clone.vecCollection = this.vecCollection.clone();
        
        return clone;
    }
    
    /**
     * Guesses the distribution to use for the number of neighbors to consider
     * @param d the dataset to get the guess for
     * @return the guess for the Neighbors parameter
     */
    public static Distribution guessNeighbors(DataSet d)
    {
        return new UniformDiscrete(1, 25);
    }
    
    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }
}
