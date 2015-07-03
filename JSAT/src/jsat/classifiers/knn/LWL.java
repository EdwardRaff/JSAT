package jsat.classifiers.knn;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.classifiers.bayesian.NaiveBayesUpdateable;
import jsat.distributions.Distribution;
import jsat.distributions.discrete.UniformDiscrete;
import jsat.distributions.empirical.kernelfunc.*;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.TrainableDistanceMetric;
import jsat.linear.vectorcollection.*;
import jsat.parameters.*;
import jsat.regression.*;

/**
 * Locally Weighted Learner (LW) is the combined generalized implementation of 
 * Locally Weighted Regression (LWR) and Locally Weighted Naive Bayes (LWNB). 
 * The concept is simple, prediction begins in a normal NN style. Instead of 
 * returning the prediction result as the average / majority of the found 
 * neighbors, a classifier is trained to represent the local area that is 
 * represented by the neighbors. The prediction result is then based on this
 * locally trained model. <br>
 * Because NN search is already slow, and increasing the search size increases 
 * the search time, it is recommended to use moderate sized values of <i>k</i> 
 * with simple models such as {@link NaiveBayesUpdateable NaiveBayes} and 
 * {@link MultipleLinearRegression LinearRegression}. <br>
 * If the learning algorithm used does not support weighted data points, it will
 * be as if the {@link UniformKF uniform kernel fucntion} was used, regardless 
 * of whatever kernel function was set. <br>
 * <br>See:<br>
 * <ul>
 * <li>Atkeson, C., Moore, A.,&amp;Schaal, S. (1997). 
 * <a href="http://www.springerlink.com/index/G8280541763Q0223.pdf">Locally 
 * Weighted Learning</a>. Artificial intelligence review, 11–73.</li>
 * <li>Frank, E., Hall, M.,&amp;Pfahringer, B. (2003). 
 * <a href="http://dl.acm.org/citation.cfm?id=2100614">Locally Weighted Naive 
 * Bayes</a>. Proceedings of the Conference on Uncertainty in Artificial 
 * Intelligence (pp. 249–256). Morgan Kaufmann.</li>
 * </ul>
 * @author Edward Raff
 */
public class LWL implements Classifier, Regressor, Parameterized
{

    private static final long serialVersionUID = 6942465758987345997L;
    private CategoricalData predicting;
    private Classifier classifier;
    private Regressor regressor;
    private int k;
    private DistanceMetric dm;
    private KernelFunction kf;
    private VectorCollectionFactory<VecPaired<Vec, Double>> vcf;
    private VectorCollection<VecPaired<Vec, Double>> vc;

    /**
     * Copy constructor
     * @param toCopy the version to copy
     */
    private LWL(LWL toCopy)
    {
        if(toCopy.predicting != null)
            this.predicting = toCopy.predicting.clone();
        if(toCopy.classifier != null)
            setClassifier(toCopy.classifier);
        if(toCopy.regressor != null)
            setRegressor(toCopy.regressor);
        setNeighbors(toCopy.k);
        setDistanceMetric(toCopy.dm.clone());
        setKernelFunction(toCopy.kf);
        this.vcf = toCopy.vcf;
        if(toCopy.vc != null)
            this.vc = toCopy.vc.clone();
    }

    /**
     * Creates a new LWL classifier 
     * @param classifier the local classifier to
     * @param k the number of neighbors to create a local classifier from
     * @param dm the metric to use when selecting the nearest points to a query
     */
    public LWL(Classifier classifier, int k, DistanceMetric dm)
    {
        this(classifier, k, dm, EpanechnikovKF.getInstance());
    }
    
    /**
     * Creates a new LWL classifier 
     * @param classifier the local classifier to
     * @param k the number of neighbors to create a local classifier from
     * @param dm the metric to use when selecting the nearest points to a query
     * @param kf the kernel function used to weight the local points
     */
    public LWL(Classifier classifier, int k, DistanceMetric dm, KernelFunction kf)
    {
        this(classifier, k, dm, kf, new DefaultVectorCollectionFactory<VecPaired<Vec, Double>>());
    }
    
    /**
     * Creates a new LWL classifier 
     * @param classifier the local classifier to
     * @param k the number of neighbors to create a local classifier from
     * @param dm the metric to use when selecting the nearest points to a query
     * @param kf the kernel function used to weight the local points
     * @param vcf the factory to create vector collections for storing the points
     */
    public LWL(Classifier classifier, int k, DistanceMetric dm, KernelFunction kf, VectorCollectionFactory<VecPaired<Vec, Double>> vcf)
    {
        setClassifier(classifier);
        setNeighbors(k);
        setDistanceMetric(dm);
        setKernelFunction(kf);
        this.vcf = vcf;
    }
    
    /**
     * Creates a new LWL Regressor
     * @param regressor the local regressor
     * @param k the number of neighbors to create a local classifier from
     * @param dm the metric to use when selecting the nearest points to a query
     */
    public LWL(Regressor regressor, int k, DistanceMetric dm)
    {
        this(regressor, k, dm, EpanechnikovKF.getInstance());
    }
            
    /**
     * Creates a new LWL Regressor
     * @param regressor the local regressor
     * @param k the number of neighbors to create a local classifier from
     * @param dm the metric to use when selecting the nearest points to a query
     * @param kf the kernel function used to weight the local points
     */
    public LWL(Regressor regressor, int k, DistanceMetric dm, KernelFunction kf)
    {
        this(regressor, k, dm, kf, new DefaultVectorCollectionFactory<VecPaired<Vec, Double>>());
    }
    /**
     * Creates a new LWL Regressor
     * @param regressor the local regressor
     * @param k the number of neighbors to create a local classifier from
     * @param dm the metric to use when selecting the nearest points to a query
     * @param kf the kernel function used to weight the local points
     * @param vcf the factory to create vector collections for storing the points
     */
    public LWL(Regressor regressor, int k, DistanceMetric dm, KernelFunction kf, VectorCollectionFactory<VecPaired<Vec, Double>> vcf)
    {
        setRegressor(regressor);
        setNeighbors(k);
        setDistanceMetric(dm);
        setKernelFunction(kf);
        this.vcf = vcf;
    }
    
    
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(classifier == null || vc == null)
            throw new UntrainedModelException("Model has not been trained");
        
        List<? extends VecPaired<VecPaired<Vec, Double>, Double>> knn = 
                vc.search(data.getNumericalValues(), k);
       
        List<DataPointPair<Integer>> localPoints = new ArrayList<DataPointPair<Integer>>(knn.size());
        
        double maxD = knn.get(knn.size()-1).getPair();
        for(int i = 0; i < knn.size(); i++)
        {
            VecPaired<VecPaired<Vec, Double>, Double> v = knn.get(i);
            DataPoint dp = new DataPoint(v, new int[0], new CategoricalData[0], 
                    kf.k(v.getPair()/maxD));
            
            localPoints.add(new DataPointPair<Integer>(dp, v.getVector().getPair().intValue()));
        }

        ClassificationDataSet localSet = new ClassificationDataSet(localPoints, predicting);
        
        Classifier localClassifier = classifier.clone();
        localClassifier.trainC(localSet);
        
        return localClassifier.classify(data);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        List<VecPaired<Vec, Double>> trainList = getVecList(dataSet);
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadPool);
        vc = vcf.getVectorCollection(trainList, dm, threadPool);
        predicting = dataSet.getPredicting();
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        List<VecPaired<Vec, Double>> trainList = getVecList(dataSet);
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet);
        vc = vcf.getVectorCollection(trainList, dm);
        predicting = dataSet.getPredicting();
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public double regress(DataPoint data)
    {
        if(regressor == null || vc == null)
            throw new UntrainedModelException("Model has not been trained");
        
        List<? extends VecPaired<VecPaired<Vec, Double>, Double>> knn = 
                vc.search(data.getNumericalValues(), k);
       
        List<DataPointPair<Double>> localPoints = new ArrayList<DataPointPair<Double>>(knn.size());
        double maxD = knn.get(knn.size()-1).getPair();
        for(int i = 0; i < knn.size(); i++)
        {
            VecPaired<VecPaired<Vec, Double>, Double> v = knn.get(i);
            DataPoint dp = new DataPoint(v, new int[0], new CategoricalData[0], 
                    kf.k(v.getPair()/maxD));
            localPoints.add(new DataPointPair<Double>(dp, v.getVector().getPair()));
        }
        
        RegressionDataSet localSet = new RegressionDataSet(localPoints);
        
        Regressor localRegressor = regressor.clone();
        localRegressor.train(localSet);
        
        return localRegressor.regress(data);
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        List<VecPaired<Vec, Double>> trainList = getVecList(dataSet);
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet, threadPool);
        vc = vcf.getVectorCollection(trainList, dm, threadPool);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        List<VecPaired<Vec, Double>> trainList = getVecList(dataSet);
        
        TrainableDistanceMetric.trainIfNeeded(dm, dataSet);
        vc = vcf.getVectorCollection(trainList, dm);
    }

    @Override
    public LWL clone()
    {
        return new LWL(this);
    }

    private List<VecPaired<Vec, Double>> getVecList(ClassificationDataSet dataSet)
    {
        List<VecPaired<Vec, Double>> trainList = 
                new ArrayList<VecPaired<Vec, Double>>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            trainList.add(new VecPaired<Vec, Double>(
                    dataSet.getDataPoint(i).getNumericalValues(), 
                    new Double(dataSet.getDataPointCategory(i))));
        return trainList;
    }
    
    private List<VecPaired<Vec, Double>> getVecList(RegressionDataSet dataSet)
    {
        List<VecPaired<Vec, Double>> trainList = 
                new ArrayList<VecPaired<Vec, Double>>(dataSet.getSampleSize());
        for(int i = 0; i < dataSet.getSampleSize(); i++)
            trainList.add(new VecPaired<Vec, Double>(
                    dataSet.getDataPoint(i).getNumericalValues(), 
                    dataSet.getTargetValue(i)));
        return trainList;
    }

    private void setClassifier(Classifier classifier)
    {
        this.classifier = classifier;
        if(classifier instanceof Regressor)
            this.regressor = (Regressor) classifier;
    }

    private void setRegressor(Regressor regressor)
    {
        this.regressor = regressor;
        if(regressor instanceof Classifier)
            this.classifier = (Classifier)regressor;
    }

    /**
     * Sets the number of neighbors that will be used to create the local model
     * @param k the number of neighbors to obtain
     */
    public void setNeighbors(int k)
    {
        if(k <= 1)
            throw new RuntimeException("An average requires at least 2 neighbors to be taken into account");
        this.k = k;
    }

    /**
     * Returns the number of neighbors that will be used to create each local model
     * @return the number of neighbors that will be used
     */
    public int getNeighbors()
    {
        return k;
    }

    /**
     * Sets the distance metric that will be used for the nearest neighbor search
     * @param dm the distance metric to use for nearest neighbor search
     */
    public void setDistanceMetric(DistanceMetric dm)
    {
        this.dm = dm;
    }

    /**
     * Returns the distance metric in use
     * @return the distance metric in use
     */
    public DistanceMetric getDistanceMetric()
    {
        return dm;
    }
    
    /**
     * Sets the kernel function that will be used to set the weights of each 
     * data point in the local set
     * @param kf the kernel function to use for weighting
     */
    public void setKernelFunction(KernelFunction kf)
    {
        this.kf = kf;
    }

    /**
     * Returns the kernel function that will be used to set the weights. 
     * @return the kernel function that will be used to set the weights
     */
    public KernelFunction getKernelFunction()
    {
        return kf;
    }
    
    /**
     * Guesses the distribution to use for the number of neighbors to consider
     *
     * @param d the dataset to get the guess for
     * @return the guess for the Neighbors parameter
     */
    public static Distribution guessNeighbors(DataSet d)
    {
        return new UniformDiscrete(25, Math.min(200, d.getSampleSize()/5));
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
}
