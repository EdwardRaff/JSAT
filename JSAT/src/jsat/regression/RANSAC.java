package jsat.regression;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.exceptions.FailedToFitException;
import jsat.parameters.*;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.utils.FakeExecutor;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;

/**
 * RANSAC is a randomized meta algorithm that is useful for fitting a model to a
 * data set that has a large amount of outliers that do not represent the true 
 * distribution well. <br>
 * RANSAC has the concept of inliers and outliers. An initial number of seed 
 * points is specified. This makes the initial inlier set.  The algorithm than 
 * iterates several times, randomly selecting the specified number of points. It
 * then regresses on all other points, adding all points within a specified 
 * absolute error to the set of inliers. The model is then trained again on the 
 * larger set, and the training error becomes the measure of the strength of the
 * model. The model that has the lowest error is then the fit model. 
 * 
 * @author Edward Raff
 */
public class RANSAC implements Regressor, Parameterized
{

	private static final long serialVersionUID = -5015748604828907703L;
	/**
     * the minimum number of data required to fit the model
     */
    private int initialTrainSize;
    /**
     * the number of iterations performed by the algorithm
     */
    private int iterations;
    /**
     * a threshold value for determining when a datum fits a model
     */
    private double maxPointError;
    
    /**
     * the number of close data values required to assert that a model fits well to data
     */
    private int minResultSize;
    
    @ParameterHolder
    private Regressor baseRegressor;
    /**
     * True marks that the data point is part of the consensus set. 
     * False indicates it is not. 
     */
    private boolean[] consensusSet;
    private double modelError;

    /**
     * Creates a new RANSAC training object. Because RANSAC is sensitive to 
     * parameter settings, which are data and model dependent, no default values
     * exist for them. 
     * 
     * @param baseRegressor the model to fit using RANSAC
     * @param iterations the number of iterations of the algorithm to perform
     * @param initialTrainSize the number of points to seed each iteration of 
     * training with
     * @param minResultSize the minimum number of inliers to make it into the 
     * model to be considered a possible fit. 
     * @param maxPointError the maximum allowed absolute difference in the 
     * output of the model and the true value for the data point to be added to
     * the inlier set. 
     */
    public RANSAC(Regressor baseRegressor, int iterations, int initialTrainSize, int minResultSize, double maxPointError)
    {
        setInitialTrainSize(initialTrainSize);
        setIterations(iterations);
        setMaxPointError(maxPointError);
        setMinResultSize(minResultSize);
        this.baseRegressor = baseRegressor;
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
    
    /**
     * class that does the loop iteration work and returns a reference to 
     * itself. The are sortable based on the lowest error
     */
    private class RANSACWorker implements Callable<RANSACWorker>, Comparable<RANSACWorker>
    {
        int maxIterations;
        RegressionDataSet dataset;
        Random rand;
        Regressor baseModel;
        public RANSACWorker(Regressor baseModel, int maxIterations, RegressionDataSet dataset)
        {
            this.baseModel = baseModel;
            this.maxIterations = maxIterations;
            this.dataset = dataset;
            rand = RandomUtil.getRandom();
        }
        
        
        //To be determined
        Regressor bestModel = null;
        boolean[] bestConsensusSet = null;
        double bestError = Double.POSITIVE_INFINITY;

        @Override
        public RANSACWorker call() throws Exception
        {
            bestConsensusSet = new boolean[dataset.getSampleSize()];
            
            boolean[] working_set = new boolean[dataset.getSampleSize()];
            
            Set<Integer> maybe_inliers = new IntSet(initialTrainSize*2);
            
            for(int iter = 0; iter < maxIterations; iter++)
            {
                //Create sub data set sample
                maybe_inliers.clear();
                Arrays.fill(working_set, false);
                while(maybe_inliers.size() < initialTrainSize)
                    maybe_inliers.add(rand.nextInt(working_set.length));
                int consensusSize = maybe_inliers.size();
                RegressionDataSet subDataSet = new RegressionDataSet(dataset.getNumNumericalVars(), dataset.getCategories());
                for(int i : maybe_inliers)
                {
                    subDataSet.addDataPointPair(dataset.getDataPointPair(i));
                    working_set[i] = true;
                }
                Regressor maybeModel = baseModel.clone();
                maybeModel.train(subDataSet);
                
                //Build consensus set
                for(int i = 0; i < working_set.length; i++)
                {
                    if(working_set[i])
                        continue;//Already part of the model
                    
                    DataPointPair<Double> dpp = dataset.getDataPointPair(i);
                    double guess = maybeModel.regress(dpp.getDataPoint());
                    double diff = Math.abs(guess - dpp.getPair());
                    
                    if(diff < maxPointError)
                    {
                        working_set[i] = true;//Add tot he consenus set
                        subDataSet.addDataPointPair(dpp);
                        consensusSize++;
                    }
                }
                
                
                if(consensusSize < minResultSize )
                    continue;//We did not fit enough points to be considered
                //Build final model
                maybeModel.train(subDataSet);
                //Copmute final model error on the consenus set
                double thisError = 0;
                for(int i = 0; i < working_set.length; i++)
                {
                    if(!working_set[i])
                        continue;
                    DataPointPair<Double> dpp = dataset.getDataPointPair(i);
                    double guess = maybeModel.regress(dpp.getDataPoint());
                    double diff = Math.abs(guess - dpp.getPair());
                    thisError += diff;
                }
                
                if(thisError < bestError)//New best model
                {
                    bestError = thisError;
                    bestModel = maybeModel;
                    System.arraycopy(working_set, 0, bestConsensusSet, 0, working_set.length);
                }
                
            }
            
            return this;
        }

        @Override
        public int compareTo(RANSACWorker o)
        {
            return Double.compare(this.bestError, o.bestError);
        }
        
    }

    /**
     * Returns the number of data points to be sampled from the training set to 
     * create initial models. 
     * 
     * @return the number of data points used to first create models
     */
    public int getInitialTrainSize()
    {
        return initialTrainSize;
    }

    /**
     * Sets  the number of data points to be sampled from the training set to 
     * create initial models. 
     * 
     * @param initialTrainSize the number of data points to use to create models
     */
    public void setInitialTrainSize(int initialTrainSize)
    {
        if(initialTrainSize < 1)
            throw new RuntimeException("Can not train on an empty data set");
        this.initialTrainSize = initialTrainSize;
    }

    /**
     * Returns the number models that will be tested on the data set. 
     * 
     * @return the number of algorithm iterations
     */
    public int getIterations()
    {
        return iterations;
    }

    /**
     * Sets the number models that will be tested on the data set. 
     * @param iterations the number of iterations to perform
     */
    public void setIterations(int iterations)
    {
        if(iterations < 1)
            throw new RuntimeException("Must perform a positive number of iterations");
        this.iterations = iterations;
    }

    /**
     * Each data point not in the initial training set will be tested against. 
     * If a data points error is sufficiently small, it will be added to the set
     * of inliers. 
     * 
     * @return the maximum error any one point may have to be an inliner
     */
    public double getMaxPointError()
    {
        return maxPointError;
    }

    /**
     * Each data point not in the initial training set will be tested against. 
     * If a data points error is sufficiently small, it will be added to the set
     * of inliers.
     * 
     * @param maxPointError the new maximum error a data point may have to be 
     * considered an inlier. 
     */
    public void setMaxPointError(double maxPointError)
    {
        if(maxPointError < 0 || Double.isInfinite(maxPointError) || Double.isNaN(maxPointError))
            throw new ArithmeticException("The error must be a positive value, not " + maxPointError );
        this.maxPointError = maxPointError;
    }

    /**
     * RANSAC requires an initial model to be accurate enough to include a 
     * minimum number of inliers before being considered as a potentially good 
     * model. This is the number of points that must make it into the inlier set
     * for a model to be considered. 
     * 
     * @return the minimum number of inliers to be considered
     */
    public int getMinResultSize()
    {
        return minResultSize;
    }

    /**
     * RANSAC requires an initial model to be accurate enough to include a 
     * minimum number of inliers before being considered as a potentially good 
     * model. This is the number of points that must make it into the inlier set
     * for a model to be considered. 
     * 
     * @param minResultSize the minimum number of inliers to be considered
     */
    public void setMinResultSize(int minResultSize)
    {
        if(minResultSize < getInitialTrainSize())
            throw new RuntimeException("The min result size must be larger than the intial train size");
        this.minResultSize = minResultSize;
    }

    
    @Override
    public double regress(DataPoint data)
    {
        return baseRegressor.regress(data);
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        try
        {
            
            int workSize = iterations/SystemInfo.LogicalCores;
            int leftOver = iterations%SystemInfo.LogicalCores;
            
            List<Future<RANSACWorker>> futures = new ArrayList<Future<RANSACWorker>>(SystemInfo.LogicalCores+1);
            if(leftOver != 0)
                futures.add(threadPool.submit(new RANSACWorker(baseRegressor, leftOver, dataSet)));
            for(int i = 0; i < SystemInfo.LogicalCores; i++)
                futures.add(threadPool.submit(new RANSACWorker(baseRegressor, workSize, dataSet)));
            
            PriorityQueue<RANSACWorker> results = new PriorityQueue<RANSACWorker>(SystemInfo.LogicalCores+1);
            
            for( Future<RANSACWorker> futureWorker : futures )
                results.add(futureWorker.get());
            
            RANSACWorker bestResult = results.peek();
            
            modelError = bestResult.bestError;
            if(Double.isInfinite(modelError))
                throw new FailedToFitException("Model could not be fit, inlier set never reach minimum size");
            baseRegressor = bestResult.bestModel;
            consensusSet = bestResult.bestConsensusSet;
            
            
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(RANSAC.class.getName()).log(Level.SEVERE, null, ex);
            throw new FailedToFitException(ex);
        }
        catch (ExecutionException ex)
        {
            Logger.getLogger(RANSAC.class.getName()).log(Level.SEVERE, null, ex);
            throw new FailedToFitException(ex);
        }
                
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, new FakeExecutor());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return baseRegressor.supportsWeightedData();
    }

    @Override
    public RANSAC clone()
    {
        RANSAC clone = new RANSAC(baseRegressor.clone(), iterations, initialTrainSize, minResultSize, maxPointError);
        
        return clone;
    }
    
    /**
     * Once RANSAC is complete, it maintains its trained version of the 
     * finalized regressor. A clone of it may be retrieved from this method. 
     * @return a clone of the learned regressor
     */
    public Regressor getBaseRegressorClone()
    {
        return baseRegressor.clone();
    }
    
    
    /**
     * Returns an boolean array where the indices correspond to data points in 
     * the original training set. <tt>true</tt> indicates that the data point 
     * was apart of the final consensus set. <tt>false</tt> indicates that it 
     * was not. 
     * 
     * @return a boolean array indicating which points made it into the 
     * consensus set
     */
    public boolean[] getConsensusSet()
    {
        return Arrays.copyOf(consensusSet, consensusSet.length);
    }
    
    /**
     * Returns the model error, which is the average absolute difference between
     * the model and all points in the set of inliers. 
     * 
     * @return the error for the learned model. Returns 
     * {@link Double#POSITIVE_INFINITY} if the model has not been trained or 
     * failed to fit. 
     */
    public double getModelError()
    {
        return modelError;
    }
}
