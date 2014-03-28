package jsat.parameters;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.evaluation.Accuracy;
import jsat.classifiers.evaluation.ClassificationScore;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.regression.Regressor;
import jsat.regression.evaluation.MeanSquaredError;
import jsat.regression.evaluation.RegressionScore;
import jsat.utils.DoubleList;
import jsat.utils.FakeExecutor;

/**
 * GridSearch is a simple method for tuning the parameters of a classification 
 * or regression algorithm. It naively tries all possible pairs of parameter 
 * values given. For this reason, it works best when only a small number of 
 * parameters need to be turned. <br>
 * The model it takes must implement the {@link Parameterized} interface. By 
 * default, no parameters are selected for optimizations. This is because 
 * parameters value ranges are often algorithm specific. As such, the user must 
 * specify the parameters and the values to test using the <tt>addParameter</tt>
 * methods. 
 * 
 * @author Edward Raff
 * 
 * @see #addParameter(jsat.parameters.DoubleParameter, double[]) 
 * @see #addParameter(jsat.parameters.IntParameter, int[]) 
 */
public class GridSearch implements Classifier, Regressor
{
    private Classifier baseClassifier;
    private Classifier trainedClassifier;
    
    private ClassificationScore classificationTargetScore = new Accuracy();  
    private RegressionScore regressionTargetScore = new MeanSquaredError(true);
    
    private Regressor baseRegressor;
    private Regressor trainedRegressor;
    
    /**
     * The list of parameters we will later, Int and Double
     */
    private List<Parameter> searchParams;
    /**
     * The matching list of values we will test. This includes the integer 
     * parameters, which will have to be cast back and forth from doubles. 
     */
    private List<List<Double>> searchValues;
    /**
     * The number of CV folds
     */
    private int folds;

    /**
     * Creates a new GridSearch to tune the specified parameters of a regression
     * model. The parameters still need to be specified by calling 
     * {@link #addParameter(jsat.parameters.DoubleParameter, double[]) }
     * 
     * @param baseRegressor the regressor to tune the parameters of
     * @param folds the number of folds of cross-validation to perform to 
     * evaluate each combination of parameters
     * @throws FailedToFitException if the base regressor does not implement 
     * {@link Parameterized}
     */
    public GridSearch(Regressor baseRegressor, int folds)
    {
        if(!(baseRegressor instanceof Parameterized))
            throw new FailedToFitException("Given regressor does not support parameterized alterations");
        this.baseRegressor = baseRegressor;
        if(baseRegressor instanceof Classifier)
            this.baseClassifier = (Classifier) baseRegressor;
        searchParams = new ArrayList<Parameter>();
        searchValues = new ArrayList<List<Double>>();
        this.folds = folds;
    }
    
    /**
     * Creates a new GridSearch to tune the specified parameters of a 
     * classification model. The parameters still need to be specified by 
     * calling 
     * {@link #addParameter(jsat.parameters.DoubleParameter, double[]) }
     * 
     * @param baseClassifier the classifier to tune the parameters of
     * @param folds the number of folds of cross-validation to perform to 
     * evaluate each combination of parameters
     * @throws FailedToFitException if the base classifier does not implement 
     * {@link Parameterized}
     */
    public GridSearch(Classifier baseClassifier, int folds)
    {
        if(!(baseClassifier instanceof Parameterized))
            throw new FailedToFitException("Given classifier does not support parameterized alterations");
        this.baseClassifier = baseClassifier;
        if(baseClassifier instanceof Regressor)
            this.baseRegressor = (Regressor) baseClassifier;
        searchParams = new ArrayList<Parameter>();
        searchValues = new ArrayList<List<Double>>();
        this.folds = folds;
    }
    
    /**
     * Finds the parameter object with the given name, or throws an exception if
     * a parameter with the given name does not exist. 
     * @param name the name to search for
     * @return the parameter object in question
     * @throws IllegalArgumentException if the name is not found
     */
    private Parameter getParameterByName(String name) throws IllegalArgumentException
    {
        Parameter param;
        if (baseClassifier != null)
            param = ((Parameterized) baseClassifier).getParameter(name);
        else
            param = ((Parameterized) baseRegressor).getParameter(name);
        if (param == null)
            throw new IllegalArgumentException("Parameter " + name + " does not exist");
        return param;
    }
    
    /**
     * Adds a new double parameter to be altered for the model being tuned. 
     * 
     * @param param the model parameter
     * @param initialSearchValues the values to try for the specified parameter
     */
    public void addParameter(DoubleParameter param, double... initialSearchValues)
    {
        if(param == null)
            throw new IllegalArgumentException("null not allowed for parameter");
        searchParams.add(param);
        DoubleList dl = new DoubleList(initialSearchValues.length);
        for(double d : initialSearchValues)
            dl.add(d);
        searchValues.add(dl);
    }

    /**
     * Adds a new double parameter to be altered for the model being tuned.
     *
     * @param name the name of the parameter
     * @param initialSearchValues the values to try for the specified parameter
     */
    public void addParameter(String name, double... initialSearchValues)
    {
        Parameter param;
        param = getParameterByName(name);
        if (!(param instanceof DoubleParameter))
            throw new IllegalArgumentException("Parameter " + name + " is not for double values");

        addParameter((DoubleParameter) param, initialSearchValues);
    }
    
    /**
     * Adds a new int parameter to be altered for the model being tuned. 
     * 
     * @param param the model parameter
     * @param initialSearchValues the values to try for the specified parameter
     */
    public void addParameter(IntParameter param, int... initialSearchValues)
    {
        searchParams.add(param);
        DoubleList dl = new DoubleList(initialSearchValues.length);
        for(double d : initialSearchValues)
            dl.add(d);
        searchValues.add(dl);
    }
    
    /**
     * Adds a new integer parameter to be altered for the model being tuned.
     *
     * @param name the name of the parameter
     * @param initialSearchValues the values to try for the specified parameter
     */
    public void addParameter(String name, int... initialSearchValues)
    {
        Parameter param;
        param = getParameterByName(name);
        if (!(param instanceof IntParameter))
            throw new IllegalArgumentException("Parameter " + name + " is not for int values");

        addParameter((IntParameter) param, initialSearchValues);
    }

    /**
     * Returns the base classifier that was originally passed in when 
     * constructing this GridSearch. If this was not constructed with a 
     * classifier, this may return null. 
     * 
     * @return the original classifier object given
     */
    public Classifier getBaseClassifier()
    {
        return baseClassifier;
    }
    
    /**
     * Returns the resultant classifier trained on the whole data set after 
     * performing parameter tuning. 
     * 
     * @return the trained classifier after a call to 
     * {@link #train(jsat.regression.RegressionDataSet, 
     * java.util.concurrent.ExecutorService) }, or null if it has not been 
     * trained. 
     */
    public Classifier getTrainedClassifier()
    {
        return trainedClassifier;
    }
    
    /**
     * Returns the base regressor that was originally passed in when 
     * constructing this GridSearch. If this was not constructed with a 
     * regressor, this may return null. 
     * 
     * @return the original regressor object given
     */
    public Regressor getBaseRegressor()
    {
        return baseRegressor;
    }
    
    /**
     * Returns the resultant regressor trained on the whole data set after 
     * performing parameter tuning. 
     * 
     * @return the trained regressor after a call to 
     * {@link #train(jsat.regression.RegressionDataSet, 
     * java.util.concurrent.ExecutorService) }, or null if it has not been
     * trained. 
     */
    public Regressor getTrainedRegressor()
    {
        return trainedRegressor;
    }

    /**
     * Sets the score to attempt to optimize when performing grid search on a
     * classification problem. 
     * @param classifierTargetScore the score to optimize via grid search
     */
    public void setClassificationTargetScore(ClassificationScore classifierTargetScore)
    {
        this.classificationTargetScore = classifierTargetScore;
    }

    /**
     * Returns the classification score that is trying to be optimized via 
     * grid search
     * @return the classification score that is trying to be optimized via 
     * grid search
     */
    public ClassificationScore getClassificationTargetScore()
    {
        return classificationTargetScore;
    }

    /**
     * Sets the score to attempt to optimize when performing grid search on a
     * regression problem. 
     * @param regressionTargetScore 
     */
    public void setRegressionTargetScore(RegressionScore regressionTargetScore)
    {
        this.regressionTargetScore = regressionTargetScore;
    }

    /**
     * Returns the regression score that is trying to be optimized via 
     * grid search
     * @return the regression score that is trying to be optimized via 
     * grid search
     */
    public RegressionScore getRegressionTargetScore()
    {
        return regressionTargetScore;
    }
    
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(trainedClassifier == null)
            throw new UntrainedModelException("Model has not yet been trained");
        return trainedClassifier.classify(data);
    }

    @Override
    public void train(final RegressionDataSet dataSet, ExecutorService threadPool)
    {
        final PriorityQueue<RegressionModelEvaluation> bestModels =
                new PriorityQueue<RegressionModelEvaluation>(folds,
                                                                 new Comparator<RegressionModelEvaluation>()
        {
            @Override
            public int compare(RegressionModelEvaluation t, RegressionModelEvaluation t1)
            {
                double v0 = t.getScoreStats(regressionTargetScore).getMean();
                double v1 = t1.getScoreStats(regressionTargetScore).getMean();
                int order = regressionTargetScore.lowerIsBetter() ? 1 : -1;
                return order*Double.compare(v0, v1);
            }
        });

        
        /**
         * Use this to keep track of which parameter we are altering. Index 
         * correspondence to the parameter, and its value corresponds to which
         * value has been used. Increment and carry counts to iterate over all 
         * possible combinations. 
         */
        int[] setTo = new int[searchParams.size()];
        
        final CountDownLatch latch = getLatch();
        
        
        while(true)
        {
            setParameters(setTo);
           
            final Regressor toTrain = baseRegressor.clone();
            
            threadPool.submit(new Runnable() {

                @Override
                public void run()
                {
                    RegressionModelEvaluation rme = new RegressionModelEvaluation(toTrain, dataSet);
                    rme.addScorer(regressionTargetScore.clone());
                    rme.evaluateCrossValidation(folds);
                    synchronized(bestModels)
                    {
                        bestModels.add(rme);
                    }
                    
                    latch.countDown();
                }
            });
            
            
            if(incrementCombination(setTo))
                break;
        }
        
        
        try
        {
            latch.await();
            //Now we know the best classifier, we need to train one on the whole data set. 
            Regressor bestRegressor = bestModels.peek().getRegressor();//Just re-train it on the whole set
            
            if(threadPool instanceof FakeExecutor)
                bestRegressor.train(dataSet);
            else
                bestRegressor.train(dataSet, threadPool);
            trainedRegressor = bestRegressor;
            
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(GridSearch.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, new FakeExecutor());
    }
    
    @Override
    public void trainC(final ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        final PriorityQueue<ClassificationModelEvaluation> bestModels =
                new PriorityQueue<ClassificationModelEvaluation>(folds,
                                                                 new Comparator<ClassificationModelEvaluation>()
        {
            @Override
            public int compare(ClassificationModelEvaluation t, ClassificationModelEvaluation t1)
            {
                double v0 = t.getScoreStats(classificationTargetScore).getMean();
                double v1 = t1.getScoreStats(classificationTargetScore).getMean();
                int order = classificationTargetScore.lowerIsBetter() ? 1 : -1;
                return order*Double.compare(v0, v1);
            }
        });

        
        /**
         * Use this to keep track of which parameter we are altering. Index 
         * correspondence to the parameter, and its value corresponds to which
         * value has been used. Increment and carry counts to iterate over all 
         * possible combinations. 
         */
        int[] setTo = new int[searchParams.size()];
        
        final CountDownLatch latch = getLatch();
        
        
        while(true)
        {
            setParameters(setTo);
           
            final Classifier toTrain = baseClassifier.clone();
            
            threadPool.submit(new Runnable() {

                @Override
                public void run()
                {
                    ClassificationModelEvaluation cme = new ClassificationModelEvaluation(toTrain, dataSet);
                    cme.addScorer(classificationTargetScore.clone());
                    cme.evaluateCrossValidation(folds);
                    synchronized(bestModels)
                    {
                        bestModels.add(cme);
                    }
                    
                    latch.countDown();
                }
            });
            
            if(incrementCombination(setTo))
                break;
        }
        
        
        try
        {
            latch.await();
            //Now we know the best classifier, we need to train one on the whole data set. 
            Classifier bestClassifier = bestModels.peek().getClassifier();//Just re-train it on the whole set
            
            if(threadPool instanceof FakeExecutor)
                bestClassifier.trainC(dataSet);
            else
                bestClassifier.trainC(dataSet, threadPool);
            trainedClassifier = bestClassifier;
            
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(GridSearch.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return baseClassifier.supportsWeightedData();
    }

    @Override
    public GridSearch clone()
    {
        GridSearch clone = new GridSearch(baseClassifier.clone(), folds);
        clone.classificationTargetScore = this.classificationTargetScore.clone();
        clone.regressionTargetScore = this.regressionTargetScore.clone();
        if(searchParams != null)
            for(Parameter dp : searchParams)
            {
                Parameter p = ((Parameterized)clone.getBaseClassifier()).getParameter(dp.getName());
                clone.searchParams.add((DoubleParameter)p);
            }
        if(searchValues != null)
            for(List<Double> ld : searchValues)
            {
                List<Double> newVals = new DoubleList(ld);
                clone.searchValues.add(newVals);
            }
        
        return clone;
    }

    /**
     * Gets a new CountDownLatch with the appropriate count for the number of models that will be tested. 
     * @return a new CountDownLatch
     */
    private CountDownLatch getLatch()
    {
        int models = 1;
        for(List<Double> vals : searchValues)
            models *= vals.size();
        final CountDownLatch latch = new CountDownLatch(models);
        return latch;
    }

    @Override
    public double regress(DataPoint data)
    {
        return trainedRegressor.regress(data);
    }

    /**
     * This increments the array used to keep track of which combinations of 
     * parameter values have been used. 
     * @param setTo the array of length equal to {@link #searchParams} 
     * indicating which parameters have already been tried
     * @return a boolean indicating <tt>true</tt> if all combinations have been 
     * tried, or <tt>false</tt> if combinations remain to be attempted. 
     */
    private boolean incrementCombination(int[] setTo)
    {
        setTo[0]++;
        
        int carryPos = 0;
        
        while(carryPos < setTo.length-1 && setTo[carryPos] >= searchValues.get(carryPos).size())
        {
            setTo[carryPos] = 0;
            setTo[++carryPos]++;
        }
        
        return setTo[setTo.length-1] >= searchValues.get(setTo.length-1).size();
    }

    /**
     * Sets the parameters according to the given array
     * @param setTo the index corresponds to the parameters, and the value which
     * parameter value to use. 
     */
    private void setParameters(int[] setTo)
    {
        for(int i = 0; i < setTo.length; i++)
        {
            Parameter param = searchParams.get(i);
            if(param instanceof DoubleParameter)
                ((DoubleParameter)param).setValue(searchValues.get(i).get(setTo[i]));
            else if(param instanceof IntParameter)
                ((IntParameter)param).setValue(searchValues.get(i).get(setTo[i]).intValue());
        }
    }
}
