/*
 * Copyright (C) 2015 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.parameters;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.evaluation.Accuracy;
import jsat.classifiers.evaluation.ClassificationScore;
import jsat.distributions.Distribution;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.regression.Regressor;
import jsat.regression.evaluation.MeanSquaredError;
import jsat.regression.evaluation.RegressionScore;
import jsat.utils.FakeExecutor;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class RandomSearch implements Classifier, Regressor
{
    private Classifier baseClassifier;
    private Classifier trainedClassifier;

    private ClassificationScore classificationTargetScore = new Accuracy();  
    private RegressionScore regressionTargetScore = new MeanSquaredError(true);
    
    private Regressor baseRegressor;
    private Regressor trainedRegressor;
    
    private int trials = 25;
    
    /**
     * The list of parameters we will later, Int and Double
     */
    private List<Parameter> searchParams;
    /**
     * The matching list of distributions we will test. 
     */
    private List<Distribution> searchValues;
    
    
    /**
     * The number of CV folds
     */
    private int folds;
    
    /**
     * If true, parallelism will be obtained by training the models in parallel.
     * If false, parallelism is obtained from the model itself.
     */
    private boolean trainModelsInParallel = true;
    
    /**
     * If true, trains the final model on the parameters used
     */
    private boolean trainFinalModel = true;

    /**
     * If true, create the CV splits once and re-use them for all parameters
     */
    private boolean reuseSameCVFolds = true;
    
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
    public RandomSearch(Regressor baseRegressor, int folds)
    {
        if(!(baseRegressor instanceof Parameterized))
            throw new FailedToFitException("Given regressor does not support parameterized alterations");
        this.baseRegressor = baseRegressor;
        if(baseRegressor instanceof Classifier)
            this.baseClassifier = (Classifier) baseRegressor;
        searchParams = new ArrayList<Parameter>();
        searchValues = new ArrayList<Distribution>();
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
    public RandomSearch(Classifier baseClassifier, int folds)
    {
        if(!(baseClassifier instanceof Parameterized))
            throw new FailedToFitException("Given classifier does not support parameterized alterations");
        this.baseClassifier = baseClassifier;
        if(baseClassifier instanceof Regressor)
            this.baseRegressor = (Regressor) baseClassifier;
        searchParams = new ArrayList<Parameter>();
        searchValues = new ArrayList<Distribution>();
        this.folds = folds;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public RandomSearch(RandomSearch toCopy)
    {
        if(toCopy.baseClassifier != null)
        {
            this.baseClassifier = toCopy.baseClassifier.clone();
            if(this.baseClassifier instanceof Regressor)
                this.baseRegressor = (Regressor) this.baseClassifier;
        }
        else
        {
            this.baseRegressor = toCopy.baseRegressor.clone();
            if (this.baseRegressor instanceof Classifier)
                this.baseClassifier = (Classifier) this.baseRegressor;
        }
        if(toCopy.trainedClassifier != null)
            this.trainedClassifier = toCopy.trainedClassifier.clone();
        if(toCopy.trainedRegressor != null)
            this.trainedRegressor = toCopy.trainedRegressor.clone();
        this.trials = toCopy.trials;
        this.searchParams = new ArrayList<Parameter>();
        for(Parameter p : toCopy.searchParams)
            this.searchParams.add(getParameterByName(p.getName()));
        this.searchValues = new ArrayList<Distribution>(toCopy.searchValues.size());
        for (Distribution d : toCopy.searchValues)
            this.searchValues.add(d.clone());
        this.folds = toCopy.folds;
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
     * @return the trained regressor after a call to      {@link #train(jsat.regression.RegressionDataSet, 
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
     *
     * @param classifierTargetScore the score to optimize via grid search
     */
    public void setClassificationTargetScore(ClassificationScore classifierTargetScore)
    {
        this.classificationTargetScore = classifierTargetScore;
    }

    /**
     * Returns the classification score that is trying to be optimized via grid
     * search
     *
     * @return the classification score that is trying to be optimized via grid
     * search
     */
    public ClassificationScore getClassificationTargetScore()
    {
        return classificationTargetScore;
    }

   
    /**
     * Sets the score to attempt to optimize when performing grid search on a
     * regression problem.
     *
     * @param regressionTargetScore
     */
    public void setRegressionTargetScore(RegressionScore regressionTargetScore)
    {
        this.regressionTargetScore = regressionTargetScore;
    }

    /**
     * Returns the regression score that is trying to be optimized via grid
     * search
     *
     * @return the regression score that is trying to be optimized via grid
     * search
     */
    public RegressionScore getRegressionTargetScore()
    {
        return regressionTargetScore;
    }

    /**
     * Sets the number of trials or samples that will be taken. This value is the number of models that will be trained and evaluated for their performance
     * @param trials the number of models to build and evaluate
     */
    public void setTrials(int trials)
    {
        this.trials = trials;
    }

    /**
     * 
     * @return the number of models that will be built to evaluate
     */
    public int getTrials()
    {
        return trials;
    }
    
    /**
     * Finds the parameter object with the given name, or throws an exception if
     * a parameter with the given name does not exist.
     *
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
     * @param initialSearchValues the distribution to sample from for this parameter
     */
    public void addParameter(DoubleParameter param, Distribution dist)
    {
        if (param == null)
            throw new IllegalArgumentException("null not allowed for parameter");
        searchParams.add(param);
        searchValues.add(dist.clone());
    }
    
    /**
     * Adds a new double parameter to be altered for the model being tuned.
     *
     * @param param the model parameter
     * @param initialSearchValues the distribution to sample from for this parameter
     */
    public void addParameter(IntParameter param, Distribution dist)
    {
        if (param == null)
            throw new IllegalArgumentException("null not allowed for parameter");
        searchParams.add(param);
        searchValues.add(dist.clone());
    }

    /**
     * Adds a new parameter to be altered for the model being tuned. 
     *
     * @param name the name of the parameter
     * @param initialSearchValues the values to try for the specified parameter
     */
    public void addParameter(String name, Distribution dist)
    {
        Parameter param = getParameterByName(name);

        if(param instanceof DoubleParameter)
            addParameter((DoubleParameter) param, dist);
        else if(param instanceof IntParameter)
            addParameter((IntParameter) param, dist);
        else
            throw new IllegalArgumentException("Parameter " + name + " is not for double or int values");
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(trainedClassifier == null)
            throw new UntrainedModelException("Model has not yet been trained");
        return trainedClassifier.classify(data);
    }


    @Override
    public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool)
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
         * Each model is set to have different combination of parameters. We 
         * then train each model to determine the best one. 
         */
        final List<Classifier> paramsToEval = new ArrayList<Classifier>();
        
        Random rand = new XORWOW();
        for(int trial = 0; trial < trials; trial++)
        {
            for(int i = 0; i < searchParams.size(); i++)
            {
                double sampledValue = searchValues.get(i).invCdf(rand.nextDouble());
                
                Parameter param = searchParams.get(i);
                if(param instanceof DoubleParameter)
                    ((DoubleParameter)param).setValue(sampledValue);
                else if(param instanceof IntParameter)
                    ((IntParameter)param).setValue((int) Math.round(sampledValue));
            }
            
            paramsToEval.add(baseClassifier.clone());
        }
        
        /*
         * This is the Executor used for training the models in parallel. If we 
         * are not supposed to do that, it will be an executor that executes 
         * them sequentually. 
         */
        final ExecutorService modelService;
        if(trainModelsInParallel && threadPool != null)
            modelService = threadPool;
        else
            modelService = new FakeExecutor();
        
        //if we are doing our CV splits ahead of time, get them done now
        final List<ClassificationDataSet> preFolded;

        /**
         * Pre-combine our training combinations so that any caching can be
         * re-used
         */
        final List<ClassificationDataSet> trainCombinations;

        if (reuseSameCVFolds)
        {
            preFolded = dataSet.cvSet(folds);
            trainCombinations = new ArrayList<ClassificationDataSet>(preFolded.size());
            for (int i = 0; i < preFolded.size(); i++)
                trainCombinations.add(ClassificationDataSet.comineAllBut(preFolded, i));
        }
        else
        {
            preFolded = null;
            trainCombinations = null;
        }
        final CountDownLatch latch = new CountDownLatch(paramsToEval.size());
        for (final Classifier c : paramsToEval)
            modelService.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    ClassificationModelEvaluation cme = trainModelsInParallel
                            ? new ClassificationModelEvaluation(c, dataSet)
                            : new ClassificationModelEvaluation(c, dataSet, threadPool);
                    cme.addScorer(classificationTargetScore.clone());
                    
                    if (reuseSameCVFolds)
                        cme.evaluateCrossValidation(preFolded, trainCombinations);
                    else
                        cme.evaluateCrossValidation(folds);

                    synchronized (bestModels)
                    {
                        bestModels.add(cme);
                    }
                    
                    latch.countDown();
                }
            });
        
        try
        {
            latch.await();
            
            Classifier bestClassifier = bestModels.peek().getClassifier();//Just re-train it on the whole set
            if (trainFinalModel)
            {

                if (threadPool instanceof FakeExecutor)
                    bestClassifier.trainC(dataSet);
                else
                    bestClassifier.trainC(dataSet, threadPool);

            }
            trainedClassifier = bestClassifier;
        }
        catch (InterruptedException ex)
        {
            throw new FailedToFitException(ex);
        }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        if(baseClassifier != null)
            return baseClassifier.supportsWeightedData();
        else
            return baseRegressor.supportsWeightedData();
    }

    @Override
    public double regress(DataPoint data)
    {
        if(trainedRegressor == null)
            throw new UntrainedModelException();
        return trainedRegressor.regress(data);
    }

    @Override
    public void train(final RegressionDataSet dataSet, final ExecutorService threadPool)
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
         * Each model is set to have different combination of parameters. We 
         * then train each model to determine the best one. 
         */
        final List<Regressor> paramsToEval = new ArrayList<Regressor>();
        
        Random rand = new XORWOW();
        for(int trial = 0; trial < trials; trial++)
        {
            for(int i = 0; i < searchParams.size(); i++)
            {
                double sampledValue = searchValues.get(i).invCdf(rand.nextDouble());
                
                Parameter param = searchParams.get(i);
                if(param instanceof DoubleParameter)
                    ((DoubleParameter)param).setValue(sampledValue);
                else if(param instanceof IntParameter)
                    ((IntParameter)param).setValue((int) Math.round(sampledValue));
            }
            
            paramsToEval.add(baseRegressor.clone());
        }
        
        /*
         * This is the Executor used for training the models in parallel. If we 
         * are not supposed to do that, it will be an executor that executes 
         * them sequentually. 
         */
        final ExecutorService modelService;
        if(trainModelsInParallel && threadPool != null)
            modelService = threadPool;
        else
            modelService = new FakeExecutor();
        
        //if we are doing our CV splits ahead of time, get them done now
        final List<RegressionDataSet> preFolded;

        /**
         * Pre-combine our training combinations so that any caching can be
         * re-used
         */
        final List<RegressionDataSet> trainCombinations;

        if (reuseSameCVFolds)
        {
            preFolded = dataSet.cvSet(folds);
            trainCombinations = new ArrayList<RegressionDataSet>(preFolded.size());
            for (int i = 0; i < preFolded.size(); i++)
                trainCombinations.add(RegressionDataSet.comineAllBut(preFolded, i));
        }
        else
        {
            preFolded = null;
            trainCombinations = null;
        }
        final CountDownLatch latch = new CountDownLatch(paramsToEval.size());
        for (final Regressor r : paramsToEval)
            modelService.submit(new Runnable()
            {

                @Override
                public void run()
                {
                    RegressionModelEvaluation cme = trainModelsInParallel
                            ? new RegressionModelEvaluation(r, dataSet)
                            : new RegressionModelEvaluation(r, dataSet, threadPool);
                    cme.addScorer(regressionTargetScore.clone());
                    
                    if (reuseSameCVFolds)
                        cme.evaluateCrossValidation(preFolded, trainCombinations);
                    else
                        cme.evaluateCrossValidation(folds);

                    synchronized (bestModels)
                    {
                        bestModels.add(cme);
                    }
                    
                    latch.countDown();
                }
            });
        
        try
        {
            latch.await();
            
            Regressor bestRegressor = bestModels.peek().getRegressor();//Just re-train it on the whole set
            if (trainFinalModel)
            {

                if (threadPool instanceof FakeExecutor)
                    bestRegressor.train(dataSet);
                else
                    bestRegressor.train(dataSet, threadPool);

            }
            trainedRegressor = bestRegressor;
        }
        catch (InterruptedException ex)
        {
            throw new FailedToFitException(ex);
        }
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, null);
    }

    @Override
    public RandomSearch clone()
    {
        return new RandomSearch(this);
    }
    
    
    
    
}
