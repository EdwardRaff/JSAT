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
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.Classifier;
import jsat.distributions.Distribution;
import jsat.exceptions.FailedToFitException;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.regression.Regressor;
import jsat.utils.FakeExecutor;
import jsat.utils.random.XORWOW;

/**
 * Random Search is a simple method for tuning the parameters of a
 * classification or regression algorithm. Each parameter is given a
 * distribution that represents the values of interest, and trials are done by
 * randomly sampling each parameter from their respective distributions.
 * Compared to {@link GridSearch} this method does better when lots of values
 * are to be tested or when 2 or more parameters are to be evaluated. <br>
 * The model it takes must implement the {@link Parameterized} interface. By
 * default, no parameters are selected for optimizations. This is because
 * parameters value ranges are often algorithm specific. As such, the user must 
 * specify the parameters and the values to test using the <tt>addParameter</tt>
 * methods. 
 * 
 * See : Bergstra, J., & Bengio, Y. (2012). <i>Random Search for Hyper-Parameter Optimization</i>. Journal ofMachine Learning Research, 13, 281–305.
 * @author Edward Raff
 */
public class RandomSearch extends ModelSearch
{
    private int trials = 25;

    /**
     * The matching list of distributions we will test.
     */
    private final List<Distribution> searchValues;

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
    public RandomSearch(final Regressor baseRegressor, final int folds)
    {
        super(baseRegressor, folds);
        searchValues = new ArrayList<Distribution>();
    }

    /**
     * Creates a new GridSearch to tune the specified parameters of a
     * classification model. The parameters still need to be specified by
     * calling {@link #addParameter(jsat.parameters.DoubleParameter, double[]) }
     * 
     * @param baseClassifier the classifier to tune the parameters of
     * @param folds the number of folds of cross-validation to perform to 
     * evaluate each combination of parameters
     * @throws FailedToFitException if the base classifier does not implement 
     * {@link Parameterized}
     */
    public RandomSearch(final Classifier baseClassifier, final int folds)
    {
        super(baseClassifier, folds);
        searchValues = new ArrayList<Distribution>();
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public RandomSearch(final RandomSearch toCopy)
    {
        super(toCopy);
        this.trials = toCopy.trials;
        this.searchValues = new ArrayList<Distribution>(toCopy.searchValues.size());
        for (final Distribution d : toCopy.searchValues) {
          this.searchValues.add(d.clone());
        }
    }
    
    /**
     * This method will automatically populate the search space with parameters
     * based on which Parameter objects return non-null distributions.<br>
     * <br>
     * Note, using this method with Cross Validation has the potential for
     * over-estimating the accuracy of results if the data set is actually used
     * to for parameter guessing.
     *
     * @param data the data set to get parameter estimates from
     * @return the number of parameters added
     */
    public int autoAddParameters(final DataSet data)
    {
        Parameterized obj;
        if (baseClassifier != null) {
          obj = (Parameterized) baseClassifier;
        } else {
          obj = (Parameterized) baseRegressor;
        }
        int totalParms = 0;
        for (final Parameter param : obj.getParameters())
        {
            Distribution dist;
            if (param instanceof DoubleParameter)
            {
                dist = ((DoubleParameter) param).getGuess(data);
                if (dist != null)
                {
                    addParameter((DoubleParameter) param, dist);
                    totalParms++;
                }
            }
            else if (param instanceof IntParameter)
            {
                dist = ((IntParameter) param).getGuess(data);
                if (dist != null)
                {
                    addParameter((IntParameter) param, dist);
                    totalParms++;
                }
            }
        }
        
        return totalParms;
    }
    
    /**
     * Sets the number of trials or samples that will be taken. This value is the number of models that will be trained and evaluated for their performance
     * @param trials the number of models to build and evaluate
     */
    public void setTrials(final int trials)
    {
        if(trials < 1) {
          throw new IllegalArgumentException("number of trials must be positive, not " + trials);
        }
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
     * Adds a new double parameter to be altered for the model being tuned.
     *
     * @param param the model parameter
     * @param initialSearchValues the distribution to sample from for this parameter
     */
    public void addParameter(final DoubleParameter param, final Distribution dist)
    {
        if (param == null) {
          throw new IllegalArgumentException("null not allowed for parameter");
        }
        searchParams.add(param);
        searchValues.add(dist.clone());
    }
    
    /**
     * Adds a new double parameter to be altered for the model being tuned.
     *
     * @param param the model parameter
     * @param initialSearchValues the distribution to sample from for this parameter
     */
    public void addParameter(final IntParameter param, final Distribution dist)
    {
        if (param == null) {
          throw new IllegalArgumentException("null not allowed for parameter");
        }
        searchParams.add(param);
        searchValues.add(dist.clone());
    }

    /**
     * Adds a new parameter to be altered for the model being tuned. 
     *
     * @param name the name of the parameter
     * @param initialSearchValues the values to try for the specified parameter
     */
    public void addParameter(final String name, final Distribution dist)
    {
        final Parameter param = getParameterByName(name);

        if(param instanceof DoubleParameter) {
          addParameter((DoubleParameter) param, dist);
        } else if(param instanceof IntParameter) {
          addParameter((IntParameter) param, dist);
        } else {
          throw new IllegalArgumentException("Parameter " + name + " is not for double or int values");
        }
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool)
    {
        final PriorityQueue<ClassificationModelEvaluation> bestModels =
                new PriorityQueue<ClassificationModelEvaluation>(folds,
                                                                 new Comparator<ClassificationModelEvaluation>()
        {
            @Override
            public int compare(final ClassificationModelEvaluation t, final ClassificationModelEvaluation t1)
            {
                final double v0 = t.getScoreStats(classificationTargetScore).getMean();
                final double v1 = t1.getScoreStats(classificationTargetScore).getMean();
                final int order = classificationTargetScore.lowerIsBetter() ? 1 : -1;
                return order*Double.compare(v0, v1);
            }
        });
        
        /**
         * Each model is set to have different combination of parameters. We 
         * then train each model to determine the best one. 
         */
        final List<Classifier> paramsToEval = new ArrayList<Classifier>();
        
        final Random rand = new XORWOW();
        for(int trial = 0; trial < trials; trial++)
        {
            for(int i = 0; i < searchParams.size(); i++)
            {
                final double sampledValue = searchValues.get(i).invCdf(rand.nextDouble());
                
                final Parameter param = searchParams.get(i);
                if(param instanceof DoubleParameter) {
                  ((DoubleParameter)param).setValue(sampledValue);
                } else if(param instanceof IntParameter) {
                  ((IntParameter)param).setValue((int) Math.round(sampledValue));
                }
            }
            
            paramsToEval.add(baseClassifier.clone());
        }
        
        /*
         * This is the Executor used for training the models in parallel. If we 
         * are not supposed to do that, it will be an executor that executes 
         * them sequentually. 
         */
        final ExecutorService modelService;
        if(trainModelsInParallel && threadPool != null) {
          modelService = threadPool;
        } else {
          modelService = new FakeExecutor();
        }
        
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
            for (int i = 0; i < preFolded.size(); i++) {
              trainCombinations.add(ClassificationDataSet.comineAllBut(preFolded, i));
            }
        }
        else
        {
            preFolded = null;
            trainCombinations = null;
        }
        final CountDownLatch latch = new CountDownLatch(paramsToEval.size());
        for (final Classifier c : paramsToEval) {
          modelService.submit(new Runnable() {
            @Override
            public void run() {
              final ClassificationModelEvaluation cme = trainModelsInParallel
                      ? new ClassificationModelEvaluation(c, dataSet)
                      : new ClassificationModelEvaluation(c, dataSet, threadPool);
              cme.addScorer(classificationTargetScore.clone());
              if (reuseSameCVFolds) {
                cme.evaluateCrossValidation(preFolded, trainCombinations);
              } else {
                cme.evaluateCrossValidation(folds);
              }
              synchronized (bestModels)
              {
                bestModels.add(cme);
              }
              latch.countDown();
            }
          });
        }
        
        try
        {
            latch.await();
            
            final Classifier bestClassifier = bestModels.peek().getClassifier();//Just re-train it on the whole set
            if (trainFinalModel)
            {

                if (threadPool instanceof FakeExecutor) {
                  bestClassifier.trainC(dataSet);
                } else {
                  bestClassifier.trainC(dataSet, threadPool);
                }

            }
            trainedClassifier = bestClassifier;
        }
        catch (final InterruptedException ex)
        {
            throw new FailedToFitException(ex);
        }
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    @Override
    public void train(final RegressionDataSet dataSet, final ExecutorService threadPool)
    {
        final PriorityQueue<RegressionModelEvaluation> bestModels =
                new PriorityQueue<RegressionModelEvaluation>(folds,
                                                                 new Comparator<RegressionModelEvaluation>()
        {
            @Override
            public int compare(final RegressionModelEvaluation t, final RegressionModelEvaluation t1)
            {
                final double v0 = t.getScoreStats(regressionTargetScore).getMean();
                final double v1 = t1.getScoreStats(regressionTargetScore).getMean();
                final int order = regressionTargetScore.lowerIsBetter() ? 1 : -1;
                return order*Double.compare(v0, v1);
            }
        });
        
        /**
         * Each model is set to have different combination of parameters. We 
         * then train each model to determine the best one. 
         */
        final List<Regressor> paramsToEval = new ArrayList<Regressor>();
        
        final Random rand = new XORWOW();
        for(int trial = 0; trial < trials; trial++)
        {
            for(int i = 0; i < searchParams.size(); i++)
            {
                final double sampledValue = searchValues.get(i).invCdf(rand.nextDouble());
                
                final Parameter param = searchParams.get(i);
                if(param instanceof DoubleParameter) {
                  ((DoubleParameter)param).setValue(sampledValue);
                } else if(param instanceof IntParameter) {
                  ((IntParameter)param).setValue((int) Math.round(sampledValue));
                }
            }
            
            paramsToEval.add(baseRegressor.clone());
        }
        
        /*
         * This is the Executor used for training the models in parallel. If we 
         * are not supposed to do that, it will be an executor that executes 
         * them sequentually. 
         */
        final ExecutorService modelService;
        if(trainModelsInParallel && threadPool != null) {
          modelService = threadPool;
        } else {
          modelService = new FakeExecutor();
        }
        
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
            for (int i = 0; i < preFolded.size(); i++) {
              trainCombinations.add(RegressionDataSet.comineAllBut(preFolded, i));
            }
        }
        else
        {
            preFolded = null;
            trainCombinations = null;
        }
        final CountDownLatch latch = new CountDownLatch(paramsToEval.size());
        for (final Regressor r : paramsToEval) {
          modelService.submit(new Runnable() {
            @Override
            public void run() {
              final RegressionModelEvaluation cme = trainModelsInParallel
                      ? new RegressionModelEvaluation(r, dataSet)
                      : new RegressionModelEvaluation(r, dataSet, threadPool);
              cme.addScorer(regressionTargetScore.clone());
              if (reuseSameCVFolds) {
                cme.evaluateCrossValidation(preFolded, trainCombinations);
              } else {
                cme.evaluateCrossValidation(folds);
              }
              synchronized (bestModels)
              {
                bestModels.add(cme);
              }
              latch.countDown();
            }
          });
        }
        
        try
        {
            latch.await();
            
            final Regressor bestRegressor = bestModels.peek().getRegressor();//Just re-train it on the whole set
            if (trainFinalModel)
            {

                if (threadPool instanceof FakeExecutor) {
                  bestRegressor.train(dataSet);
                } else {
                  bestRegressor.train(dataSet, threadPool);
                }

            }
            trainedRegressor = bestRegressor;
        }
        catch (final InterruptedException ex)
        {
            throw new FailedToFitException(ex);
        }
    }

    @Override
    public void train(final RegressionDataSet dataSet)
    {
        train(dataSet, null);
    }

    @Override
    public RandomSearch clone()
    {
        return new RandomSearch(this);
    }
    
}
