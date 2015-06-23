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
import java.util.List;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.evaluation.Accuracy;
import jsat.classifiers.evaluation.ClassificationScore;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.regression.Regressor;
import jsat.regression.evaluation.MeanSquaredError;
import jsat.regression.evaluation.RegressionScore;

/**
 * This abstract class provides boilerplate for algorithms that search a model's
 * parameter space to find the parameters that provide the best overall
 * performance.
 *
 * @author Edward Raff
 */
abstract public class ModelSearch implements Classifier, Regressor
{
    protected Classifier baseClassifier;
    protected Classifier trainedClassifier;

    protected ClassificationScore classificationTargetScore = new Accuracy();
    protected RegressionScore regressionTargetScore = new MeanSquaredError(true);

    protected Regressor baseRegressor;
    protected Regressor trainedRegressor;

    /**
     * The list of parameters we will search for, currently only Int and Double
     * params should be used
     */
    protected List<Parameter> searchParams;

    /**
     * The number of CV folds
     */
    protected int folds;

    /**
     * If true, parallelism will be obtained by training the models in parallel.
     * If false, parallelism is obtained from the model itself.
     */
    protected boolean trainModelsInParallel = true;

    /**
     * If true, trains the final model on the parameters used
     */
    protected boolean trainFinalModel = true;

    /**
     * If true, create the CV splits once and re-use them for all parameters
     */
    protected boolean reuseSameCVFolds = true;

    public ModelSearch(Regressor baseRegressor, int folds)
    {
        if (!(baseRegressor instanceof Parameterized))
            throw new FailedToFitException("Given regressor does not support parameterized alterations");
        this.baseRegressor = baseRegressor;
        if (baseRegressor instanceof Classifier)
            this.baseClassifier = (Classifier) baseRegressor;
        searchParams = new ArrayList<Parameter>();
        this.folds = folds;
    }

    public ModelSearch(Classifier baseClassifier, int folds)
    {
        if (!(baseClassifier instanceof Parameterized))
            throw new FailedToFitException("Given classifier does not support parameterized alterations");
        this.baseClassifier = baseClassifier;
        if (baseClassifier instanceof Regressor)
            this.baseRegressor = (Regressor) baseClassifier;
        searchParams = new ArrayList<Parameter>();
        this.folds = folds;
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public ModelSearch(ModelSearch toCopy)
    {
        if (toCopy.baseClassifier != null)
        {
            this.baseClassifier = toCopy.baseClassifier.clone();
            if (this.baseClassifier instanceof Regressor)
                this.baseRegressor = (Regressor) this.baseClassifier;
        }
        else
        {
            this.baseRegressor = toCopy.baseRegressor.clone();
            if (this.baseRegressor instanceof Classifier)
                this.baseClassifier = (Classifier) this.baseRegressor;
        }
        if (toCopy.trainedClassifier != null)
            this.trainedClassifier = toCopy.trainedClassifier.clone();
        if (toCopy.trainedRegressor != null)
            this.trainedRegressor = toCopy.trainedRegressor.clone();
        this.searchParams = new ArrayList<Parameter>();
        for (Parameter p : toCopy.searchParams)
            this.searchParams.add(getParameterByName(p.getName()));
        this.folds = toCopy.folds;
    }

    /**
     * When set to {@code true} (the default) parallelism is obtained by
     * training as many models in parallel as possible. If {@code false},
     * parallelsm will be obtained by training the model using the {@link Classifier#trainC(jsat.classifiers.ClassificationDataSet, java.util.concurrent.ExecutorService)
     * } and {@link Regressor#train(jsat.regression.RegressionDataSet, java.util.concurrent.ExecutorService)
     * } methods.<br>
     * <br>
     * When a model supports {@link #setUseWarmStarts(boolean) warms starts},
     * parallelism obtained by training the models in parallel is intrinsically
     * reduced, as a model can not be warms started until another model has
     * finished. In the case that one of the parameters is annotated as a
     * {@link Parameter.WarmParameter warm paramter} , that parameter will be
     * the one rained sequential, and for every other parameter combination
     * models will be trained in parallel. If there is no warm parameter, the
     * first parameter added will be used for warm training. If there is only
     * one parameter and warm training is occurring, no parallelism will be
     * obtained.
     *
     * @param trainInParallel {@code true} to get parallelism from training many
     * models at the same time, {@code false} to get parallelism from getting
     * the model's implicit parallelism.
     */
    public void setTrainModelsInParallel(boolean trainInParallel)
    {
        this.trainModelsInParallel = trainInParallel;
    }

    /**
     *
     * @return {@code true} if parallelism is obtained from training many models
     * at the same time, {@code false} if parallelism is obtained from using the
     * model's implicit parallelism.
     */
    public boolean isTrainModelsInParallel()
    {
        return trainModelsInParallel;
    }

    /**
     * If {@code true} (the default) the model that was found to be best is
     * trained on the whole data set at the end. If {@code false}, the final
     * model will not be trained. This means that this Object will not be usable
     * for predictoin. This should only be set if you know you will not be using
     * this model but only want to get the information about which parameter
     * combination is best.
     *
     * @param trainFinalModel {@code true} to train the final model after grid
     * search, {@code false} to not do that.
     */
    public void setTrainFinalModel(boolean trainFinalModel)
    {
        this.trainFinalModel = trainFinalModel;
    }

    /**
     *
     * @return {@code true} to train the final model after grid search,
     * {@code false} to not do that.
     */
    public boolean isTrainFinalModel()
    {
        return trainFinalModel;
    }

    /**
     * Sets whether or not one set of CV folds is created and re used for every
     * parameter combination (the default), or if a difference set of CV folds
     * will be used for every parameter combination.
     *
     * @param reuseSameSplit {@code true} if the same split is re-used for every
     * combination, {@code false} if a new CV set is used for every parameter
     * combination.
     */
    public void setReuseSameCVFolds(boolean reuseSameSplit)
    {
        this.reuseSameCVFolds = reuseSameSplit;
    }

    /**
     *
     * @return {@code true} if the same split is re-used for every combination,
     * {@code false} if a new CV set is used for every parameter combination.
     */
    public boolean isReuseSameCVFolds()
    {
        return reuseSameCVFolds;
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
     * @return the trained classifier after a call to      {@link #train(jsat.regression.RegressionDataSet, 
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
     * Finds the parameter object with the given name, or throws an exception if
     * a parameter with the given name does not exist.
     *
     * @param name the name to search for
     * @return the parameter object in question
     * @throws IllegalArgumentException if the name is not found
     */
    protected Parameter getParameterByName(String name) throws IllegalArgumentException
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

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if (trainedClassifier == null)
            throw new UntrainedModelException("Model has not yet been trained");
        return trainedClassifier.classify(data);
    }

    @Override
    public double regress(DataPoint data)
    {
        if (trainedRegressor == null)
            throw new UntrainedModelException("Model has not yet been trained");
        return trainedRegressor.regress(data);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return baseClassifier != null ? baseClassifier.supportsWeightedData() : baseRegressor.supportsWeightedData();
    }

    @Override
    abstract public ModelSearch clone();

}
