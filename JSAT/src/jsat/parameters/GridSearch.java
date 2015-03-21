package jsat.parameters;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.*;
import jsat.classifiers.evaluation.Accuracy;
import jsat.classifiers.evaluation.ClassificationScore;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.UntrainedModelException;
import jsat.regression.*;
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
     * Use warm starts when possible
     */
    private boolean useWarmStarts = true;
    
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
     * Sets whether or not warm starts are used, but only if the model in use
     * supports warm starts. This is set to {@code true} by default. 
     * 
     * @param useWarmStarts {@code true} if warm starts should be used when 
     * possible, {@code false} otherwise. 
     */
    public void setUseWarmStarts(boolean useWarmStarts)
    {
        this.useWarmStarts = useWarmStarts;
    }

    /**
     *
     * @return {@code true} if warm starts will be used when possible.
     * {@code false} if they will not.
     */
    public boolean isUseWarmStarts()
    {
        return useWarmStarts;
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
     * @return  {@code true} to train the final model after grid
     * search, {@code false} to not do that.
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
     * @return {@code true} if the same split is re-used for every
     * combination, {@code false} if a new CV set is used for every parameter
     * combination.
     */
    public boolean isReuseSameCVFolds()
    {
        return reuseSameCVFolds;
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
        Arrays.sort(dl.getBackingArray());//convience, only really needed if param is warm
        if (param.isWarmParameter() && !param.preferredLowToHigh())
            Collections.reverse(dl);//put it in the prefered order
        if (param.isWarmParameter())//put it at the front!
            searchValues.add(0, dl);
        else
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
        Arrays.sort(dl.getBackingArray());//convience, only really needed if param is warm
        if (param.isWarmParameter() && !param.preferredLowToHigh())
            Collections.reverse(dl);//put it in the prefered order
        if (param.isWarmParameter())//put it at the front!
            searchValues.add(0, dl);
        else
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
         * Use this to keep track of which parameter we are altering. Index 
         * correspondence to the parameter, and its value corresponds to which
         * value has been used. Increment and carry counts to iterate over all 
         * possible combinations. 
         */
        int[] setTo = new int[searchParams.size()];
        /**
         * Each model is set to have different combination of parameters. We 
         * then train each model to determine the best one. 
         */
        final List<Regressor> paramsToEval = new ArrayList<Regressor>();
        
        while(true)
        {
            setParameters(setTo);
           
            paramsToEval.add(baseRegressor.clone());
            
            if(incrementCombination(setTo))
                break;
        }
        /*
         * This is the Executor used for training the models in parallel. If we 
         * are not supposed to do that, it will be an executor that executes 
         * them sequentually. 
         */
        final ExecutorService modelService;
        if(trainModelsInParallel)
            modelService = threadPool;
        else
            modelService = new FakeExecutor();
        
        final CountDownLatch latch;//used for stopping in both cases
        
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
        
        boolean considerWarm = useWarmStarts && baseRegressor instanceof WarmRegressor;
        /**
         * make sure we don't do a warm start if its only supported when trained
         * on the same data but we aren't reuse-ing the same CV splits So we get
         * the truth table
         * 
         * a | b | (a&&b)||¬a
         * T | T | T
         * T | F | F
         * F | T | T
         * F | F | T
         * 
         * where a = warmFromSameDataOnly and b  = reuseSameSplit
         * So we can instead use 
         * ¬ a || b
         */

        if (considerWarm && (!((WarmRegressor) baseRegressor).warmFromSameDataOnly() || reuseSameCVFolds))
        {
            /* we want all of the first parameter (which is the warm paramter, 
             * taken care of for us) values done in a group. So We can get this
             * by just dividing up the larger list into sub lists, each sub list
             * is adjacent in the original and is the number of parameter values
             * we wanted to try
             */
            
            int stepSize = searchValues.get(0).size();
            int totalJobs = paramsToEval.size()/stepSize;
            latch = new CountDownLatch(totalJobs);
            for(int startPos = 0; startPos < paramsToEval.size(); startPos += stepSize)
            {
                final List<Regressor> subSet = paramsToEval.subList(startPos, startPos+stepSize);
                modelService.submit(new Runnable()
                {

                    @Override
                    public void run()
                    {
                        Regressor[] prevModels = null;
                        for(Regressor r : subSet)
                        {
                            RegressionModelEvaluation rme = trainModelsInParallel ?
                                    new RegressionModelEvaluation(r, dataSet) 
                                    : new RegressionModelEvaluation(r, dataSet, threadPool);
                            rme.setKeepModels(true);//we need these to do warm starts!
                            rme.setWarmModels(prevModels);
                            rme.addScorer(regressionTargetScore.clone());
                            if(reuseSameCVFolds)
                                rme.evaluateCrossValidation(preFolded, trainCombinations);
                            else
                                rme.evaluateCrossValidation(folds);
                            prevModels = rme.getKeptModels();
                            synchronized(bestModels)
                            {
                                bestModels.add(rme);
                            }
                        }
                        latch.countDown();
                    }
                });
            }
        }
        else//regular CV, train a new model from scratch at every step
        {
            latch = new CountDownLatch(paramsToEval.size());

            for (final Regressor toTrain : paramsToEval)
            {

                modelService.submit(new Runnable()
                {

                    @Override
                    public void run()
                    {
                        RegressionModelEvaluation rme = trainModelsInParallel ?
                                    new RegressionModelEvaluation(toTrain, dataSet) 
                                    : new RegressionModelEvaluation(toTrain, dataSet, threadPool);
                        rme.addScorer(regressionTargetScore.clone());
                        if (reuseSameCVFolds)
                            rme.evaluateCrossValidation(preFolded, trainCombinations);
                        else
                            rme.evaluateCrossValidation(folds);
                        synchronized (bestModels)
                        {
                            bestModels.add(rme);
                        }

                        latch.countDown();
                    }
                });
            }
        }
        
        
        try
        {
            latch.await();
            //Now we know the best classifier, we need to train one on the whole data set. 
            Regressor bestRegressor = bestModels.peek().getRegressor();//Just re-train it on the whole set
            if(trainFinalModel)
            {
                //try and warm start the final model if we can
                if(useWarmStarts && bestRegressor instanceof WarmRegressor && 
                        !((WarmRegressor)bestRegressor).warmFromSameDataOnly())//last line here needed to make sure we can do this warm train
                {
                    WarmRegressor wr = (WarmRegressor) bestRegressor;
                    if(threadPool instanceof FakeExecutor)
                        wr.train(dataSet, wr.clone());
                    else
                        wr.train(dataSet, wr.clone(), threadPool);
                }
                else
                {
                    if (threadPool instanceof FakeExecutor)
                        bestRegressor.train(dataSet);
                    else
                        bestRegressor.train(dataSet, threadPool);
                }
            }
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
         * Use this to keep track of which parameter we are altering. Index 
         * correspondence to the parameter, and its value corresponds to which
         * value has been used. Increment and carry counts to iterate over all 
         * possible combinations. 
         */
        int[] setTo = new int[searchParams.size()];
        
        /**
         * Each model is set to have different combination of parameters. We 
         * then train each model to determine the best one. 
         */
        final List<Classifier> paramsToEval = new ArrayList<Classifier>();
        
        while(true)
        {
            setParameters(setTo);
           
            paramsToEval.add(baseClassifier.clone());
            
            if(incrementCombination(setTo))
                break;
        }
        /*
         * This is the Executor used for training the models in parallel. If we 
         * are not supposed to do that, it will be an executor that executes 
         * them sequentually. 
         */
        final ExecutorService modelService;
        if(trainModelsInParallel)
            modelService = threadPool;
        else
            modelService = new FakeExecutor();
        
        final CountDownLatch latch;//used for stopping in both cases
        
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
        
        boolean considerWarm = useWarmStarts && baseClassifier instanceof WarmClassifier;
        
        /**
         * make sure we don't do a warm start if its only supported when trained
         * on the same data but we aren't reuse-ing the same CV splits So we get
         * the truth table
         * 
         * a | b | (a&&b)||¬a
         * T | T | T
         * T | F | F
         * F | T | T
         * F | F | T
         * 
         * where a = warmFromSameDataOnly and b  = reuseSameSplit
         * So we can instead use 
         * ¬ a || b
         */

        if (considerWarm && (!((WarmClassifier) baseClassifier).warmFromSameDataOnly() || reuseSameCVFolds))
        {
            /* we want all of the first parameter (which is the warm paramter, 
             * taken care of for us) values done in a group. So We can get this
             * by just dividing up the larger list into sub lists, each sub list
             * is adjacent in the original and is the number of parameter values
             * we wanted to try
             */
            
            int stepSize = searchValues.get(0).size();
            int totalJobs = paramsToEval.size()/stepSize;
            latch = new CountDownLatch(totalJobs);
            for(int startPos = 0; startPos < paramsToEval.size(); startPos += stepSize)
            {
                final List<Classifier> subSet = paramsToEval.subList(startPos, startPos+stepSize);
                modelService.submit(new Runnable()
                {

                    @Override
                    public void run()
                    {
                        Classifier[] prevModels = null;
                        
                        for(Classifier c : subSet)
                        {
                            ClassificationModelEvaluation cme = trainModelsInParallel ?
                                    new ClassificationModelEvaluation(c, dataSet) 
                                    : new ClassificationModelEvaluation(c, dataSet, threadPool);
                            cme.setKeepModels(true);//we need these to do warm starts!
                            cme.setWarmModels(prevModels);
                            cme.addScorer(classificationTargetScore.clone());
                            if(reuseSameCVFolds)
                                cme.evaluateCrossValidation(preFolded, trainCombinations);
                            else
                                cme.evaluateCrossValidation(folds);
                            prevModels = cme.getKeptModels();
                            synchronized(bestModels)
                            {
                                bestModels.add(cme);
                            }
                        }
                        latch.countDown();
                    }
                });
            }
        }
        else//regular CV, train a new model from scratch at every step
        {
            latch = new CountDownLatch(paramsToEval.size());
            
            for (final Classifier toTrain : paramsToEval)
            {

                modelService.submit(new Runnable()
                {

                    @Override
                    public void run()
                    {
                        ClassificationModelEvaluation cme = trainModelsInParallel ?
                                    new ClassificationModelEvaluation(toTrain, dataSet) 
                                    : new ClassificationModelEvaluation(toTrain, dataSet, threadPool);
                        cme.addScorer(classificationTargetScore.clone());
                        if(reuseSameCVFolds)
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
            }
        }

        //now wait for everyone to finish
        try
        {
            latch.await();
            //Now we know the best classifier, we need to train one on the whole data set. 
            Classifier bestClassifier = bestModels.peek().getClassifier();//Just re-train it on the whole set
            if(trainFinalModel)
            {
                //try and warm start the final model if we can
                if(useWarmStarts && bestClassifier instanceof WarmClassifier && 
                        !((WarmClassifier)bestClassifier).warmFromSameDataOnly())//last line here needed to make sure we can do this warm train
                {
                    WarmClassifier wc = (WarmClassifier) bestClassifier;
                    if(threadPool instanceof FakeExecutor)
                        wc.trainC(dataSet, wc.clone());
                    else
                        wc.trainC(dataSet, wc.clone(), threadPool);
                }
                else
                {
                    if(threadPool instanceof FakeExecutor)
                        bestClassifier.trainC(dataSet);
                    else
                        bestClassifier.trainC(dataSet, threadPool);
                }
            }
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
        GridSearch clone; 
        if(baseClassifier != null)
            clone = new GridSearch(baseClassifier.clone(), folds);
        else
            clone = new GridSearch(baseRegressor.clone(), folds);
        clone.classificationTargetScore = this.classificationTargetScore.clone();
        clone.regressionTargetScore = this.regressionTargetScore.clone();
        clone.useWarmStarts = this.useWarmStarts;
        clone.trainModelsInParallel = this.trainModelsInParallel;
        clone.trainFinalModel = this.trainFinalModel;
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
