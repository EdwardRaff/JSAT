package jsat.parameters;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.distributions.Distribution;
import jsat.exceptions.FailedToFitException;
import jsat.regression.*;
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
public class GridSearch extends ModelSearch
{
    private static final long serialVersionUID = -1987196172499143753L;

    /**
     * The matching list of values we will test. This includes the integer 
     * parameters, which will have to be cast back and forth from doubles. 
     */
    private List<List<Double>> searchValues;
    
    /**
     * Use warm starts when possible
     */
    private boolean useWarmStarts = true;


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
        super(baseRegressor, folds);
        searchValues = new ArrayList<List<Double>>();
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
        super(baseClassifier, folds);
        searchValues = new ArrayList<List<Double>>();
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public GridSearch(GridSearch toCopy)
    {
        super(toCopy);
        this.useWarmStarts = toCopy.useWarmStarts;
        
        if(toCopy.searchValues != null)
        {
            this.searchValues = new ArrayList<List<Double>>();
            for(List<Double> ld : toCopy.searchValues)
            {
                List<Double> newVals = new DoubleList(ld);
                this.searchValues.add(newVals);
            }
        }
    }

    /**
     * This method will automatically populate the search space with parameters
     * based on which Parameter objects return non-null distributions. Each
     * parameter will be tested with 10 different values<br>
     * <br>
     * Note, using this method with Cross Validation has the potential for
     * over-estimating the accuracy of results if the data set is actually used
     * to for parameter guessing.<br>
     * <br>
     * It is possible for this method to return 0, indicating that no default
     * parameters could be found. The intended interpretation is that there are
     * no parameters that you <i>need</i> to tune to get good performance from
     * the given model. Though there will be cases where the author has simply
     * missed a class.
     *
     * @param data the data set to get parameter estimates from
     * @return the number of parameters added
     */
    public int autoAddParameters(DataSet data)
    {
        return autoAddParameters(data, 10);
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
     * @param paramsEach the number of parameters value to try for each parameter found
     * @return the number of parameters added
     */
    public int autoAddParameters(DataSet data, int paramsEach)
    {
        Parameterized obj;
        if(baseClassifier != null)
            obj = (Parameterized) baseClassifier;
        else 
            obj = (Parameterized) baseRegressor;
        int totalParms = 0;
        for(Parameter param : obj.getParameters())
        {
            Distribution dist;
            if (param instanceof DoubleParameter)
            {
                dist = ((DoubleParameter) param).getGuess(data);
                if (dist != null)
                    totalParms++;
            }
            else if (param instanceof IntParameter)
            {
                dist = ((IntParameter) param).getGuess(data);
                if (dist != null)
                    totalParms++;
            }
        }
        if(totalParms < 1)
            return 0;
        
        double[] quantiles = new double[paramsEach];
        for(int i = 0; i < quantiles.length; i++)
            quantiles[i] = (i+1.0)/(paramsEach+1.0);
        for(Parameter param : obj.getParameters())
        {
            Distribution dist;
            if (param instanceof DoubleParameter)
            {
                dist = ((DoubleParameter) param).getGuess(data);
                if (dist == null)
                    continue;
                double[] vals = new double[paramsEach];
                for (int i = 0; i < vals.length; i++)
                    vals[i] = dist.invCdf(quantiles[i]);

                addParameter((DoubleParameter) param, vals);

            }
            else if (param instanceof IntParameter)
            {
                dist = ((IntParameter) param).getGuess(data);
                if (dist == null)
                    continue;
                int[] vals = new int[paramsEach];
                for (int i = 0; i < vals.length; i++)
                    vals[i] = (int) Math.round(dist.invCdf(quantiles[i]));

                addParameter((IntParameter) param, vals);
            }
        }
        
        return totalParms;
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
    public GridSearch clone()
    {
        return new GridSearch(this);
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
