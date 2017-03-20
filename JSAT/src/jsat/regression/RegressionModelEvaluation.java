package jsat.regression;

import static java.lang.Math.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.*;
import jsat.datatransform.DataTransformProcess;
import jsat.exceptions.UntrainedModelException;
import jsat.math.OnLineStatistics;
import jsat.regression.evaluation.RegressionScore;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;

/**
 * Provides a mechanism to quickly evaluate a regression model on a data set. 
 * This can be done by cross validation or with a separate testing set. 
 * 
 * @author Edward Raff
 */
public class RegressionModelEvaluation
{
    private Regressor regressor;
    
    private RegressionDataSet dataSet;
    
    /**
     * The source of threads
     */
    private ExecutorService threadpool;
    
    private OnLineStatistics sqrdErrorStats;
    
    private long totalTrainingTime = 0, totalClassificationTime = 0;
    
    private DataTransformProcess dtp;
    
    private Map<RegressionScore, OnLineStatistics> scoreMap;
    private boolean keepModels = false;
    /**
     * This holds models for each index that will be kept. If using a test set,
     * only index 0 is used.
     */
    private Regressor[] keptModels;
    /**
     * This holds models for each fold index that will be used for warm starts.
     * If using a test set, only index 0 is used.
     */
    private Regressor[] warmModels;

    /**
     * Creates a new RegressionModelEvaluation that will perform parallel training. 
     * @param regressor the regressor model to evaluate
     * @param dataSet the data set to train or perform cross validation from
     * @param threadpool the source of threads for training of models
     */
    public RegressionModelEvaluation(Regressor regressor, RegressionDataSet dataSet, ExecutorService threadpool)
    {
        this.regressor = regressor;
        this.dataSet = dataSet;
        this.threadpool = threadpool;
        this.dtp =new DataTransformProcess();
        
        scoreMap = new LinkedHashMap<RegressionScore, OnLineStatistics>();
    }
    
    /**
     * Creates a new RegressionModelEvaluation that will perform serial training
     * @param regressor the regressor model to evaluate
     * @param dataSet the data set to train or perform cross validation from
     */
    public RegressionModelEvaluation(Regressor regressor, RegressionDataSet dataSet)
    {
        this(regressor, dataSet, null);
    }
    
    /**
     * Set this to {@code true} in order to keep the trained models after
     * evaluation. They can then be retrieved used the {@link #getKeptModels() }
     * methods. The default value is {@code false}.
     *
     * @param keepModels {@code true} to keep the trained models after
     * evaluation, {@code false} to discard them.
     */
    public void setKeepModels(boolean keepModels)
    {
        this.keepModels = keepModels;
    }

    /**
     * This will keep the models trained when evaluating the model. The models
     * can be obtained after an evaluation from {@link #getKeptModels() }.
     *
     * @return {@code true} if trained models will be kept after evaluation.
     */
    public boolean isKeepModels()
    {
        return keepModels;
    }

    /**
     * Returns the models that were kept after the last evaluation. {@code null}
     * will be returned instead if {@link #isKeepModels() } returns
     * {@code false}, which is the default.
     *
     * @return the models that were kept after the last evaluation. Or
     * {@code null} if if models are not being kept.
     */
    public Regressor[] getKeptModels()
    {
        return keptModels;
    }

    /**
     * Sets the models that will be used for warm starting training. If using
     * cross-validation, the number of models given should match the number of
     * folds. If using a test set, only one model should be given.
     *
     * @param warmModels the models to use for warm start training
     */
    public void setWarmModels(Regressor... warmModels)
    {
        this.warmModels = warmModels;
    }
    
    /**
     * Sets the data transform process to use when performing cross validation. 
     * By default, no transforms are applied
     * @param dtp the transformation process to clone for use during evaluation
     */
    public void setDataTransformProcess(DataTransformProcess dtp)
    {
        this.dtp = dtp.clone();
    }
    
    /**
     * Performs an evaluation of the regressor using the training data set. 
     * The evaluation is done by performing cross validation.
     * @param folds the number of folds for cross validation
     * @throws UntrainedModelException if the number of folds given is less than 2
     */
    public void evaluateCrossValidation(int folds)
    {
        evaluateCrossValidation(folds, RandomUtil.getRandom());
    }
    
    /**
     * Performs an evaluation of the regressor using the training data set. 
     * The evaluation is done by performing cross validation.
     * @param folds the number of folds for cross validation
     * @param rand the source of randomness for generating the cross validation sets
     * @throws UntrainedModelException if the number of folds given is less than 2
     */
    public void evaluateCrossValidation(int folds, Random rand)
    {
        if(folds < 2)
            throw new UntrainedModelException("Model could not be evaluated because " + folds + " is < 2, and not valid for cross validation");
        
        List<RegressionDataSet> lcds = dataSet.cvSet(folds, rand);
        evaluateCrossValidation(lcds);
    }
    
    /**
     * Performs an evaluation of the regressor using the training data set,
     * where the folds of the training data set are provided by the user. The
     * folds do not need to be the same sizes, though it is assumed that they
     * are all approximately the same size. It is the caller's responsibility to
     * ensure that the folds are only from the original training data set. <br>
     * <br>
     * This method exists so that the user can provide very specific folds if
     * they so desire. This can be useful when there is known bias in the data
     * set, such as when caused by duplicate data point values. The caller can
     * then manually make sure duplicate values all occur in the same fold to
     * avoid over-estimating the accuracy of the model.
     *
     * @param lcds the training data set already split into folds
     */
    public void evaluateCrossValidation(List<RegressionDataSet> lcds)
    {
        List<RegressionDataSet> trainCombinations = new ArrayList<RegressionDataSet>(lcds.size());
        for (int i = 0; i < lcds.size(); i++)
            trainCombinations.add(RegressionDataSet.comineAllBut(lcds, i));
        evaluateCrossValidation(lcds, trainCombinations);
    }
    
    /**
     * Note: Most people should never need to call this method. Make sure you
     * understand what you are doing before you do.<br>
     * <br>
     * Performs an evaluation of the regressor using the training data set,
     * where the folds of the training data set, and their combinations, are
     * provided by the user. The folds do not need to be the same sizes, though
     * it is assumed that they are all approximately the same size - and the the
     * training combination corresponding to each index will be the sum of the
     * folds in the other indices. It is the caller's responsibility to ensure
     * that the folds are only from the original training data set. <br>
     * <br>
     * This method exists so that the user can provide very specific folds if
     * they so desire, and when the same folds will be used multiple times.
     * Doing so allows the algorithms called to take advantage of any potential
     * caching of results based on the data set and avoid all possible excessive
     * memory movement. (For example, {@link jsat.DataSet#getNumericColumns() } may
     * get re-used and benefit from its caching)<br>
     * The same behavior of this method can be obtained by calling {@link #evaluateCrossValidation(java.util.List)
     * }.
     *
     * @param lcds training data set already split into folds
     * @param trainCombinations each index contains the training data sans the
     * data stored in the fold associated with that index
     */
    public void evaluateCrossValidation(List<RegressionDataSet> lcds, List<RegressionDataSet> trainCombinations)
    {
        
        sqrdErrorStats = new OnLineStatistics();
        totalTrainingTime = totalClassificationTime = 0;
        
        for(int i = 0; i < lcds.size(); i++)
        {
            RegressionDataSet trainSet = trainCombinations.get(i);
            RegressionDataSet testSet = lcds.get(i);
            evaluationWork(trainSet, testSet, i);
        }
    }
    
    /**
     * Performs an evaluation of the regressor using the initial data set to 
     * train, and testing on the given data set. 
     * @param testSet the data set to perform testing on
     */
    public void evaluateTestSet(RegressionDataSet testSet)
    {
        sqrdErrorStats = new OnLineStatistics();
        totalTrainingTime = totalClassificationTime = 0;
        evaluationWork(dataSet, testSet, 0);
    }
    
    private void evaluationWork(RegressionDataSet trainSet, RegressionDataSet testSet, int index)
    {
        trainSet = trainSet.shallowClone();
        DataTransformProcess curProccess = dtp.clone();
        curProccess.learnApplyTransforms(trainSet);
        
        long startTrain = System.currentTimeMillis();
        final Regressor regressorTouse = regressor.clone();
        if(warmModels != null && regressorTouse instanceof WarmRegressor)//train from the warm model
        {
            WarmRegressor wr = (WarmRegressor) regressorTouse;
            if(threadpool != null)
                wr.train(trainSet, warmModels[index], threadpool);
            else
                wr.train(trainSet, warmModels[index]);
        }
        else//do the normal thing
        {
            if(threadpool != null)
                regressorTouse.train(trainSet, threadpool);
            else
                regressorTouse.train(trainSet);
        }
        totalTrainingTime += (System.currentTimeMillis() - startTrain);
        if(keptModels != null)
            keptModels[index] = regressorTouse;
        
        //place to store the scores that may get updated by several threads
        final Map<RegressionScore, RegressionScore> scoresToUpdate = new HashMap<RegressionScore, RegressionScore>();
        for(Entry<RegressionScore, OnLineStatistics> entry : scoreMap.entrySet())
        {
            RegressionScore score = entry.getKey().clone();
            score.prepare();
            scoresToUpdate.put(score, score);
        }
        
        CountDownLatch latch;
        if(testSet.getSampleSize() < SystemInfo.LogicalCores || threadpool == null)
        {
            latch = new CountDownLatch(1);
            new Evaluator(testSet, curProccess, 0, testSet.getSampleSize(), scoresToUpdate, regressorTouse, latch).run();
        }
        else//go parallel!
        {
            latch = new CountDownLatch(SystemInfo.LogicalCores);
            final int blockSize = testSet.getSampleSize()/SystemInfo.LogicalCores;
            int extra = testSet.getSampleSize()%SystemInfo.LogicalCores;
            
            int start = 0;
            while(start < testSet.getSampleSize())
            {
                int end = start+blockSize;
                if(extra-- > 0)
                    end++;
                threadpool.submit(new Evaluator(testSet, curProccess, start, end, scoresToUpdate, regressorTouse, latch));
                start = end;
            }
        }
        try
        {
            latch.await();
            //accumulate score info
            for(Entry<RegressionScore, OnLineStatistics> entry : scoreMap.entrySet())
            {
                RegressionScore score = entry.getKey().clone();
                score.prepare();
                score.addResults(scoresToUpdate.get(score));
                entry.getValue().add(score.getScore());
            }
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(ClassificationModelEvaluation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * Adds a new score object that will be used as part of the evaluation when 
     * calling {@link #evaluateCrossValidation(int, java.util.Random) } or 
     * {@link #evaluateTestSet(jsat.regression.RegressionDataSet) }. The 
     * statistics for the given score are reset on every call, and the mean / 
     * standard deviation comes from multiple folds in cross validation. <br>
     * <br>
     * The score statistics can be obtained from 
     * {@link #getScoreStats(jsat.regression.evaluation.RegressionScore) }
     * after one of the evaluation methods have been called. 
     * 
     * @param scorer the score method to keep track of. 
     */
    public void addScorer(RegressionScore scorer)
    {
        scoreMap.put(scorer, new OnLineStatistics());
    }
    
    /**
     * Gets the statistics associated with the given score. If the score is not
     * currently in the model evaluation {@code null} will be returned. The 
     * object passed in does not need to be the exact same object passed to 
     * {@link #addScorer(jsat.regression.evaluation.RegressionScore) },
     * it only needs to be equal to the object. 
     * 
     * @param score the score type to get the result statistics
     * @return the result statistics for the given score, or {@code null} if the 
     * score is not in th evaluation set
     */
    public OnLineStatistics getScoreStats(RegressionScore score)
    {
        return scoreMap.get(score);
    }
    
    private class Evaluator implements Runnable
    {
        RegressionDataSet testSet;
        DataTransformProcess curProccess;
        int start, end;
        CountDownLatch latch;
        long localPredictionTime;
        Regressor toUse;
        final Map<RegressionScore, RegressionScore> scoresToUpdate;

        public Evaluator(RegressionDataSet testSet, DataTransformProcess curProccess, int start, int end, Map<RegressionScore, RegressionScore> scoresToUpdate, Regressor toUse, CountDownLatch latch)
        {
            this.testSet = testSet;
            this.curProccess = curProccess;
            this.start = start;
            this.end = end;
            this.latch = latch;
            localPredictionTime = 0;
            this.toUse = toUse;
            this.scoresToUpdate = scoresToUpdate;
        }

        @Override
        public void run()
        {
            try
            {
                //create a local set of scores to update
                Set<RegressionScore> localScores = new HashSet<RegressionScore>();
                for (Entry<RegressionScore, RegressionScore> entry : scoresToUpdate.entrySet())
                    localScores.add(entry.getKey().clone());
                for (int i = start; i < end; i++)
                {
                    DataPoint di = testSet.getDataPoint(i);
                    double trueVal = testSet.getTargetValue(i);
                    DataPoint tranDP = curProccess.transform(di);
                    long startTime = System.currentTimeMillis();
                    double predVal = toUse.regress(tranDP);
                    localPredictionTime += (System.currentTimeMillis() - startTime);

                    double sqrdError = pow(trueVal - predVal, 2);
                    
                    for (RegressionScore score : localScores)
                        score.addResult(predVal, trueVal, di.getWeight());

                    synchronized (sqrdErrorStats)
                    {
                        sqrdErrorStats.add(sqrdError, di.getWeight());
                    }
                }
                
                synchronized (sqrdErrorStats)
                {
                    totalClassificationTime += localPredictionTime;
                    for (RegressionScore score : localScores)
                        scoresToUpdate.get(score).addResults(score);
                }
                latch.countDown();
            }
            catch (Exception ex)
            {
                ex.printStackTrace();
            }
        }
        
    }
    
    /**
     * Prints out the classification information in a convenient format. If no
     * additional scores were added via the 
     * {@link #addScorer(RegressionScore) }
     * method, nothing will be printed. 
     */
    public void prettyPrintRegressionScores()
    {
        int nameLength = 10;
        for(Entry<RegressionScore, OnLineStatistics> entry : scoreMap.entrySet())
            nameLength = Math.max(nameLength, entry.getKey().getName().length()+2);
        final String pfx = "%-" + nameLength;//prefix
        for(Entry<RegressionScore, OnLineStatistics> entry : scoreMap.entrySet())
            System.out.printf(pfx+"s %-5f (%-5f)\n", entry.getKey().getName(), entry.getValue().getMean(), entry.getValue().getStandardDeviation());
    }
    
    /**
     * Returns the minimum squared error from all runs. 
     * @return the minimum observed squared error
     */
    public double getMinError()
    {
        return sqrdErrorStats.getMin();
    }
    
    /**
     * Returns the maximum squared error observed from all runs. 
     * @return the maximum observed squared error
     */
    public double getMaxError()
    {
        return sqrdErrorStats.getMax();
    }
    
    /**
     * Returns the mean squared error from all runs. 
     * @return the overall mean squared error
     */
    public double getMeanError()
    {
        return sqrdErrorStats.getMean();
    }
    
    /**
     * Returns the standard deviation of the error from all runs
     * @return the overall standard deviation of the errors
     */
    public double getErrorStndDev()
    {
        return sqrdErrorStats.getStandardDeviation();
    }
    
    /***
     * Returns the total number of milliseconds spent training the regressor. 
     * @return the total number of milliseconds spent training the regressor. 
     */
    public long getTotalTrainingTime()
    {
        return totalTrainingTime;
    }

    /**
     * Returns the total number of milliseconds spent performing regression on the testing set. 
     * @return the total number of milliseconds spent performing regression on the testing set. 
     */
    public long getTotalClassificationTime()
    {
        return totalClassificationTime;
    }
    
    /**
     * Returns the regressor that was to be evaluated
     * @return the regressor original given
     */
    public Regressor getRegressor()
    {
        return regressor;
    }
    
}
