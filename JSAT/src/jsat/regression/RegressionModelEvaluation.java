package jsat.regression;

import static java.lang.Math.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
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
        evaluateCrossValidation(folds, new Random());
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
        
        sqrdErrorStats = new OnLineStatistics();
        totalTrainingTime = totalClassificationTime = 0;
        
        for(int i = 0; i < lcds.size(); i++)
        {
            RegressionDataSet trainSet = RegressionDataSet.comineAllBut(lcds, i);
            RegressionDataSet testSet = lcds.get(i);
            evaluationWork(trainSet, testSet);
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
        evaluationWork(dataSet, testSet);
    }
    
    private void evaluationWork(RegressionDataSet trainSet, RegressionDataSet testSet)
    {
        trainSet = trainSet.shallowClone();
        DataTransformProcess curProccess = dtp.clone();
        curProccess.learnApplyTransforms(trainSet);
        
        long startTrain = System.currentTimeMillis();
        if(threadpool != null)
            regressor.train(trainSet, threadpool);
        else
            regressor.train(trainSet);            
        totalTrainingTime += (System.currentTimeMillis() - startTrain);
        
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
            new Evaluator(testSet, curProccess, 0, testSet.getSampleSize(), scoresToUpdate, latch).run();
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
                threadpool.submit(new Evaluator(testSet, curProccess, start, end, scoresToUpdate, latch));
                start = end;
            }
        }
        try
        {
            latch.await();
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
        final Map<RegressionScore, RegressionScore> scoresToUpdate;

        public Evaluator(RegressionDataSet testSet, DataTransformProcess curProccess, int start, int end, Map<RegressionScore, RegressionScore> scoresToUpdate, CountDownLatch latch)
        {
            this.testSet = testSet;
            this.curProccess = curProccess;
            this.start = start;
            this.end = end;
            this.latch = latch;
            localPredictionTime = 0;
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
                    double predVal = regressor.regress(tranDP);
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
     * {@link #addScorer(jsat.classifiers.evaluations.ClassificationScore) }
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
