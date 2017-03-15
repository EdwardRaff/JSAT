
package jsat.classifiers;

import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.evaluation.ClassificationScore;
import jsat.datatransform.DataTransformProcess;
import jsat.exceptions.UntrainedModelException;
import jsat.math.OnLineStatistics;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;

/**
 * Provides a mechanism to quickly perform an evaluation of a model on a data set. 
 * This can be done with cross validation or with a testing set. 
 * 
 * @author Edward Raff
 */
public class ClassificationModelEvaluation
{
    /**
     * The model to evaluate
     */
    private Classifier classifier;
    /**
     * The data set to train with. 
     */
    private ClassificationDataSet dataSet;
    /**
     * The source of threads
     */
    private ExecutorService threadpool;
    private double[][] confusionMatrix;
    /**
     * The sum of all the weights for each data point that was used in testing. 
     */
    private double sumOfWeights;
    private long totalTrainingTime = 0, totalClassificationTime = 0;
    private DataTransformProcess dtp;
    private boolean keepPredictions;
    private CategoricalResults[] predictions;
    private int[] truths;
    private double[] pointWeights;
    private OnLineStatistics errorStats;
    private Map<ClassificationScore, OnLineStatistics> scoreMap;
    private boolean keepModels = false;
    /**
     * This holds models for each index that will be kept. If using a test set,
     * only index 0 is used.
     */
    private Classifier[] keptModels;
    /**
     * This holds models for each fold index that will be used for warm starts.
     * If using a test set, only index 0 is used.
     */
    private Classifier[] warmModels;
    
    /**
     * Constructs a new object that can perform evaluations on the model. 
     * The model will not be trained until evaluation time. 
     * 
     * @param classifier the model to train and evaluate
     * @param dataSet the training data set. 
     */
    public ClassificationModelEvaluation(Classifier classifier, ClassificationDataSet dataSet)
    {
        this(classifier, dataSet, null);
    }
    
    /**
     * Constructs a new object that can perform evaluations on the model. 
     * The model will not be trained until evaluation time. 
     * 
     * @param classifier the model to train and evaluate
     * @param dataSet the training data set. 
     * @param threadpool the source of threads for parallel training. 
     * If set to null, training will be done using the 
     * {@link Classifier#trainC(jsat.classifiers.ClassificationDataSet) } method. 
     */
    public ClassificationModelEvaluation(Classifier classifier, ClassificationDataSet dataSet, ExecutorService threadpool)
    {
        this.classifier = classifier;
        this.dataSet = dataSet;
        this.threadpool = threadpool;
        this.dtp = new DataTransformProcess();
        keepPredictions = false;
        errorStats = new OnLineStatistics();
        scoreMap = new LinkedHashMap<ClassificationScore, OnLineStatistics>();
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
    public Classifier[] getKeptModels()
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
    public void setWarmModels(Classifier... warmModels)
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
     * Performs an evaluation of the classifier using the training data set. 
     * The evaluation is done by performing cross validation.
     * @param folds the number of folds for cross validation
     * @throws UntrainedModelException if the number of folds given is less than 2
     */
    public void evaluateCrossValidation(int folds)
    {
        evaluateCrossValidation(folds, RandomUtil.getRandom());
    }
    
    /**
     * Performs an evaluation of the classifier using the training data set. 
     * The evaluation is done by performing cross validation.
     * @param folds the number of folds for cross validation
     * @param rand the source of randomness for generating the cross validation sets
     * @throws UntrainedModelException if the number of folds given is less than 2
     */
    public void evaluateCrossValidation(int folds, Random rand)
    {
        if(folds < 2)
            throw new UntrainedModelException("Model could not be evaluated because " + folds + " is < 2, and not valid for cross validation");
        List<ClassificationDataSet> lcds = dataSet.cvSet(folds, rand);
        evaluateCrossValidation(lcds);
    }
    
    /**
     * Performs an evaluation of the classifier using the training data set,
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
    public void evaluateCrossValidation(List<ClassificationDataSet> lcds)
    {
        List<ClassificationDataSet> trainCombinations = new ArrayList<ClassificationDataSet>(lcds.size());
        for (int i = 0; i < lcds.size(); i++)
            trainCombinations.add(ClassificationDataSet.comineAllBut(lcds, i));
        evaluateCrossValidation(lcds, trainCombinations);
    }
    
    /**
     * Note: Most people should never need to call this method. Make sure you
     * understand what you are doing before you do.<br>
     * <br>
     * Performs an evaluation of the classifier using the training data set,
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
     * memory movement. (For example, {@link DataSet#getNumericColumns() } may
     * get re-used and benefit from its caching)<br>
     * The same behavior of this method can be obtained by calling {@link #evaluateCrossValidation(java.util.List)
     * }.
     *
     * @param lcds training data set already split into folds
     * @param trainCombinations each index contains the training data sans the
     * data stored in the fold associated with that index
     */
    public void evaluateCrossValidation(List<ClassificationDataSet> lcds, List<ClassificationDataSet> trainCombinations)
    {
        int numOfClasses = dataSet.getClassSize();
        sumOfWeights = 0.0;
        confusionMatrix = new double[numOfClasses][numOfClasses];
        totalTrainingTime = 0;
        totalClassificationTime = 0;
        if(keepModels)
            keptModels = new Classifier[lcds.size()];
        
        setUpResults(dataSet.getSampleSize());
        int end = dataSet.getSampleSize();
        for (int i = lcds.size() - 1; i >= 0; i--)
        {
            ClassificationDataSet trainSet = trainCombinations.get(i);
            ClassificationDataSet testSet = lcds.get(i);
            evaluationWork(trainSet, testSet, i);
            int testSize = testSet.getSampleSize();
            if (keepPredictions)
            {
                System.arraycopy(predictions, 0, predictions, end - testSize, testSize);
                System.arraycopy(truths, 0, truths, end-testSize, testSize);
                System.arraycopy(pointWeights, 0, pointWeights, end-testSize, testSize);
            }
            end -= testSize;
        }
    }
    
    /**
     * Performs an evaluation of the classifier using the initial data set to train, and testing on the given data set. 
     * @param testSet the data set to perform testing on
     */
    public void evaluateTestSet(ClassificationDataSet testSet)
    {
        if(keepModels)
            keptModels = new Classifier[1];
        int numOfClasses = dataSet.getClassSize();
        sumOfWeights = 0.0;
        confusionMatrix = new double[numOfClasses][numOfClasses];
        setUpResults(testSet.getSampleSize());
        totalTrainingTime = totalClassificationTime = 0;
        evaluationWork(dataSet, testSet, 0);
    }

    private void evaluationWork(ClassificationDataSet trainSet, ClassificationDataSet testSet, int index)
    {
        DataTransformProcess curProcess = dtp.clone();
        if (curProcess.getNumberOfTransforms() > 0)
        {
            trainSet = trainSet.shallowClone();
            curProcess.learnApplyTransforms(trainSet);
        }
        final Classifier classifierToUse = classifier.clone();
        
        long startTrain = System.currentTimeMillis();
        if(warmModels != null && classifierToUse instanceof WarmClassifier)//train from the warm model
        {
            WarmClassifier wc = (WarmClassifier) classifierToUse;
            if(threadpool != null)
                wc.trainC(trainSet, warmModels[index], threadpool);
            else
                wc.trainC(trainSet, warmModels[index]);
        }
        else//do the normal thing
        {
            if(threadpool != null)
                classifierToUse.trainC(trainSet, threadpool);
            else
                classifierToUse.trainC(trainSet);
        }
        totalTrainingTime += (System.currentTimeMillis() - startTrain);
        
        if(keptModels != null)
            keptModels[index] = classifierToUse;
        
        CountDownLatch latch;
        final double[] evalErrorStats = new double[2];//first index is correct, 2nd is total
        //place to store the scores that may get updated by several threads
        final Map<ClassificationScore, ClassificationScore> scoresToUpdate = new HashMap<ClassificationScore, ClassificationScore>();
        for(Entry<ClassificationScore, OnLineStatistics> entry : scoreMap.entrySet())
        {
            ClassificationScore score = entry.getKey().clone();
            score.prepare(dataSet.getPredicting());
            scoresToUpdate.put(score, score);
        }
        
        if(testSet.getSampleSize() < SystemInfo.LogicalCores || threadpool == null)
        {
            latch = new CountDownLatch(1);
            new Evaluator(testSet, curProcess, 0, testSet.getSampleSize(), evalErrorStats, scoresToUpdate, classifierToUse ,latch).run();
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
                threadpool.submit(new Evaluator(testSet, curProcess, start, end, evalErrorStats, scoresToUpdate, classifierToUse , latch));
                start = end;
            }
        }
        try
        {
            latch.await();
            errorStats.add(evalErrorStats[0]/evalErrorStats[1]);
            //accumulate score info
            for(Entry<ClassificationScore, OnLineStatistics> entry : scoreMap.entrySet())
            {
                ClassificationScore score = entry.getKey().clone();
                score.prepare(dataSet.getPredicting());
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
     * {@link #evaluateTestSet(jsat.classifiers.ClassificationDataSet) }. The 
     * statistics for the given score are reset on every call, and the mean / 
     * standard deviation comes from multiple folds in cross validation. <br>
     * <br>
     * The score statistics can be obtained from 
     * {@link #getScoreStats(ClassificationScore) }
     * after one of the evaluation methods have been called. 
     * 
     * @param scorer the score method to keep track of. 
     */
    public void addScorer(ClassificationScore scorer)
    {
        scoreMap.put(scorer, new OnLineStatistics());
    }
    
    /**
     * Gets the statistics associated with the given score. If the score is not
     * currently in the model evaluation {@code null} will be returned. The 
     * object passed in does not need to be the exact same object passed to 
     * {@link #addScorer(ClassificationScore) },
     * it only needs to be equal to the object. 
     * 
     * @param score the score type to get the result statistics
     * @return the result statistics for the given score, or {@code null} if the 
     * score is not in th evaluation set
     */
    public OnLineStatistics getScoreStats(ClassificationScore score)
    {
        return scoreMap.get(score);
    }
    
    private class Evaluator implements Runnable
    {
        ClassificationDataSet testSet;
        Classifier toUse;
        DataTransformProcess curProcess;
        int start, end;
        CountDownLatch latch;
        long localClassificationTime;
        double localCorrect;
        double localSumOfWeights;
        double[] errorStats;
        final Map<ClassificationScore, ClassificationScore> scoresToUpdate;

        public Evaluator(ClassificationDataSet testSet, DataTransformProcess curProcess, int start, int end, double[] errorStats, Map<ClassificationScore, ClassificationScore> scoresToUpdate, Classifier toUse, CountDownLatch latch)
        {
            this.testSet = testSet;
            this.curProcess = curProcess;
            this.start = start;
            this.end = end;
            this.latch = latch;
            this.localClassificationTime = 0;
            this.localSumOfWeights = 0;
            this.localCorrect = 0;
            this.errorStats = errorStats;
            this.toUse = toUse;
            this.scoresToUpdate = scoresToUpdate;
        }

        @Override
        public void run()
        {
            try
            {
                //create a local set of scores to update
                Set<ClassificationScore> localScores = new HashSet<ClassificationScore>();
                for (Entry<ClassificationScore, ClassificationScore> entry : scoresToUpdate.entrySet())
                    localScores.add(entry.getKey().clone());
                for (int i = start; i < end; i++)
                {
                    DataPoint dp = testSet.getDataPoint(i);
                    dp = curProcess.transform(dp);
                    long stratClass = System.currentTimeMillis();
                    CategoricalResults result = toUse.classify(dp);
                    localClassificationTime += (System.currentTimeMillis() - stratClass);

                    for (ClassificationScore score : localScores)
                        score.addResult(result, testSet.getDataPointCategory(i), dp.getWeight());

                    if (predictions != null)
                    {
                        predictions[i] = result;
                        truths[i] = testSet.getDataPointCategory(i);
                        pointWeights[i] = dp.getWeight();
                    }
                    final int trueCat = testSet.getDataPointCategory(i);
                    synchronized(confusionMatrix[trueCat])
                    {
                        confusionMatrix[trueCat][result.mostLikely()] += dp.getWeight();
                    }
                    if(trueCat == result.mostLikely())
                        localCorrect += dp.getWeight();
                    localSumOfWeights += dp.getWeight();
                }

                synchronized(confusionMatrix)
                {
                    totalClassificationTime += localClassificationTime;
                    sumOfWeights += localSumOfWeights;
                    errorStats[0] += localSumOfWeights-localCorrect;
                    errorStats[1] += localSumOfWeights;

                    for (ClassificationScore score : localScores)
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
     * Indicates whether or not the predictions made during evaluation should be
     * stored with the expected value for retrieval later. 
     * @param keepPredictions <tt>true</tt> if space should be allocated to 
     * store the predictions made
     */
    public void keepPredictions(boolean keepPredictions)
    {
        this.keepPredictions = keepPredictions;
    }

    /**
     * 
     * @return <tt>true</tt> if the predictions are being stored
     */
    public boolean doseStoreResults()
    {
        return keepPredictions;
    }

    /**
     * If {@link #keepPredictions(boolean) } was set, this method will return 
     * the array storing the predictions made by the classifier during 
     * evaluation. These results may not be in the same order as the data set 
     * they came from, but the order is paired with {@link #getTruths() }
     * 
     * @return the array of predictions, or null
     */
    public CategoricalResults[] getPredictions()
    {
        return predictions;
    }
    
    /**
     * If {@link #keepPredictions(boolean) } was set, this method will return 
     * the array storing the target classes that should have been predicted 
     * during evaluation. These results may not be in the same order as the data
     * set they came from, but the order is paired with {@link #getPredictions()}
     * 
     * @return the array of target class values, or null
     */
    public int[] getTruths()
    {
        return truths;
    }

    /**
     * If {@link #keepPredictions(boolean) } was set, this method will return 
     * the array storing the weights for each of the points that were classified
     * 
     * @return the array of data point weights, or null
     */
    public double[] getPointWeights()
    {
        return pointWeights;
    }
    
    public double[][] getConfusionMatrix()
    {
        return confusionMatrix;
    }
    
    /**
     * Assuming that we are on the start of a new line, the confusion matrix will be pretty printed to {@link System#out System.out}
     */
    public void prettyPrintConfusionMatrix()
    {
        CategoricalData predicting = dataSet.getPredicting();
        int classCount = predicting.getNumOfCategories();
        int nameLength = 10;
        for(int i = 0; i < classCount; i++)
            nameLength = Math.max(nameLength, predicting.getOptionName(i).length()+2);
        final String pfx = "%-" + nameLength;//prefix
        
        System.out.printf(pfx+"s ", "Matrix");
        for(int i = 0; i < classCount-1; i++)
            System.out.printf(pfx+"s ", predicting.getOptionName(i).toUpperCase());
        System.out.printf(pfx+"s\n", predicting.getOptionName(classCount-1).toUpperCase());
        //Now the rows that have data! 
        for(int i = 0; i <confusionMatrix.length; i++)
        {
            System.out.printf(pfx+"s ", predicting.getOptionName(i).toUpperCase());
            for(int j = 0; j < classCount-1; j++)
                System.out.printf(pfx+"f ", confusionMatrix[i][j]);
            System.out.printf(pfx+"f\n", confusionMatrix[i][classCount-1]);
        }

    }
    
    /**
     * Prints out the classification information in a convenient format. If no
     * additional scores were added via the 
     * {@link #addScorer(ClassificationScore) }
     * method, nothing will be printed. 
     */
    public void prettyPrintClassificationScores()
    {
        int nameLength = 10;
        for(Entry<ClassificationScore, OnLineStatistics> entry : scoreMap.entrySet())
            nameLength = Math.max(nameLength, entry.getKey().getName().length()+2);
        final String pfx = "%-" + nameLength;//prefix
        for(Entry<ClassificationScore, OnLineStatistics> entry : scoreMap.entrySet())
            System.out.printf(pfx+"s %-5f (%-5f)\n", entry.getKey().getName(), entry.getValue().getMean(), entry.getValue().getStandardDeviation());
    }
    
    /**
     * Returns the total value of the weights for data points that were classified correctly. 
     * @return the total value of the weights for data points that were classified correctly. 
     */
    public double getCorrectWeights()
    {
        double val = 0.0;
        for(int i = 0; i < confusionMatrix.length; i++)
            val += confusionMatrix[i][i];
        return val;
    }

    /**
     * Returns the total value of the weights for all data points that were tested against
     * @return the total value of the weights for all data points that were tested against
     */
    public double getSumOfWeights()
    {
        return sumOfWeights;
    }
    
    /**
     * Computes the weighted error rate of the classifier. If all weights of the data 
     * points tested were equal, then the value returned is also the percent of data 
     * points that the classifier erred on. 
     * @return the weighted error rate of the classifier.
     */
    public double getErrorRate()
    {
        return 1.0 - getCorrectWeights()/sumOfWeights;
    }
    
    /**
     * Returns the object that keeps track of the error on 
     * individual evaluations. If cross-validation was used, 
     * it is the statistics for the errors of each fold. If 
     * not, it is for each time {@link #evaluateTestSet(jsat.classifiers.ClassificationDataSet) } was called. 
     * @return the statistics for the error of all evaluation sets
     */
    public OnLineStatistics getErrorRateStats()
    {
        return errorStats;
    }

    /***
     * Returns the total number of milliseconds spent training the classifier. 
     * @return the total number of milliseconds spent training the classifier. 
     */
    public long getTotalTrainingTime()
    {
        return totalTrainingTime;
    }

    /**
     * Returns the total number of milliseconds spent performing classification on the testing set. 
     * @return the total number of milliseconds spent performing classification on the testing set. 
     */
    public long getTotalClassificationTime()
    {
        return totalClassificationTime;
    }
    
    /**
     * Returns the classifier that was original given for evaluation. 
     * @return the classifier that was original given for evaluation. 
     */
    public Classifier getClassifier()
    {
        return classifier;
    }

    private void setUpResults(int resultSize)
    {
        if(keepPredictions)
        {
            predictions = new CategoricalResults[resultSize];
            truths = new int[predictions.length];
            pointWeights = new double[predictions.length];
        }
        else
        {
            predictions = null;
            truths = null;
            pointWeights = null;
        }
    }
}
