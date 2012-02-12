
package jsat.classifiers;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.exceptions.UntrainedModelException;

/**
 * Provides a mechanism to quickly perform an evaluation of a model on a data set. 
 * This can be done with cross validation or with a testing set. 
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
    }
    
    /**
     * Performs an evaluation of the classifier using the training data set. 
     * The evaluation is done by performing cross validation.
     * @param folds the number of folds for cross validation
     * @throws UntrainedModelException if the number of folds given is less than 2
     */
    public void evaluateCrossValidation(int folds)
    {
        evaluateCrossValidation(folds, new Random());
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
        int numOfClasses = dataSet.getPredicting().getNumOfCategories();
        sumOfWeights = 0.0;
        confusionMatrix = new double[numOfClasses][numOfClasses];
        List<ClassificationDataSet> lcds = dataSet.cvSet(folds, rand);
        totalTrainingTime = 0;
        totalClassificationTime = 0;
        
        for(int i = 0; i < lcds.size(); i++)
        {
            ClassificationDataSet trainSet = ClassificationDataSet.comineAllBut(lcds, i);
            ClassificationDataSet testSet = lcds.get(i);
            evaluationWork(trainSet, testSet);
        }
    }
    /**
     * Performs an evaluation of the classifier using the initial data set to train, and testing on the given data set. 
     * @param testSet the data set to perform testing on
     */
    public void evaluateTestSet(ClassificationDataSet testSet)
    {
        int numOfClasses = dataSet.getPredicting().getNumOfCategories();
        sumOfWeights = 0.0;
        confusionMatrix = new double[numOfClasses][numOfClasses];
        totalTrainingTime = totalClassificationTime = 0;
        evaluationWork(dataSet, testSet);
    }

    private void evaluationWork(ClassificationDataSet trainSet, ClassificationDataSet testSet)
    {
        long startTrain = System.currentTimeMillis();
        if(threadpool != null)
            classifier.trainC(trainSet, threadpool);
        else
            classifier.trainC(trainSet);            
        totalTrainingTime += (System.currentTimeMillis() - startTrain);
        
        for(int j = 0; j < testSet.getPredicting().getNumOfCategories(); j++)
        {
            for (DataPoint dp : testSet.getSamples(j))
            {
                long stratClass = System.currentTimeMillis();
                CategoricalResults results = classifier.classify(dp);
                totalClassificationTime += (System.currentTimeMillis() - stratClass);
                
                confusionMatrix[j][results.mostLikely()] += dp.getWeight();
                sumOfWeights += dp.getWeight();
            }
        }
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
        System.out.printf("%-15s ", "Matrix");
        for(int i = 0; i < classCount-1; i++)
            System.out.printf("%-15s ", predicting.getOptionName(i).toUpperCase());
        System.out.printf("%-15s\n", predicting.getOptionName(classCount-1).toUpperCase());
        //Now the rows that have data! 
        for(int i = 0; i <confusionMatrix.length; i++)
        {
            System.out.printf("%-15s ", predicting.getOptionName(i).toUpperCase());
            for(int j = 0; j < classCount-1; j++)
                System.out.printf("%-15f ", confusionMatrix[i][j]);
            System.out.printf("%-15f\n", confusionMatrix[i][classCount-1]);
        }
        
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
    
}
