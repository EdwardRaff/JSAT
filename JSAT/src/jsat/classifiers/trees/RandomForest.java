package jsat.classifiers.trees;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.boosting.Bagging;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.math.OnLineStatistics;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * Random Forest is an extension of {@link Bagging} that is applied only to {@link DecisionTree DecisionTrees}. 
 * It works in a similar manner, but also only uses a random sub set of the features for each tree trained. 
 * This provides increased performance in accuracy of predictions, and reduced training time over just Bagging. 
 * 
 * @author Edward Raff
 * @see Bagging
 */
public class RandomForest implements Classifier, Regressor
{
    //TODO implement Out of Bag estimates of proximity, importance, and outlier detection 
    
    /**
     * Only used when training for a classification problem
     */
    private CategoricalData predicting;
    private int extraSamples;
    /**
     * Setting the number of features to use. Default value is -1, indicating the heuristic 
     * of sqrt(N) or N/3 should be used for classification and regression respectively. This
     * value should be set away from -1 before training work begins, and set back if it 
     * was not set explicitly by the used
     */
    private int featureSamples;
    private int maxForestSize;
    private DecisionTree baseLearner = new DecisionTree(Integer.MAX_VALUE, 10, DecisionTree.PruningMethod.NONE, 0.01);
    private List<DecisionTree> forest;

    public RandomForest(int maxForestSize)
    {
        setExtraSamples(0);
        setMaxForestSize(maxForestSize);
        autoFeatureSample();
    }
    
    /**
     * RandomForest performs Bagging. Bagging samples from the training set with replacement, and draws 
     * a sampleWithReplacement at least as large as the training set. This controls how many extra samples are 
     * taken. If negative, fewer samples will be taken. Using negative values is not recommended. 
     * 
     * @param i how many extra samples to take
     */
    public void setExtraSamples(int i)
    {
        extraSamples = i;
    }

    public int getExtraSamples()
    {
        return extraSamples;
    }

    /**
     * Instead of using a heuristic, the exact number of features to sample is provided. 
     * If equal to or larger then the number of features in one of the training data sets,
     * RandomForest degrades to {@link Bagging} performed on {@link DecisionTree}.<br>
     * <br>
     * To re-enable the heuristic mode, call {@link #autoFeatureSample() }
     * 
     * @param featureSamples the number of features to randomly select for each tree in the forest. 
     * @throws ArithmeticException if the number given is less then or equal to zero
     * @see #autoFeatureSample() 
     * @see Bagging
     */
    public void setFeatureSamples(int featureSamples)
    {
        if(featureSamples <= 0)
            throw new ArithmeticException("A positive number of features must be given");
        this.featureSamples = featureSamples;
    }
    
    /**
     * Tells the class to automatically select the number of features to use. For 
     * classification problems, this is the square root of the number of features.
     * For regression, the number of features divided by 3 is used. 
     */
    public void autoFeatureSample()
    {
        featureSamples = -1;
    }
    
    /**
     * Returns true if heuristics are currently in use for the number of features, or false if the number has been specified. 
     * @return true if heuristics are currently in use for the number of features, or false if the number has been specified. 
     */
    public boolean isAutoFeatureSample()
    {
        return featureSamples == -1;
    }
    
    /**
     * Sets the maximum number of trees to create for the forest. 
     * @param maxForestSize the number of base learners to train
     * @throws ArithmeticException if the number specified is not a positive value
     */
    public void setMaxForestSize(int maxForestSize)
    {
        if(maxForestSize <= 0)
            throw new ArithmeticException("Must train a positive number of learners");
        this.maxForestSize = maxForestSize;
    }

    /**
     * Returns the number of rounds of boosting that will be done, which is also the number of base learners that will be trained
     * @return the number of rounds of boosting that will be done, which is also the number of base learners that will be trained
     */
    public int getMaxForestSize()
    {
        return maxForestSize;
    }

    public CategoricalResults classify(DataPoint data)
    {
        if(forest == null || forest.isEmpty())
            throw new RuntimeException("Classifier has not yet been trained");
        else if(predicting == null)
            throw new RuntimeException("Classifier has been trained for regression");
        CategoricalResults totalResult = new CategoricalResults(predicting.getNumOfCategories());
        for(DecisionTree tree : forest)
            totalResult.incProb(tree.classify(data).mostLikely(), 1.0);
        
        totalResult.normalize();
        return totalResult;
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        this.predicting = dataSet.getPredicting();
        this.forest = new ArrayList<DecisionTree>(maxForestSize);
        trainStep(dataSet, threadPool);
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    public boolean supportsWeightedData()
    {
        return true;
    }

    public double regress(DataPoint data)
    {
        if(forest == null || forest.isEmpty())
            throw new RuntimeException("Classifier has not yet been trained");
        else if(predicting != null)
            throw new RuntimeException("Classifier has been trained for classification");
        OnLineStatistics stats = new OnLineStatistics();
        for(DecisionTree tree : forest)
            stats.add(tree.regress(data));
        return stats.getMean();
    }

    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        this.predicting = null;
        this.forest = new ArrayList<DecisionTree>(maxForestSize);
        trainStep(dataSet, threadPool);
    }

    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, new FakeExecutor());
    }
    
    /**
     * Does the actual set up and training. {@link #predicting } and {@link #forest} should be
     * set up appropriately first. Everything else is handled by this and {@link LearningWorker}
     * 
     * @param dataSet the data set, classification or regression
     * @param threadPool the source of threads
     */
    private void trainStep(DataSet dataSet, ExecutorService threadPool)
    {
        boolean autoLearners = isAutoFeatureSample();//We will need to set it back after, so remember if we need to
        if(autoLearners)
            setFeatureSamples(Math.max((int)Math.sqrt(dataSet.getNumFeatures()), 1));
        
        int roundsToDistribut = maxForestSize;
        int roundShare = roundsToDistribut / SystemInfo.LogicalCores;//The number of rounds each thread gets
        int extraRounds = roundsToDistribut % SystemInfo.LogicalCores;//The number of extra rounds that need to get distributed
                
        if(threadPool == null || threadPool instanceof FakeExecutor)//No point in duplicatin recources
            roundShare = roundsToDistribut;//All the rounds get shoved onto one thread
        
        //Random used for creating more random objects, faster to duplicate such a small recourse then share and lock
        Random rand = new Random(dataSet.getNumFeatures() * dataSet.getSampleSize());
        List<Future<List<DecisionTree>>> futures = new ArrayList<Future<List<DecisionTree>>>(SystemInfo.LogicalCores);

        while (roundsToDistribut > 0)
        {
            int extra = (extraRounds-- > 0) ? 1 : 0;
            Future<List<DecisionTree>> future = threadPool.submit(new LearningWorker(dataSet, roundShare + extra, new Random(rand.nextInt())));
            roundsToDistribut -= (roundShare + extra);
            futures.add(future);
        }

        for (Future<List<DecisionTree>> future : futures)
            try
            {
                this.forest.addAll(future.get());
            }
            catch (Exception ex)
            {
                Logger.getLogger(RandomForest.class.getName()).log(Level.SEVERE, null, ex);
            }

        if(autoLearners)
            autoFeatureSample();
    }

    @Override
    public RandomForest clone()
    {
        RandomForest clone = new RandomForest(maxForestSize);
        clone.extraSamples = this.extraSamples;
        clone.featureSamples = this.featureSamples;
        if(this.predicting != null)
            clone.predicting = this.predicting.clone();
        if(this.forest != null)
        {
            clone.forest = new ArrayList<DecisionTree>(this.forest.size());
            for(DecisionTree tree : this.forest)
                clone.forest.add(tree.clone());
        }
        clone.baseLearner = this.baseLearner.clone();
        
        return clone;
    }
    
    private class LearningWorker implements Callable<List<DecisionTree>>
    {
        int toLearn;
        List<DecisionTree> learned;
        DataSet dataSet;
        Random random;

        public LearningWorker(DataSet dataSet, int toLearn, Random random)
        {
            this.dataSet = dataSet;
            this.toLearn = toLearn;
            this.random = random;
            this.learned = new ArrayList<DecisionTree>(toLearn);
        }
        
        public List<DecisionTree> call() throws Exception
        {
            List sample = new ArrayList(dataSet.getSampleSize()+extraSamples);
            Set<Integer> features = new HashSet<Integer>(featureSamples);
            for(int i = 0; i < toLearn; i++)
            {
                //Sample to get the training points
                Bagging.sampleWithReplacement(dataSet, sample, extraSamples, random);
                //Sample to select the feature subset
                features.clear();
                while(features.size() < Math.min(featureSamples, dataSet.getNumFeatures()))//The user could have specified too many
                    features.add(random.nextInt(dataSet.getNumFeatures()));
                
                DecisionTree learner = baseLearner.clone();
                
                if(dataSet instanceof ClassificationDataSet)
                    learner.trainC(new ClassificationDataSet(sample, predicting), features);
                else //It must be regression!
                    learner.train(new RegressionDataSet(sample),features);
                learned.add(learner);
            }
            return learned;
        }
        
    }
}
