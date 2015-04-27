
package jsat.classifiers.boosting;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.*;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.classifiers.trees.DecisionTree;
import jsat.math.OnLineStatistics;
import jsat.parameters.*;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.SystemInfo;

/**
 * An implementation of Bootstrap Aggregating, as described by LEO BREIMAN in "Bagging Predictors". <br>
 * <br>
 * Bagging is an ensemble learner, it takes a weak learner and trains several to create a better over result. 
 * Bagging is particularly useful when the base classifier has some amount of predictive power, but is hindered 
 * by variance in the output (small change in input causes large change in output), or variances in input 
 * (handles noisy data badly or is has a brittle learning algorithm). It is common to perform bagging on 
 * {@link DecisionTree Decision Trees}, because they meet these strengths and weaknesses.  <br>
 * Bagging produces little to no improvement when using learners that have low variance and robust learning 
 * methods. {@link NearestNeighbour} is an example of a particularly bad method to bag. 
 * <br>
 * Bagging has many similarities to boosting. 
 * @author Edward Raff
 */
public class Bagging implements Classifier, Regressor, Parameterized
{

	private static final long serialVersionUID = -6566453570170428838L;
	private Classifier baseClassifier;
    private Regressor baseRegressor;
    private CategoricalData predicting;
    private int extraSamples;
    private int rounds;
    private boolean simultaniousTraining;
    private Random random;
    private List learners;
    
    /**
     * The number of rounds of bagging that will be used by default in the constructor: {@value #DEFAULT_ROUNDS}
     */
    public static final int DEFAULT_ROUNDS = 20;
    /**
     * The number of extra samples to take when bagging in each round used by default in the constructor: {@value #DEFAULT_EXTRA_SAMPLES}
     */
    public static final int DEFAULT_EXTRA_SAMPLES = 0;
    /**
     * The default behavior for parallel training, as specified by {@link #setSimultaniousTraining(boolean) } is {@value #DEFAULT_SIMULTANIOUS_TRAINING}
     */
    public static final boolean DEFAULT_SIMULTANIOUS_TRAINING = true;
    
    /**
     * Creates a new Bagger for classification. This can not be changed after construction.
     * 
     * @param baseClassifier the base learner to use.
     */
    public Bagging(Classifier baseClassifier)
    {
        this(baseClassifier, DEFAULT_EXTRA_SAMPLES, DEFAULT_SIMULTANIOUS_TRAINING);
    }

    /**
     * Creates a new Bagger for classification. This can not be changed after construction.
     * 
     * @param baseClassifier the base learner to use.
     * @param extraSamples how many extra samples past the training size to take
     * @param simultaniousTraining controls whether base learners are trained sequentially or simultaneously 
     */
    public Bagging(Classifier baseClassifier, int extraSamples, boolean simultaniousTraining)
    {
        this(baseClassifier, extraSamples, simultaniousTraining, DEFAULT_ROUNDS, new Random(1));
    }
    
    /**
     * Creates a new Bagger for classification. This can not be changed after construction.
     * 
     * @param baseClassifier the base learner to use.
     * @param extraSamples how many extra samples past the training size to take
     * @param simultaniousTraining controls whether base learners are trained sequentially or simultaneously 
     * @param rounds how many rounds of bagging to perform. 
     * @param random the source of randomness for sampling 
     */
    public Bagging(Classifier baseClassifier, int extraSamples, boolean simultaniousTraining, int rounds, Random random)
    {
        this(extraSamples, simultaniousTraining, rounds, random);
        this.baseClassifier = baseClassifier;
    }
    
    /**
     * Creates a new Bagger for regression. This can not be changed after construction.
     * 
     * @param baseRegressor the base learner to use.
     */
    public Bagging(Regressor baseRegressor)
    {
        this(baseRegressor, DEFAULT_EXTRA_SAMPLES, DEFAULT_SIMULTANIOUS_TRAINING);
    }
    
    /**
     * Creates a new Bagger for regression. This can not be changed after construction.
     * 
     * @param baseRegressor the base learner to use.
     * @param extraSamples how many extra samples past the training size to take
     * @param simultaniousTraining controls whether base learners are trained sequentially or simultaneously 
     */
    public Bagging(Regressor baseRegressor, int extraSamples, boolean simultaniousTraining)
    {
        this(baseRegressor, extraSamples, simultaniousTraining, DEFAULT_ROUNDS, new Random(1));
    }
    
    /**
     * Creates a new Bagger for regression. This can not be changed after construction.
     * 
     * @param baseRegressor the base learner to use.
     * @param extraSamples how many extra samples past the training size to take
     * @param simultaniousTraining controls whether base learners are trained sequentially or simultaneously 
     * @param rounds how many rounds of bagging to perform. 
     * @param random the source of randomness for sampling 
     */
    public Bagging(Regressor baseRegressor, int extraSamples, boolean simultaniousTraining, int rounds, Random random)
    {
        this(extraSamples, simultaniousTraining, rounds, random);
        this.baseRegressor = baseRegressor;
    }
    
    //For internal use
    private Bagging(int extraSamples, boolean simultaniousTraining, int rounds, Random random)
    {
        setExtraSamples(extraSamples);
        setSimultaniousTraining(simultaniousTraining);
        setRounds(rounds);
        this.random = random;
    }
    
    /**
     * Bagging samples from the training set with replacement, and draws a sampleWithReplacement at least as large
     * as the training set. This controls how many extra samples are taken. If negative, fewer 
     * samples will be taken. Using negative values is not recommended. 
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
     * Sets the number of rounds that bagging is done, meaning how many base learners are trained
     * @param rounds the number of base learners to train
     * @throws ArithmeticException if the number specified is not a positive value
     */
    public void setRounds(int rounds)
    {
        if(rounds <= 0)
            throw new ArithmeticException("Must train a positive number of learners");
        this.rounds = rounds;
    }

    /**
     * Returns the number of rounds of boosting that will be done, which is also the number of base learners that will be trained
     * @return the number of rounds of boosting that will be done, which is also the number of base learners that will be trained
     */
    public int getRounds()
    {
        return rounds;
    }
    
    /**
     * Bagging produces multiple base learners. These can all be trained at the same time, using more memory, 
     * or sequentially using the base learner's parallel training method. If set to true, the base learners 
     * will be trained simultaneously. 
     * 
     * @param simultaniousTraining true to train all learners at the same time, false to train them sequentially 
     */
    public void setSimultaniousTraining(boolean simultaniousTraining)
    {
        this.simultaniousTraining = simultaniousTraining;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(baseClassifier == null)
            throw new RuntimeException("Bagging instance created for regression, not classification");
        else if(learners == null || learners.isEmpty())
            throw new RuntimeException("Classifier has not yet been trained");
        CategoricalResults totalResult = new CategoricalResults(predicting.getNumOfCategories());
        for(int i = 0; i < learners.size(); i++)
        {
            CategoricalResults result = ((Classifier) learners.get(i)).classify(data);
            totalResult.incProb(result.mostLikely(), 1.0);
        }
        
        totalResult.normalize();
        return totalResult;
    }

    @Override
    public void trainC(final ClassificationDataSet dataSet, final ExecutorService threadPool)
    {
        predicting = dataSet.getPredicting();
        learners = new ArrayList(rounds);
        //Used to make the main thread wait for the working threads to finish before submiting a new job so we dont waist too much memory then we can use at once
        final Semaphore waitForThread = new Semaphore(SystemInfo.LogicalCores);
        //Used to make the main thread wait for the working threads to finish before returning 
        final CountDownLatch waitForFinish = new CountDownLatch(rounds);
        
        //Creat a synchrnozied view so we can add safely 
        final List synchronizedLearners = Collections.synchronizedList(learners);
        final int[] sampleCounts = new int[dataSet.getSampleSize()];
        for(int i = 0; i < rounds; i++)
        {
            sampleWithReplacement(sampleCounts, sampleCounts.length+extraSamples, random);
            final ClassificationDataSet sampleSet = getSampledDataSet(dataSet, sampleCounts);
            
            final Classifier learner = baseClassifier.clone();
            if(simultaniousTraining && threadPool != null)
            {
                try
                {
                    //Wait for an available thread
                    waitForThread.acquire();
                    threadPool.submit(new Runnable() {

                        @Override
                        public void run()
                        {
                            learner.trainC(sampleSet);
                            synchronizedLearners.add(learner);
                            waitForThread.release();//Finish, allow another one to pass through
                            waitForFinish.countDown();
                        }
                    });
                }
                catch (InterruptedException ex)
                {
                    Logger.getLogger(Bagging.class.getName()).log(Level.SEVERE, null, ex);
                    System.err.println(ex.getMessage());
                }
                
            }
            else
            {
                if(threadPool != null)
                    learner.trainC(sampleSet, threadPool);
                else
                    learner.trainC(sampleSet);
                learners.add(learner);
            }
        }

        if (simultaniousTraining && threadPool != null)
            try
            {
                waitForFinish.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(Bagging.class.getName()).log(Level.SEVERE, null, ex);
                System.err.println(ex.getMessage());
            }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, null);
    }

    /**
     * Creates a new data set from the given sample counts. Points sampled 
     * multiple times will have multiple entries in the data set. 
     * @param dataSet the data set that was sampled from
     * @param sampledCounts the sampling values obtained from 
     * {@link #sampleWithReplacement(int[], int, java.util.Random) }
     * @return a new sampled classification data set
     */
    public static ClassificationDataSet getSampledDataSet(ClassificationDataSet dataSet, int[] sampledCounts)
    {
        ClassificationDataSet destination = new ClassificationDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories(), dataSet.getPredicting());
            
            for (int i = 0; i < sampledCounts.length; i++)
                for(int j = 0; j < sampledCounts[i]; j++)
                {
                    DataPoint dp = dataSet.getDataPoint(i);
                    destination.addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), dataSet.getDataPointCategory(i));
                }
            
            return destination;
    }
    
    /**
     * Creates a new data set from the given sample counts. Points sampled
     * multiple times will be added once to the data set with their weight
     * multiplied by the number of times it was sampled. 
     * @param dataSet the data set that was sampled from
     * @param sampledCounts the sampling values obtained from 
     * {@link #sampleWithReplacement(int[], int, java.util.Random) }
     * @return a new sampled classification data set
     */
    public static ClassificationDataSet getWeightSampledDataSet(ClassificationDataSet dataSet, int[] sampledCounts)
    {
        ClassificationDataSet destination = new ClassificationDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories(), dataSet.getPredicting());

        for (int i = 0; i < sampledCounts.length; i++)
        {
            if(sampledCounts[i] <= 0)
                continue;
            DataPoint dp = dataSet.getDataPoint(i);
            destination.addDataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), dataSet.getDataPointCategory(i), dp.getWeight()*sampledCounts[i]);
        }

        return destination;
    }
    
    /**
     * Creates a new data set from the given sample counts. Points sampled 
     * multiple times will have multiple entries in the data set. 
     * @param dataSet the data set that was sampled from
     * @param sampledCounts the sampling values obtained from 
     * {@link #sampleWithReplacement(int[], int, java.util.Random) }
     * @return a new sampled classification data set
     */
    public static RegressionDataSet getSampledDataSet(RegressionDataSet dataSet, int[] sampledCounts)
    {
        RegressionDataSet destination = new RegressionDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories());
        for (int i = 0; i < sampledCounts.length; i++)
            for (int j = 0; j < sampledCounts[i]; j++)
            {
                DataPoint dp = dataSet.getDataPoint(i);
                destination.addDataPoint(dp, dataSet.getTargetValue(i));
            }
        return destination;
    }
    
    /**
     * Creates a new data set from the given sample counts. Points sampled
     * multiple times will be added once to the data set with their weight
     * multiplied by the number of times it was sampled. 
     * @param dataSet the data set that was sampled from
     * @param sampledCounts the sampling values obtained from 
     * {@link #sampleWithReplacement(int[], int, java.util.Random) }
     * @return a new sampled classification data set
     */
    public static RegressionDataSet getWeightSampledDataSet(RegressionDataSet dataSet, int[] sampledCounts)
    {
        RegressionDataSet destination = new RegressionDataSet(dataSet.getNumNumericalVars(), dataSet.getCategories());

        for (int i = 0; i < sampledCounts.length; i++)
        {
            if(sampledCounts[i] <= 0)
                continue;
            DataPoint dp = dataSet.getDataPoint(i);
            DataPoint reWeighted = new DataPoint(dp.getNumericalValues(), dp.getCategoricalValues(), dp.getCategoricalData(), dp.getWeight()*sampledCounts[i]);
            destination.addDataPoint(reWeighted, dataSet.getTargetValue(i));
        }

        return destination;
    }

    /**
     * Performs the sampling based on the number of data points, storing the 
     * counts in an array to be constructed from XXXX
     * @param sampleCounts an array to keep count of how many times each data 
     * point was sampled. The array will be filled with zeros before sampling 
     * starts
     * @param samples the number of samples to take from the data set
     * @param rand the source of randomness
     */
    static public void sampleWithReplacement(int[] sampleCounts, int samples, Random rand)
    {
        Arrays.fill(sampleCounts, 0);
        for(int i = 0; i < samples; i++)
            sampleCounts[rand.nextInt(sampleCounts.length)]++;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public double regress(DataPoint data)
    {
        if(baseRegressor == null)
            throw new RuntimeException("Bagging instance created for classification, not regression");
        else if(learners == null || learners.isEmpty())
            throw new RuntimeException("Regressor has not yet been trained");
        OnLineStatistics stats = new OnLineStatistics();
        for(int i = 0; i < learners.size(); i++)
        {
            double x = ((Regressor) learners.get(i)).regress(data);
            stats.add(x);
        }
        
        return stats.getMean();
    }

    @Override
    public void train(RegressionDataSet dataSet, final ExecutorService threadPool)
    {
        learners = new ArrayList(rounds);
        //Used to make the main thread wait for the working threads to finish before submiting a new job so we dont waist too much memory then we can use at once
        final Semaphore waitForThread = new Semaphore(SystemInfo.LogicalCores);
        //Used to make the main thread wait for the working threads to finish before returning 
        final CountDownLatch waitForFinish = new CountDownLatch(rounds);
        
        //Creat a synchrnozied view so we can add safely 
        final List synchronizedLearners = Collections.synchronizedList(learners);
        final int[] sampleCount = new int[dataSet.getSampleSize()];
        for(int i = 0; i < rounds; i++)
        {
            sampleWithReplacement(sampleCount, sampleCount.length+extraSamples, random);
            final RegressionDataSet sampleSet = getSampledDataSet(dataSet, sampleCount);
            final Regressor learner = baseRegressor.clone();
            if(simultaniousTraining && threadPool != null)
            {
                try
                {
                    //Wait for an available thread
                    waitForThread.acquire();
                    threadPool.submit(new Runnable() {

                        @Override
                        public void run()
                        {
                            learner.train(sampleSet);
                            synchronizedLearners.add(learner);
                            waitForThread.release();//Finish, allow another one to pass through
                            waitForFinish.countDown();
                        }
                    });
                }
                catch (InterruptedException ex)
                {
                    Logger.getLogger(Bagging.class.getName()).log(Level.SEVERE, null, ex);
                    System.err.println(ex.getMessage());
                }
                
            }
            else
            {
                if(threadPool != null)
                    learner.train(sampleSet, threadPool);
                else
                    learner.train(sampleSet);
                learners.add(learner);
            }
        }

        if (simultaniousTraining && threadPool != null)
            try
            {
                waitForFinish.await();
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(Bagging.class.getName()).log(Level.SEVERE, null, ex);
                System.err.println(ex.getMessage());
            }
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, null);
    }

    @Override
    public Bagging clone()
    {
        Bagging clone = new Bagging(extraSamples, simultaniousTraining, rounds, new Random(rounds));
        if(baseClassifier != null)
            clone.baseClassifier = baseClassifier.clone();
        if(predicting != null)
            clone.predicting = this.predicting.clone();
        if(baseRegressor != null)
            clone.baseRegressor = baseRegressor.clone();
        if(learners != null && !learners.isEmpty())
        {
            clone.learners = new ArrayList(this.learners.size());
            for(Object learner : learners)
                if(learner instanceof Classifier)
                    clone.learners.add( ((Classifier)learner).clone());
                else
                    clone.learners.add( ((Regressor)learner).clone());
        }
        return clone;
    }

    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
