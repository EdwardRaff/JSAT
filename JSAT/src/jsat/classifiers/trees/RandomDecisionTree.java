
package jsat.classifiers.trees;

import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPointPair;
import jsat.utils.ModifiableCountDownLatch;
import jsat.utils.random.RandomUtil;

/**
 * An extension of Decision Trees, it ignores the given set of features to use-
 * and selects a new random subset of features at each node for use. <br>
 * <br>
 * The Random Decision Tree supports missing values in training and prediction. 
 * 
 * @author Edward Raff
 */
public class RandomDecisionTree extends DecisionTree
{

    private static final long serialVersionUID = -809244056947507494L;
    private int numFeatures;

    public RandomDecisionTree()
    {
        this(1);
    }

    /**
     * Creates a new Random Decision Tree 
     * @param numFeatures the number of random features to use
     */
    public RandomDecisionTree(int numFeatures)
    {
        setRandomFeatureCount(numFeatures);
    }

    /**
     * Creates a new Random Decision Tree
     * @param numFeatures the number of random features to use
     * @param maxDepth the maximum depth of the tree to create
     * @param minSamples the minimum number of samples needed to continue branching
     * @param pruningMethod the method of pruning to use after construction 
     * @param testProportion the proportion of the data set to put aside to use for pruning
     */
    public RandomDecisionTree(int numFeatures, int maxDepth, int minSamples, TreePruner.PruningMethod pruningMethod, double testProportion)
    {
        super(maxDepth, minSamples, pruningMethod, testProportion);
        setRandomFeatureCount(numFeatures);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public RandomDecisionTree(RandomDecisionTree toCopy)
    {
        super(toCopy);
        this.numFeatures = toCopy.numFeatures;
    }
    
    /**
     * Sets the number of random features to and use at each node of
     * the decision tree
     * @param numFeatures the number of random features
     */
    public void setRandomFeatureCount(int numFeatures)
    {
        if(numFeatures < 1)
            throw new IllegalArgumentException("Number of features must be positive, not " + numFeatures);
        this.numFeatures = numFeatures;
    }

    /**
     * Returns the number of random features used at each node of the tree
     * @return the number of random features used at each node of the tree
     */
    public int getRandomFeatureCount()
    {
        return numFeatures;
    }
    
    @Override
    protected Node makeNodeC(List<DataPointPair<Integer>> dataPoints, Set<Integer> options, int depth, ExecutorService threadPool, ModifiableCountDownLatch mcdl)
    {
        if(dataPoints.isEmpty())
        {
            mcdl.countDown();
            return null;
        }
        final int featureCount = dataPoints.get(0).getDataPoint().numCategoricalValues()+dataPoints.get(0).getDataPoint().numNumericalValues();
        fillWithRandomFeatures(options, featureCount);
        return super.makeNodeC(dataPoints, options, depth, threadPool, mcdl); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected Node makeNodeR(List<DataPointPair<Double>> dataPoints, Set<Integer> options, int depth, ExecutorService threadPool, ModifiableCountDownLatch mcdl)
    {
        if(dataPoints.isEmpty())
        {
            mcdl.countDown();
            return null;
        }
        final int featureCount = dataPoints.get(0).getDataPoint().numCategoricalValues()+dataPoints.get(0).getDataPoint().numNumericalValues();
        fillWithRandomFeatures(options, featureCount);
        return super.makeNodeR(dataPoints, options, depth, threadPool, mcdl); //To change body of generated methods, choose Tools | Templates.
    }

    private void fillWithRandomFeatures(Set<Integer> options, final int featureCount)
    {
        options.clear();
        Random rand = RandomUtil.getRandom();
        
        while(options.size() < numFeatures)
        {
            options.add(rand.nextInt(featureCount));
        }
    }

    @Override
    public RandomDecisionTree clone()
    {
        return new RandomDecisionTree(this);
    }
     
}
