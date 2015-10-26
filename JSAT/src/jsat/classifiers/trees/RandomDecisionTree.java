
package jsat.classifiers.trees;

import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPointPair;
import jsat.utils.ModifiableCountDownLatch;

/**
 * An extension of Decision Trees, it ignores the given set of features to use-
 * and selects a new random subset of features at each node for use. 
 * 
 * @author Edward Raff
 */
public class RandomDecisionTree extends DecisionTree
{

	private static final long serialVersionUID = -809244056947507494L;
	private int numFeatures;

    /**
     * Creates a new Random Decision Tree 
     * @param numFeatures the number of random features to use
     */
    public RandomDecisionTree(final int numFeatures)
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
    public RandomDecisionTree(final int numFeatures, final int maxDepth, final int minSamples, final TreePruner.PruningMethod pruningMethod, final double testProportion)
    {
        super(maxDepth, minSamples, pruningMethod, testProportion);
        setRandomFeatureCount(numFeatures);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public RandomDecisionTree(final RandomDecisionTree toCopy)
    {
        super(toCopy);
        this.numFeatures = toCopy.numFeatures;
    }
    
    /**
     * Sets the number of random features to and use at each node of
     * the decision tree
     * @param numFeatures the number of random features
     */
    public void setRandomFeatureCount(final int numFeatures)
    {
        if(numFeatures < 1) {
          throw new IllegalArgumentException("Number of features must be positive, not " + numFeatures);
        }
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
    protected Node makeNodeC(final List<DataPointPair<Integer>> dataPoints, final Set<Integer> options, final int depth, final ExecutorService threadPool, final ModifiableCountDownLatch mcdl)
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
    protected Node makeNodeR(final List<DataPointPair<Double>> dataPoints, final Set<Integer> options, final int depth, final ExecutorService threadPool, final ModifiableCountDownLatch mcdl)
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

    private void fillWithRandomFeatures(final Set<Integer> options, final int featureCount)
    {
        options.clear();
        final Random rand = new Random();
        
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
