
package jsat.classifiers.trees;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.utils.FakeExecutor;
import jsat.utils.ModifiableCountDownLatch;

/**
 * Creates a decision tree from {@link DecisionStump DecisionStumps}. How this tree 
 * performs is controlled by pruning method selected, and the methods used in the stump. 
 * 
 * @author Edward Raff
 */
public class DecisionTree implements Classifier
{
    private int maxDepth;
    private int minSamples;
    private Node root;
    private CategoricalData predicting;
    private PruningMethod pruningMethod;
    /**
     * What portion of the training data will be set aside for pruning. 
     */
    private double testProportion;
    
    public static enum PruningMethod
    {
        /**
         * The tree will be left as generated, no pruning will occur. 
         */
        NONE, 
        /**
         * The tree will be pruned in a bottom up fashion, removing 
         * leaf nodes if the remove provides an increase in accuracy 
         * on the testing set. 
         */
        REDUCED_ERROR
    };

    public DecisionTree()
    {
        maxDepth = Integer.MAX_VALUE;
        minSamples = 10;
        testProportion = 0.1;
        pruningMethod = PruningMethod.REDUCED_ERROR;
    }

    /**
     * Creates a new decision tree classifier
     * 
     * @param maxDepth the maximum depth of the tree to create
     * @param minSamples the minimum number of samples needed to continue branching
     * @param pruningMethod the method of pruning to use after construction 
     * @param testProportion the proportion of the data set to put aside to use for pruning
     */
    public DecisionTree(int maxDepth, int minSamples, PruningMethod pruningMethod, double testProportion)
    {
        setMaxDepth(maxDepth);
        setMinSamples(minSamples);
        setPruningMethod(pruningMethod);
        setTestProportion(testProportion);
    }

    /**
     * Sets the maximum depth that this classifier may build trees to. 
     * @param maxDepth the maximum depth of the trained tree
     */
    public void setMaxDepth(int maxDepth)
    {
        if(maxDepth <= 0)
            throw new RuntimeException("The maximum depth must be a positive number");
        this.maxDepth = maxDepth;
    }

    /**
     * The maximum depth that this classifier may build trees to. 
     * @return the maximum depth that this classifier may build trees to. 
     */
    public int getMaxDepth()
    {
        return maxDepth;
    }

    /**
     * Sets the minimum number of samples needed at each step in order to continue branching 
     * @param minSamples the minimum number of samples needed to branch
     */
    public void setMinSamples(int minSamples)
    {
        this.minSamples = minSamples;
    }

    /**
     * The minimum number of samples needed at each step in order to continue branching 
     * @return the minimum number of samples needed at each step in order to continue branching 
     */
    public int getMinSamples()
    {
        return minSamples;
    }

    /**
     * Sets the method of pruning that will be used after tree construction 
     * @param pruningMethod the method of pruning that will be used after tree construction 
     * @see PruningMethod
     */
    public void setPruningMethod(PruningMethod pruningMethod)
    {
        this.pruningMethod = pruningMethod;
    }
    
    /**
     * Returns the method of pruning used after tree construction 
     * @return the method of pruning used after tree construction 
     */
    public PruningMethod getPruningMethod()
    {
        return pruningMethod;
    }

    /**
     * Returns the proportion of the training set that is put aside to perform pruning with 
     * @return the proportion of the training set that is put aside to perform pruning with 
     */
    public double getTestProportion()
    {
        return testProportion;
    }

    /**
     * Sets the proportion of the training set that is put aside to perform pruning with. 
     * @param testProportion the proportion, must be in the range (0, 1)
     */
    public void setTestProportion(double testProportion)
    {
        if(testProportion <= 0 || testProportion >= 1 || Double.isInfinite(testProportion) || Double.isNaN(testProportion))
            throw new ArithmeticException("Proportion must be in the range (0, 1), not " + testProportion);
        this.testProportion = testProportion;
    }

    
    public CategoricalResults classify(DataPoint data)
    {
        return root.classify(data);
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        this.predicting = dataSet.getPredicting();
        Set<Integer> options = new HashSet<Integer>(this.predicting.getNumOfCategories());
        for(int i = 0; i < this.predicting.getNumOfCategories(); i++)
            options.add(i);
        
        ModifiableCountDownLatch mcdl = new ModifiableCountDownLatch(1);
        
        List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
        List<DataPointPair<Integer>> testPoints = new ArrayList<DataPointPair<Integer>>();
        
        if(pruningMethod != PruningMethod.NONE)//Then we need to set aside a testing set
        {
            int testSize = (int) (dataPoints.size()*testProportion);
            Random rand = new Random(testSize);
            for(int i = 0; i < testSize; i++)
                testPoints.add(dataPoints.remove(rand.nextInt(dataPoints.size())));
        }
        
        this.root = makeNode(dataPoints, options, 0, threadPool, mcdl);
        
        try
        {
            mcdl.await();
        }
        catch (InterruptedException ex)
        {
            System.err.println(ex.getMessage());
            Logger.getLogger(DecisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        prune(root, pruningMethod, testPoints);
    }
    
    private static void prune(Node root, PruningMethod method, List<DataPointPair<Integer>> testSet)
    {
        if(method == PruningMethod.NONE )
            return;
        else if(method == PruningMethod.REDUCED_ERROR)
        {
            boolean keepRunning = true;
            while (keepRunning)//Not the most efficent implementation... 
                keepRunning = pruneReduceError(null, -1, root, testSet);
        }
    }
    
    private static boolean pruneReduceError(Node parent, int pathFollowed, Node current, List<DataPointPair<Integer>> subSet)
    {
        if(current == null)
            return false;
        else if(current.isLeaf() && parent != null)//Compare this nodes accuracy vs its parrent
        {
            double childCorrect = 0;
            double parrentCorrect = 0;
            
            for(DataPointPair<Integer> dpp : subSet)
            {
                if(current.classify(dpp.getDataPoint()).mostLikely() == dpp.getPair())
                    childCorrect += dpp.getDataPoint().getWeight();
                if(parent.stump.classify(dpp.getDataPoint()).mostLikely() == dpp.getPair())
                    parrentCorrect += dpp.getDataPoint().getWeight();
            }
            
            if(parrentCorrect >= childCorrect)//We use >= b/c if they are the same, we assume smaller trees are better
            {
                parent.paths[pathFollowed] = null;//We have just made this node unreachable, will get GCed
                return true;//We made a change!
            }
            return false;
        }
        //ELSE 
        int numSplits = current.getNumberOfPaths();
        List<List<DataPointPair<Integer>>> splits = new ArrayList<List<DataPointPair<Integer>>>(numSplits);
        for(int i =0; i < numSplits; i++)
            splits.add(new ArrayList<DataPointPair<Integer>>());
        for(DataPointPair<Integer> dpp : subSet)
            splits.get(current.stump.whichPath(dpp.getDataPoint())).add(dpp);
        
        boolean madeChange = false;
        for(int i = 0; i < numSplits; i++)
        {
            boolean tmp = pruneReduceError(current, i, current.paths[i], splits.get(i));
            if(!madeChange)//Will become true on first true, never change
                madeChange = tmp;
        }
        
        return madeChange;
    }
    
    private Node makeNode(List<DataPointPair<Integer>> dataPoints, final Set<Integer> options, final int depth,
            final ExecutorService threadPool, final ModifiableCountDownLatch mcdl)
    {
        if(depth > maxDepth || options.isEmpty() || dataPoints.size() < minSamples || dataPoints.isEmpty())
        {
            mcdl.countDown();
            return null;
        }
        DecisionStump stump = new DecisionStump();
        stump.setPredicting(this.predicting);
        final List<List<DataPointPair<Integer>>> splits = stump.trainC(dataPoints, options);
        
        final Node node = new Node(stump);
        if(stump.getNumberOfPaths() > 1)//If there is 1 path, we are perfectly classifier - nothing more to do 
            for(int i = 0; i < node.paths.length; i++)
            {
                final int ii = i;
                final List<DataPointPair<Integer>> splitI = splits.get(i);
                mcdl.countUp();
                threadPool.submit(new Runnable() {

                    public void run()
                    {
                        node.paths[ii] = makeNode(splitI, new HashSet<Integer>(options), depth+1, threadPool, mcdl);
                    }
                });
            }
        
        mcdl.countDown();
        return node;
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    public boolean supportsWeightedData()
    {
        return true;
    }

    public Classifier copy()
    {
        DecisionTree copy = new DecisionTree(maxDepth, minSamples, pruningMethod, testProportion);
        copy.predicting = this.predicting.copy();
        copy.root = this.root.copy();
        return copy;
    }
    
    private static class Node
    {
        final protected DecisionStump stump;
        protected Node[] paths;
        
        public Node(DecisionStump stump)
        {
            this.stump = stump;
            paths = new Node[stump.getNumberOfPaths()];
        }
        
        public boolean isLeaf()
        {
            if(paths == null )
                return true;
            for(int i = 0; i < paths.length; i++)
                if(paths[i] != null)
                    return false;
            return true;
        }
        
        protected int getNumberOfPaths()
        {
            return paths.length;
        }
        
        protected CategoricalResults classify(DataPoint dataPoint)
        {
            int path = stump.whichPath(dataPoint);
            if(paths[path] == null)
                return stump.result(path);
            else
                return paths[path].classify(dataPoint);
        }
        
        protected Node copy()
        {
            Node copy = new Node( (DecisionStump)this.stump.copy());
            for(int i = 0; i < this.paths.length; i++)
                copy.paths[i] = this.paths[i] == null ? null : this.paths[i].copy();
            
            return copy;
        }
    }
    
}
