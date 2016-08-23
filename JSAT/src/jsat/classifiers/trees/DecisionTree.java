
package jsat.classifiers.trees;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.trees.ImpurityScore.ImpurityMeasure;
import static jsat.classifiers.trees.TreePruner.*;
import jsat.classifiers.trees.TreePruner.PruningMethod;
import jsat.exceptions.FailedToFitException;
import jsat.exceptions.ModelMismatchException;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.*;

/**
 * Creates a decision tree from {@link DecisionStump DecisionStumps}. How this
 * tree performs is controlled by pruning method selected, and the methods used
 * in the stump.<br>
 * A Decision Tree supports missing values in training and prediction. 
 *
 * @author Edward Raff
 */
public class DecisionTree implements Classifier, Regressor, Parameterized, TreeLearner
{

    private static final long serialVersionUID = 9220980056440500214L;
    private int maxDepth;
    private int minSamples;
    private Node root;
    private CategoricalData predicting;
    private PruningMethod pruningMethod;
    /**
     * What portion of the training data will be set aside for pruning. 
     */
    private double testProportion;
    /**
     * Base decision stump used to clone so that we can keep certain features 
     * inside the stump instead of duplicating them here. 
     */
    private DecisionStump baseStump = new DecisionStump();

    @Override
    public double regress(DataPoint data)
    {
        if(data.numNumericalValues() != root.stump.numNumeric() || data.numCategoricalValues() != root.stump.numCategorical())
            throw new ModelMismatchException("Tree expected " + root.stump.numNumeric() + " numeric and " + 
                    root.stump.numCategorical() + " categorical features, instead received data with " + 
                    data.numNumericalValues() + " and " + data.numCategoricalValues() + " features respectively");
        return root.regress(data);
    }
    
    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        Set<Integer> options = new IntSet(dataSet.getNumFeatures());
        for(int i = 0; i < dataSet.getNumFeatures(); i++)
            options.add(i);
        train(dataSet, options, threadPool);
    }
    
    public void train(RegressionDataSet dataSet, Set<Integer> options)
    {
        train(dataSet, options, new FakeExecutor());
    }

    public void train(RegressionDataSet dataSet, Set<Integer> options, ExecutorService threadPool)
    {
        ModifiableCountDownLatch mcdl = new ModifiableCountDownLatch(1);
        root = makeNodeR(dataSet.getDPPList(), options, 0, threadPool, mcdl);
        try
        {
            mcdl.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(DecisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }
        if(root == null)//fitting issure, most likely too few datums. try just a stump 
        {
            DecisionStump stump = new DecisionStump();
            stump.train(dataSet, threadPool);
            root = new Node(stump);
        }
        //TODO add pruning for regression 
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, new FakeExecutor());
    }

    /**
     * Creates a Decision Tree that uses {@link PruningMethod#REDUCED_ERROR}
     * pruning on a held out 10% of the data.
     */
    public DecisionTree()
    {
        this(Integer.MAX_VALUE, 10, PruningMethod.REDUCED_ERROR, 0.1);
    }

    /**
     * Creates a Decision Tree that does not do any pruning, and is built out only to the specified depth
     * @param maxDepth 
     */
    public DecisionTree(int maxDepth)
    {
        this(maxDepth, 10, PruningMethod.NONE, 0.00001);
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
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected DecisionTree(DecisionTree toCopy)
    {
        this.maxDepth = toCopy.maxDepth;
        this.minSamples = toCopy.minSamples;
        if(toCopy.root != null)
            this.root = toCopy.root.clone();
        if(toCopy.predicting != null)
            this.predicting = toCopy.predicting.clone();
        this.pruningMethod = toCopy.pruningMethod;
        this.testProportion = toCopy.testProportion;
        this.baseStump = toCopy.baseStump.clone();
    }

    /**
     * Returns a Decision Tree with settings initialized so that its behavior is
     * approximately that of the C4.5 decision tree algorithm when used on 
     * classification data. The exact behavior not identical, and certain 
     * base cases may not behave in the exact same manner. However, it uses all
     * of the same general algorithms. <br><br>
     * The returned tree does not perform or support
     * <ul>
     * <li>discrete attribute grouping</li>
     * <li>windowing</li>
     * <li>subsidiary cutpoints (soft boundaries)</li>
     * </ul>
     * 
     * @return a decision tree that will behave in a manner similar to C4.5
     */
    public static DecisionTree getC45Tree()
    {
        DecisionTree tree = new DecisionTree();
        tree.setMinResultSplitSize(2);
        tree.setMinSamples(3);
        tree.setMinResultSplitSize(2);
        tree.setTestProportion(1.0);
        tree.setPruningMethod(PruningMethod.ERROR_BASED);
        tree.baseStump.setGainMethod(ImpurityMeasure.INFORMATION_GAIN_RATIO);
        return tree;
    }
    
    public void setGainMethod(ImpurityMeasure gainMethod)
    {
        baseStump.setGainMethod(gainMethod);
    }
    
    public ImpurityMeasure getGainMethod()
    {
        return baseStump.getGainMethod();
    }
    
    /**
     * When a split is made, it may be that outliers cause the split to 
     * segregate a minority of points from the majority. The min result split 
     * size parameter specifies the minimum allowable number of points to end up
     * in one of the splits for it to be admisible for consideration.
     * @param size the minimum result split size to use
     */
    public void setMinResultSplitSize(int size)
    {
        baseStump.setMinResultSplitSize(size);
    }
    
    /**
     * Returns the minimum result split size that may be considered for use as 
     * the attribute to split on.
     * @return the minimum result split size in use 
     */
    public int getMinResultSplitSize()
    {
        return baseStump.getMinResultSplitSize();
    }
    
    /**
     * Sets the maximum depth that this classifier may build trees to. 
     * @param maxDepth the maximum depth of the trained tree
     */
    public void setMaxDepth(int maxDepth)
    {
        if(maxDepth < 0)
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
     * <br> NOTE: The values 0 and 1 are special cases. <br>
     * 0 indicates that no pruning will occur regardless of the set pruning method <br>
     * 1 indicates that the training set will be used as the testing set. This is 
     * valid for some pruning methods. 
     * @param testProportion the proportion, must be in the range [0, 1]
     */
    public void setTestProportion(double testProportion)
    {
        if(testProportion < 0 || testProportion > 1 || Double.isInfinite(testProportion) || Double.isNaN(testProportion))
            throw new ArithmeticException("Proportion must be in the range [0, 1], not " + testProportion);
        this.testProportion = testProportion;
    }

    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        if(data.numNumericalValues() != root.stump.numNumeric() || data.numCategoricalValues() != root.stump.numCategorical())
            throw new ModelMismatchException("Tree expected " + root.stump.numNumeric() + " numeric and " + 
                    root.stump.numCategorical() + " categorical features, instead received data with " + 
                    data.numNumericalValues() + " and " + data.numCategoricalValues() + " features respectively");
        return root.classify(data);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        Set<Integer> options = new IntSet(dataSet.getNumFeatures());
        for(int i = 0; i < dataSet.getNumFeatures(); i++)
            options.add(i);
        trainC(dataSet, options, threadPool);
    }
    /**
     * Performs exactly the same as 
     * {@link #trainC(jsat.classifiers.ClassificationDataSet, java.util.concurrent.ExecutorService) }, 
     * but the user can specify a subset of the features to be considered.
     * 
     * @param dataSet the data set to train from
     * @param options the subset of features to split on
     * @param threadPool the source of threads for training. 
     */
    protected void trainC(ClassificationDataSet dataSet, Set<Integer> options, ExecutorService threadPool)
    {
        if(dataSet.getSampleSize() < minSamples)
            throw new FailedToFitException("There are only " + 
                    dataSet.getSampleSize() + 
                    " data points in the sample set, at least " + minSamples + 
                    " are needed to make a tree");
        this.predicting = dataSet.getPredicting();
        
        ModifiableCountDownLatch mcdl = new ModifiableCountDownLatch(1);
        
        List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
        List<DataPointPair<Integer>> testPoints = new ArrayList<DataPointPair<Integer>>();
        
        if(pruningMethod != PruningMethod.NONE && testProportion != 0.0)//Then we need to set aside a testing set
        {
            if(testProportion != 1)
            {
                int testSize = (int) (dataPoints.size()*testProportion);
                Random rand = new Random(testSize);
                for(int i = 0; i < testSize; i++)
                    testPoints.add(dataPoints.remove(rand.nextInt(dataPoints.size())));
            }
            else
                testPoints.addAll(dataPoints);
        }
        
        this.root = makeNodeC(dataPoints, options, 0, threadPool, mcdl);
        
        try
        {
            mcdl.await();
        }
        catch (InterruptedException ex)
        {
            System.err.println(ex.getMessage());
            Logger.getLogger(DecisionTree.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        if(root == null)//fitting issure, most likely too few datums. try just a stump 
        {
            DecisionStump stump = new DecisionStump();
            stump.trainC(dataSet, threadPool);
            root = new Node(stump);
        }
        else
            prune(root, pruningMethod, testPoints);
    }
    
    /**
     * Makes a new node for classification 
     * @param dataPoints the list of data points paired with their class
     * @param options the attributes that this tree may select from
     * @param depth the current depth of the tree
     * @param threadPool the source of threads
     * @param mcdl count down latch 
     * @return the node created, or null if no node was created
     */
    protected Node makeNodeC(List<DataPointPair<Integer>> dataPoints, final Set<Integer> options, final int depth,
            final ExecutorService threadPool, final ModifiableCountDownLatch mcdl)
    {
        //figure out what level of parallelism we are going to use, feature wise or depth wise
        boolean mePara = (1L<<depth) < SystemInfo.LogicalCores*2;//should THIS node use the Stump parallelism
        boolean depthPara = (1L<<(depth+1)) >= SystemInfo.LogicalCores*2;//should the NEXT node use the stump parallelism

        if(depth > maxDepth || options.isEmpty() || dataPoints.size() < minSamples || dataPoints.isEmpty())
        {
            mcdl.countDown();
            return null;
        }
        DecisionStump stump = baseStump.clone();
        stump.setPredicting(this.predicting);
        final List<List<DataPointPair<Integer>>> splits;
        if(mePara)
            splits = stump.trainC(dataPoints, options, threadPool);
        else 
            splits = stump.trainC(dataPoints, options);
        
        final Node node = new Node(stump);
        if(stump.getNumberOfPaths() > 1)//If there is 1 path, we are perfectly classifier - nothing more to do 
            for(int i = 0; i < node.paths.length; i++)
            {
                final int ii = i;
                final List<DataPointPair<Integer>> splitI = splits.get(i);
                mcdl.countUp();
                if(depthPara)
                {
                    threadPool.submit(new Runnable() {

                        public void run()
                        {
                            node.paths[ii] = makeNodeC(splitI, new IntSet(options), depth+1, threadPool, mcdl);
                        }
                    });
                }
                else
                    node.paths[ii] = makeNodeC(splitI, new IntSet(options), depth+1, threadPool, mcdl);
            }
        
        mcdl.countDown();
        return node;
    }
    
    /**
     * Makes a new node for regression
     * @param dataPoints the list of data points paired with their associated real value
     * @param options the attributes that this tree may select from 
     * @param depth the current depth of the tree
     * @param threadPool the source of threads
     * @param mcdl count down latch 
     * @return the node created, or null if no node was created
     */
    protected Node makeNodeR(List<DataPointPair<Double>> dataPoints, final Set<Integer> options, final int depth,
            final ExecutorService threadPool, final ModifiableCountDownLatch mcdl)
    {
        //figure out what level of parallelism we are going to use, feature wise or depth wise
        boolean mePara = (1L<<depth) < SystemInfo.LogicalCores*2;//should THIS node use the Stump parallelism
        boolean depthPara = (1L<<(depth+1)) >= SystemInfo.LogicalCores*2;//should the NEXT node use the stump parallelism
        
        if(depth > maxDepth || options.isEmpty() || dataPoints.size() < minSamples || dataPoints.isEmpty())
        {
            mcdl.countDown();
            return null;
        }
        DecisionStump stump = baseStump.clone();
        final List<List<DataPointPair<Double>>> splits;
        if(mePara)
            splits = stump.trainR(dataPoints, options, threadPool);
        else 
            splits = stump.trainR(dataPoints, options);
        if(splits == null)//an error occured, probably not enough data for many categorical values
        {
            mcdl.countDown();
            return null;
        }
        
        final Node node = new Node(stump);
        if(stump.getNumberOfPaths() > 1)//If there is 1 path, we are perfectly classifier - nothing more to do 
            for(int i = 0; i < node.paths.length; i++)
            {
                final int ii = i;
                final List<DataPointPair<Double>> splitI = splits.get(i);
                mcdl.countUp();
                if(depthPara)
                {
                    threadPool.submit(new Runnable() {

                        @Override
                        public void run()
                        {
                            node.paths[ii] = makeNodeR(splitI, new IntSet(options), depth+1, threadPool, mcdl);
                        }
                    });
                }
                else
                    node.paths[ii] = makeNodeR(splitI, new IntSet(options), depth+1, threadPool, mcdl);
            }
        
        mcdl.countDown();
        return node;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }
    
    public void trainC(ClassificationDataSet dataSet, Set<Integer> options)
    {
        trainC(dataSet, options, new FakeExecutor());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public DecisionTree clone()
    {
        DecisionTree copy = new DecisionTree(maxDepth, minSamples, pruningMethod, testProportion);
        if(this.predicting != null)
            copy.predicting = this.predicting.clone();
        if(this.root != null)
            copy.root = this.root.clone();
        copy.baseStump = this.baseStump.clone();
        return copy;
    }

    @Override
    public TreeNodeVisitor getTreeNodeVisitor()
    {
        return root;
    }
    
    protected static class Node extends TreeNodeVisitor
    {
        private static final long serialVersionUID = -7507748424627088734L;
        final protected DecisionStump stump;
        protected Node[] paths;
        
        public Node(DecisionStump stump)
        {
            this.stump = stump;
            paths = new Node[stump.getNumberOfPaths()];
        }

        @Override
        public double getPathWeight(int path)
        {
            return stump.pathRatio[path];
        }
        
        @Override
        public boolean isLeaf()
        {
            if(paths == null )
                return true;
            for(int i = 0; i < paths.length; i++)
                if(paths[i] != null)
                    return false;
            return true;
        }
        
        @Override
        public int childrenCount()
        {
            return paths.length;
        }

        @Override
        public CategoricalResults localClassify(DataPoint dp)
        {
            return stump.classify(dp);
        }

        @Override
        public double localRegress(DataPoint dp)
        {
            return stump.regress(dp);
        }
        
        @Override
        public Node clone()
        {
            Node copy = new Node( (DecisionStump)this.stump.clone());
            for(int i = 0; i < this.paths.length; i++)
                copy.paths[i] = this.paths[i] == null ? null : this.paths[i].clone();
            
            return copy;
        }

        @Override
        public TreeNodeVisitor getChild(int child)
        {
            if(isLeaf())
                return null;
            else
                return paths[child];
        }

        @Override
        public void setPath(int child, TreeNodeVisitor node)
        {
            if(node instanceof Node)
                paths[child] = (Node) node;
            else
                super.setPath(child, node);
        }

        @Override
        public void disablePath(int child)
        {
            paths[child] = null;
        }

        @Override
        public int getPath(DataPoint dp)
        {
            return stump.whichPath(dp);
        }

        @Override
        public boolean isPathDisabled(int child)
        {
            if(isLeaf())
                return true;
            return paths[child] == null;
        }

        @Override
        public Collection<Integer> featuresUsed()
        {
            IntList used = new IntList(1);
            used.add(stump.getSplittingAttribute());
            return used;
        }
    }
    

    @Override
    public List<Parameter> getParameters()
    {
        List<Parameter> toRet = new ArrayList<Parameter>(Parameter.getParamsFromMethods(this));
        for (Parameter param : baseStump.getParameters())//We kno the two setGainMethods will colide
            if(!param.getName().contains("Gain Method") && !param.getName().contains("Numeric Handling"))
                toRet.add(param);
        return Collections.unmodifiableList(toRet);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
    
}
