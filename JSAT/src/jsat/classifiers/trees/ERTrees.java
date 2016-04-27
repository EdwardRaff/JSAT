
package jsat.classifiers.trees;

import static java.lang.Math.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;
import jsat.math.OnLineStatistics;
import jsat.regression.RegressionDataSet;
import jsat.utils.FakeExecutor;
import jsat.utils.SystemInfo;

/**
 * Extra Randomized Trees (ERTrees) is an ensemble method built on top of 
 * {@link ExtraTree}. The randomness of the trees provides incredibly high 
 * variance, yet a low bias. The sum of many randomized trees proves to be
 * a powerful and fast learner. <br>
 * The default settings are those suggested in the paper. However, the default 
 * stop size suggested (especially for classification) is often too small. You
 * may want to consider increasing it if the accuracy is too low. <br>
 * See: <br>
 * Geurts, P., Ernst, D.,&amp;Wehenkel, L. (2006). <i>Extremely randomized trees
 * </i>. Machine learning, 63(1), 3–42. doi:10.1007/s10994-006-6226-1
 * 
 * @author Edward Raff
 */
public class ERTrees extends ExtraTree
{

    private static final long serialVersionUID = 7139392253403373132L;

    //NOTE ExtraTrees uses the dynamic reflection, so extening it the new getter/setter paris are automatically picked up
    private ExtraTree baseTree = new ExtraTree();
    
    private boolean useDefaultSelectionCount = true;
    private boolean useDefaultStopSize = true;
    
    private CategoricalData predicting;
    private ExtraTree[] forrest;
    
    private int forrestSize;

    /**
     * Creates a new Extremely Randomized Trees learner
     */
    public ERTrees()
    {
        this(100);
    }

    /**
     * Creates a new Extremely Randomized Trees learner
     * @param forrestSize the number of trees to construct
     */
    public ERTrees(int forrestSize)
    {
        this.forrestSize = forrestSize;
    }
    
    /**
     * Measures the statistics of feature importance from the trees in this
     * forest. For classification datasets, the {@link MDI} method with Gini
     * impurity will be used. For others, the {@link ImportanceByUses} method
     * will be used. This may change in the future.
     *
     * @param <Type>
     * @param data the dataset to infer the feature importance from with respect
     * to the current model.
     * @return an array of statistics, which each index corresponds to a
     * specific feature. Numeric features start from the zero index, categorical
     * features start from the index equal to the number of numeric features.
     */
    public <Type extends DataSet> OnLineStatistics[] evaluateFeatureImportance(DataSet<Type> data)
    {
        if(data instanceof ClassificationDataSet)
            return evaluateFeatureImportance(data, new MDI(ImpurityScore.ImpurityMeasure.GINI));
        else
            return evaluateFeatureImportance(data, new ImportanceByUses());
    }
    
    /**
     * Measures the statistics of feature importance from the trees in this
     * forest.
     *
     * @param <Type>
     * @param data the dataset to infer the feature importance from with respect
     * to the current model.
     * @param imp the method of determing the feature importance that will be
     * applied to each tree in this model
     * @return an array of statistics, which each index corresponds to a
     * specific feature. Numeric features start from the zero index, categorical
     * features start from the index equal to the number of numeric features.
     */
    public <Type extends DataSet> OnLineStatistics[] evaluateFeatureImportance(DataSet<Type> data, TreeFeatureImportanceInference imp)
    {
        OnLineStatistics[] importances = new OnLineStatistics[data.getNumFeatures()];
        for(int i = 0; i < importances.length; i++)
            importances[i] = new OnLineStatistics();
        
        for(ExtraTree tree :forrest)
        {
            double[] feats = imp.getImportanceStats(tree, data);
            for(int i = 0; i < importances.length; i++)
                importances[i].add(feats[i]);
        }
        
        return importances;
    }

    /**
     * Sets whether or not to use the default heuristic for the number of random 
     * features to select as candidates for each node. If <tt>true</tt> the 
     * value of selectionCount will be modified during training, using sqrt(n) 
     * features for classification and all features for regression. Otherwise, 
     * whatever value set before hand will be used. 
     * @param useDefaultSelectionCount whether or not to use the heuristic 
     * version
     */
    public void setUseDefaultSelectionCount(boolean useDefaultSelectionCount)
    {
        this.useDefaultSelectionCount = useDefaultSelectionCount;
    }

    /**
     * Returns if the default heuristic for the selection count is used
     * @return if the default heuristic for the selection count is used
     */
    public boolean getUseDefaultSelectionCount()
    {
        return useDefaultSelectionCount;
    }

    /**
     * Sets whether or not to us the default heuristic for the number of points
     * to force a new node to be a leaf. If <tt>true</tt> the value for stopSize
     * will be altered during training, set to 2 for classification and 5 for 
     * regression. Otherwise, whatever value set beforehand will be used. 
     * @param useDefaultStopSize whether or not to use the heuristic version
     */
    public void setUseDefaultStopSize(boolean useDefaultStopSize)
    {
        this.useDefaultStopSize = useDefaultStopSize;
    }

    /**
     * Returns if the default heuristic for the stop size is used
     * @return if the default heuristic for the stop size is used
     */
    public boolean getUseDefaultStopSize()
    {
        return useDefaultStopSize;
    }

    public void setForrestSize(int forrestSize)
    {
        this.forrestSize = forrestSize;
    }

    public int getForrestSize()
    {
        return forrestSize;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
        
        for(ExtraTree tree : forrest)
            cr.incProb(tree.classify(data).mostLikely(), 1.0);
        cr.normalize();
        return cr;
                
    }

    private void doTraining(ExecutorService threadPool, DataSet dataSet) throws FailedToFitException
    {
        forrest = new ExtraTree[forrestSize];
        int chunkSize = forrestSize/SystemInfo.LogicalCores;
        int extra = forrestSize%SystemInfo.LogicalCores;
        
        int planted = 0;
        
        CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
        while(planted < forrestSize)
        {
            int start = planted;
            int end = start+chunkSize;
            if(extra-- > 0)
                end++;
            planted = end;
            threadPool.submit(new ForrestPlanter(start, end, dataSet, latch));
        }
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            throw new FailedToFitException(ex);
        }
    }
    
    private class ForrestPlanter implements Runnable
    {
        int start;
        int end;
        
        DataSet dataSet;
        CountDownLatch latch;

        public ForrestPlanter(int start, int end, DataSet dataSet, CountDownLatch latch)
        {
            this.start = start;
            this.end = end;
            this.dataSet = dataSet;
            this.latch = latch;
        }

        @Override
        public void run()
        {
            if(dataSet instanceof ClassificationDataSet)
            {
                ClassificationDataSet cds = (ClassificationDataSet) dataSet;
                for(int i = start; i < end; i++)
                {
                    forrest[i] = baseTree.clone();
                    forrest[i].trainC(cds);
                }
            }
            else if(dataSet instanceof RegressionDataSet)
            {
                RegressionDataSet rds = (RegressionDataSet)dataSet;
                for(int i = start; i < end; i++)
                {
                    forrest[i] = baseTree.clone();
                    forrest[i].train(rds);
                }
            }
            else
                throw new RuntimeException("BUG: Please report");
            
            latch.countDown();
        }
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if(useDefaultSelectionCount)
            baseTree.setSelectionCount((int)max(round(sqrt(dataSet.getNumFeatures())), 1));
        if(useDefaultStopSize)
            baseTree.setStopSize(2);
        
        predicting = dataSet.getPredicting();
        
        doTraining(threadPool, dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return true;
    }

    @Override
    public double regress(DataPoint data)
    {
        double mean = 0.0;
        for(ExtraTree tree : forrest)
            mean += tree.regress(data);
        return mean/forrest.length;
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        if(useDefaultSelectionCount)
            baseTree.setSelectionCount(dataSet.getNumFeatures());
        if(useDefaultStopSize)
            baseTree.setStopSize(5);
        
        doTraining(threadPool, dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, new FakeExecutor());
    }

    @Override
    public ERTrees clone()
    {
        ERTrees clone = new ERTrees();
        clone.forrestSize = this.forrestSize;
        clone.useDefaultSelectionCount = this.useDefaultSelectionCount;
        clone.useDefaultStopSize = this.useDefaultStopSize;
        clone.baseTree = this.baseTree.clone();
        if(this.predicting != null)
            clone.predicting = this.predicting.clone();
        if (this.forrest != null)
        {
            clone.forrest = new ExtraTree[this.forrest.length];
            for (int i = 0; i < this.forrest.length; i++)
                clone.forrest[i] = this.forrest[i].clone();
        }
        
        return clone;
    }

    @Override
    public TreeNodeVisitor getTreeNodeVisitor()
    {
        throw new UnsupportedOperationException("Can not get the tree node vistor becase ERTrees is really a ensemble");
    }
}
