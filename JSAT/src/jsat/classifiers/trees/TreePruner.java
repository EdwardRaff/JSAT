package jsat.classifiers.trees;

import java.util.ArrayList;
import java.util.List;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;

/**
 * Provides post-pruning algorithms for any decision tree that can be altered 
 * using the {@link TreeNodeVisitor}. Pruning is done with a held out testing 
 * set
 * <br>
 * NOTE: API still under work, expect changes
 * 
 * @author Edward Raff
 */
public class TreePruner
{

    private TreePruner()
    {
    }
    
    
    /**
     * The method of pruning to use
     */
    public static enum PruningMethod
    {
        /**
         * The tree will be left as generated, no pruning will occur. 
         */
        NONE, 
        /**
         * The tree will be pruned in a bottom up fashion, removing 
         * leaf nodes if the removal does not reduce the accuracy on the testing
         * set
         */
        REDUCED_ERROR
    };
    
    /**
     * Performs pruning starting from the root node of a tree
     * @param root the root node of a decision tree
     * @param method the pruning method to use
     * @param testSet the test set of data points to use for pruning
     */
    public static void prune(TreeNodeVisitor root, PruningMethod method, ClassificationDataSet testSet)
    {
        prune(root, method, testSet.getAsDPPList());
    }
    
    /**
     * Performs pruning starting from the root node of a tree
     * @param root the root node of a decision tree
     * @param method the pruning method to use
     * @param testSet the test set of data points to use for pruning
     */
    public static void prune(TreeNodeVisitor root, PruningMethod method, List<DataPointPair<Integer>> testSet)
    {
        if(method == PruningMethod.NONE )
            return;
        else if(method == PruningMethod.REDUCED_ERROR)
            pruneReduceError(null, -1, root, testSet);
        else
            throw new RuntimeException("BUG: please report");
    }
    
    /**
     * Performs pruning to reduce error on the testing set
     * @param parent the parent of the current node, may be null
     * @param pathFollowed the path from the parent that lead to the current node
     * @param current the current node being considered
     * @param testSet the set of testing points to apply to this node
     * @return the number of nodes pruned from the tree
     */
    private static int pruneReduceError(TreeNodeVisitor parent, int pathFollowed, TreeNodeVisitor current, List<DataPointPair<Integer>> testSet)
    {
        if(current == null)
            return 0;
        
        int nodesPruned = 0;
        //If we are not a leaf, prune our children
        if(!current.isLeaf())
        {
            //Each child should only be given testing points that would decend down that path
            int numSplits = current.childrenCount();
            List<List<DataPointPair<Integer>>> splits = new ArrayList<List<DataPointPair<Integer>>>(numSplits);
            //TODO if splits = 2, reorder the original array and use subList to return memory efficent references
            for (int i = 0; i < numSplits; i++)
                splits.add(new ArrayList<DataPointPair<Integer>>());
            for (DataPointPair<Integer> dpp : testSet)
                splits.get(current.getPath(dpp.getDataPoint())).add(dpp);

            
            for (int i = numSplits - 1; i >= 0; i--)//Go backwards so child removals dont affect indices
                nodesPruned += pruneReduceError(current, i, current.getChild(i), splits.get(i));
        }
        
        //If we pruned all our children, we may have become a leaf! Should we prune ourselves?
        if(current.isLeaf() && parent != null)//Compare this nodes accuracy vs its parrent
        {
            double childCorrect = 0;
            double parrentCorrect = 0;
            
            for(DataPointPair<Integer> dpp : testSet)
            {
                DataPoint dp = dpp.getDataPoint();
                int truth = dpp.getPair();
                if(current.localClassify(dp).mostLikely() == truth)
                    childCorrect += dp.getWeight();
                if(parent.localClassify(dp).mostLikely() == truth)
                    parrentCorrect += dp.getWeight();
            }
            
            if(parrentCorrect >= childCorrect)//We use >= b/c if they are the same, we assume smaller trees are better
            {
                parent.disablePath(pathFollowed);
                return nodesPruned+1;//We prune our children and ourselves
            }
            
            return nodesPruned;
        }
        
        return nodesPruned;
    }
}
