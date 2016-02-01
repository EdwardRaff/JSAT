package jsat.classifiers.trees;

import java.util.ArrayList;
import java.util.List;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.math.SpecialMath;

/**
 * Provides post-pruning algorithms for any decision tree that can be altered 
 * using the {@link TreeNodeVisitor}. Pruning is done with a held out testing 
 * set
 * <br>
 * All pruning methods handle missing values
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
        REDUCED_ERROR,
        
        /**
         * Bottom-Up pessimistic pruning using Error based Pruning from the 
         * C4.5 algorithm. If the node visitor supports 
         * {@link TreeNodeVisitor#setPath(int, jsat.classifiers.trees.TreeNodeVisitor) }
         * it will perform sub tree replacement for the maximal sub tree. <br>
         * The default Confidence (CF) is 0.25, as used in the C4.5 algorithm.<br>
         * <br>
         * NOTE: For the one case where the root would be pruned by taking the sub tree
         * with the most nodes, this implementation will not perform that step. However,
         * this is incredibly rare - and otherwise performs the same.
         */
        ERROR_BASED
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
        //TODO add vargs for extra arguments that may be used by pruning methods
        if(method == PruningMethod.NONE )
            return;
        else if(method == PruningMethod.REDUCED_ERROR)
            pruneReduceError(null, -1, root, testSet);
        else if(method == PruningMethod.ERROR_BASED)
            pruneErrorBased(null, -1, root, testSet, 0.25);
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
            List<DataPointPair<Integer>> hadMissing = new ArrayList<DataPointPair<Integer>>(0);
            //TODO if splits = 2, reorder the original array and use subList to return memory efficent references
            for (int i = 0; i < numSplits; i++)
                splits.add(new ArrayList<DataPointPair<Integer>>());
            for (DataPointPair<Integer> dpp : testSet)
            {
                int path = current.getPath(dpp.getDataPoint());
                if(path >= 0)
                    splits.get(path).add(dpp);
                else//missing value
                    hadMissing.add(dpp);
            }

            if(!hadMissing.isEmpty())
                DecisionStump.distributMissing(splits, hadMissing);
            
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
    
    /**
     * 
     * @param parent the parent node, or null if there is no parent
     * @param pathFollowed the path from the parent node to the current node
     * @param current the current node to evaluate
     * @param testSet the set of points to estimate error from
     * @param alpha the Confidence 
     * @return expected upperbound on errors
     */
    private static double pruneErrorBased(TreeNodeVisitor parent, int pathFollowed, TreeNodeVisitor current, List<DataPointPair<Integer>> testSet, double alpha)
    {
        //TODO this does a lot of redundant computation. Re-write this code to keep track of where datapoints came from to avoid redudancy. 
        if(current == null || testSet.isEmpty())
            return 0;
        else if(current.isLeaf())//return number of errors
        {
            int errors = 0;
            double N = 0;
            for(DataPointPair<Integer> dpp : testSet)
            {
                if(current.localClassify(dpp.getDataPoint()).mostLikely() != dpp.getPair())
                    errors+=dpp.getDataPoint().getWeight();
                N+=dpp.getDataPoint().getWeight();
            }
            return computeBinomialUpperBound(N, alpha, errors);
        }
        List<List<DataPointPair<Integer>>> splitSet = new ArrayList<List<DataPointPair<Integer>>>(current.childrenCount());
        List<DataPointPair<Integer>> hadMissing = new ArrayList<DataPointPair<Integer>>(0);
        for(int i = 0; i < current.childrenCount(); i++)
            splitSet.add(new ArrayList<DataPointPair<Integer>>());
        
        int localErrors = 0;
        double subTreeScore = 0;
        
        double N = 0.0;
        for(DataPointPair<Integer> dpp : testSet)
        {
            DataPoint dp = dpp.getDataPoint();
            if(current.localClassify(dp).mostLikely() != dpp.getPair())
                localErrors+=dp.getWeight();
            N += dp.getWeight();
            
            int path = current.getPath(dp);
            if(path >= 0)
                splitSet.get(path).add(dpp);
            else
                hadMissing.add(dpp);
        }
        
        if(!hadMissing.isEmpty())
            DecisionStump.distributMissing(splitSet, hadMissing);
        
        //Find child wich gets the most of the test set as the candidate for sub-tree replacement
        int maxChildCount = 0;
        int maxChild = -1;
        for(int path = 0; path < splitSet.size(); path++)
            if(!current.isPathDisabled(path))
            {
                subTreeScore += pruneErrorBased(current, path, current.getChild(path), splitSet.get(path), alpha);
                
                if(maxChildCount < splitSet.get(path).size())
                {
                    maxChildCount = splitSet.get(path).size();
                    maxChild = path;
                }
            }

        /* Original uses normal approximation of p + Z_alpha * sqrt(p (1-p) / n).
         * Instead, just compute exact using inverse beta
         * Upper Bound = 1.0 - BetaInv(alpha, n-k, k+1)
         */
        final double prunedTreeScore = computeBinomialUpperBound(N, alpha, localErrors);

        double maxChildTreeScore;
        if(maxChild == -1)
            maxChildTreeScore = Double.POSITIVE_INFINITY;
        else
        {
            TreeNodeVisitor maxChildNode = current.getChild(maxChild);
            int otherE = 0;
            for (int path = 0; path < splitSet.size(); path++)
                    for (DataPointPair<Integer> dpp : splitSet.get(path))
                        if (maxChildNode.classify(dpp.getDataPoint()).mostLikely() != dpp.getPair())
                            otherE+=dpp.getDataPoint().getWeight();

            maxChildTreeScore = computeBinomialUpperBound(N, alpha, otherE);
        }
        
        
        if(maxChildTreeScore < prunedTreeScore && maxChildTreeScore < subTreeScore && parent != null)
        {
            try//NodeVisitor may not support setPath method, which is optional
            {
                parent.setPath(pathFollowed, current.getChild(maxChild));

                return maxChildTreeScore;
            }
            catch(UnsupportedOperationException ex)
            {
                //fall out to others, this is ok
            }
        }
        //MaxChildTreeScore is not the min, or it was not supported - so we do not compare against it any more
        if(prunedTreeScore < subTreeScore  )
        {
            for(int i = 0; i < current.childrenCount(); i++)
                current.disablePath(i);
            return prunedTreeScore;
        }
        else//no change
            return subTreeScore;
        
    }
    
    private static double computeBinomialUpperBound(final double N, double alpha, double errors)
    {
        return N * (1.0 - SpecialMath.invBetaIncReg(alpha, N - errors+1e-9, errors + 1.0));
    }
}
