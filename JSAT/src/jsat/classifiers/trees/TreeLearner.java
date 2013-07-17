package jsat.classifiers.trees;

import java.io.Serializable;

/**
 * This interface provides a contract that allows for the mutation and pruning 
 * of a tree using the {@link TreeNodeVisitor} and related classes. 
 * 
 * @author Edward Raff
 * @see TreeNodeVisitor
 * @see TreePruner
 */
public interface TreeLearner extends Serializable
{
    /**
     * Obtains a node visitor for the tree learner that can be used to traverse 
     * and predict from the learned tree
     * @return the root node visitor for the learned tree
     */
    public TreeNodeVisitor getTreeNodeVisitor();
}
