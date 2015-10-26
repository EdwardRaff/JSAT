package jsat.classifiers.trees;

import java.io.Serializable;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;

/**
 * Provides an abstracted mechanism for traversing and predicting from nodes in 
 * a tree meant for a supervised learning problem. <i>Paths</i> and <i>children
 * </i> are used interchangeably, every node has one path to one child<br> 
 * Paths to children nodes can be disabled, but not removed. This is done so 
 * that the implementation does not have to worry about changes in the indices 
 * to children nodes, which would be complicated to implement. Once a path is
 * disabled, it can not be re-enabled.<br>
 * If all paths to any children have been disabled, {@link #childrenCount() } 
 * may choose to return 0, otherwise - it must return the original number of 
 * paths to children nodes. 
 * 
 * The implementation for a given tree should override 
 * {@link #localClassify(jsat.classifiers.DataPoint) } and 
 * {@link #localRegress(jsat.classifiers.DataPoint) } if the operations are 
 * supported. 
 * 
 * @author Edward Raff
 * @see TreeLearner
 * @see TreePruner
 */
public abstract class TreeNodeVisitor implements Serializable, Cloneable
{

	private static final long serialVersionUID = 4026847401940409114L;

	/**
     * Returns the number of children this node of the tree has, and may return 
     * a non zero value even if the node is a leaf
     * @return the number of children this node has
     */
    abstract public int childrenCount();
    
    /**
     * Returns true if the node is a leaf, meaning it has no valid paths to any 
     * children
     * 
     * @return <tt>true</tt> if the node is purely a leaf node
     */
    abstract public boolean isLeaf();
    
    /**
     * Returns the node for the specific child, or null if the child index was 
     * not valid
     * @param child the index of the child node to obtain
     * @return the node for the child
     */
    abstract public TreeNodeVisitor getChild(int child);
    
    /**
     * Disables the selected path to the specified child from the current node. 
     * All child indices will not be effected by this operation.
     * 
     * @param child the index of the child to disable the path too
     */
    abstract public void disablePath(int child);
    
    /**
     * Optional operation!<br>
     * This method, if supported, will set the path so that the child is set to the given value. 
     * <br>
     * The implementation may choose to throw an exception if the NodeVisitor is not of the same
     * implementing class. 
     * @param child the child path
     * @param node the node to make the child
     */
    public void setPath(final int child, final TreeNodeVisitor node)
    {
        throw new UnsupportedOperationException("setPath is an optional operation.");
    }
    
    /**
     * Returns true if the path to the specified child is disabled, meaning it 
     * can not be traveled to. It will also return true for an invalid child 
     * path, since a non existent node can not be reached.
     * 
     * @param child the child index to check the path for
     * @return <tt>true</tt> if the path is unavailable, <tt>false</tt> if the 
     * path is good. 
     */
    abstract public boolean isPathDisabled(int child);
    
    /**
     * Returns the classification result that would have been obtained if the 
     * current node was a leaf node.
     * 
     * @param dp the data point to localClassify
     * @return the classification result
     * @throws UnsupportedOperationException if the tree node does not support 
     * or was not trained for classification
     */
    public CategoricalResults localClassify(final DataPoint dp)
    {
        throw new UnsupportedOperationException("This tree does not support classification");
    }
    
    /**
     * Returns the path down the tree the given data point would have taken, or
     * -1 if this node was a leaf node
     * @param dp the data point to send down the tree
     * @return the path that would be taken
     */
    abstract public int getPath(DataPoint dp);
    
    public CategoricalResults classify(final DataPoint dp)
    {
        TreeNodeVisitor node = this;
        while(!node.isLeaf())
        {
            final int path = node.getPath(dp);
            if(node.isPathDisabled(path)) {
              break;
            }
            node = node.getChild(path);
        }
        return node.localClassify(dp);
    }
    
    /**
     * Returns the regression result that would have been obtained if the 
     * current node was a leaf node.
     * 
     * @param dp the data point to regress
     * @return the classification result
     * @throws UnsupportedOperationException if the tree node does not support 
     * or was not trained for classification
     */
    public double localRegress(final DataPoint dp)
    {
        throw new UnsupportedOperationException("This tree does not support classification");
    }
    
    /**
     * Performs regression on the given data point by following it down the tree
     * until it finds the correct terminal node.
     * 
     * @param dp the data point to regress
     * @return the regression result from the tree starting from the current node
     */
    public double regress(final DataPoint dp)
    {
        TreeNodeVisitor node = this;
        while(!node.isLeaf())
        {
            final int path = node.getPath(dp);
            if(node.isPathDisabled(path)) {
              break;
            }
            node = node.getChild(path);
        }
        return node.localRegress(dp);
    }
    
    @Override
    abstract public TreeNodeVisitor clone();
}
