package jsat.datatransform;

import java.util.*;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.linear.*;

/**
 * This Data Transform allows the complete removal of specific features from the data set. 
 * @author Edward Raff
 */
public class RemoveAttributeTransform implements DataTransform
{
    /**
     * Categorical attributes to be removed
     */
    private Set<Integer> catRemove;
    /**
     * Numerical attributes to be removed
     */
    private Set<Integer> numRemove;

    /**
     * Creates a new transform for removing specified features from a data set
     * @param dataSet the data set that this transform is meant for
     * @param categoricalToRemove the set of categorical attributes to remove, in the rage of [0, {@link DataSet#getNumCategoricalVars() }). 
     * @param numericalToRemove the set of numerical attributes to remove, in the rage of [0, {@link DataSet#getNumNumericalVars() }). 
     */
    public RemoveAttributeTransform(DataSet dataSet, Set<Integer> categoricalToRemove, Set<Integer> numericalToRemove)
    {
        for(int i : categoricalToRemove)
            if (i >= dataSet.getNumCategoricalVars())
                throw new RuntimeException("The data set does not have a categorical value " + i + " to remove");
        for(int i : numericalToRemove)
            if (i >= dataSet.getNumNumericalVars())
                throw new RuntimeException("The data set does not have a numercal value " + i + " to remove");
        this.catRemove = new HashSet<Integer>(categoricalToRemove);
        this.numRemove = new HashSet<Integer>(numericalToRemove);
    }
    
    /**
     * Copy constructor
     * @param other the transform to copy
     */
    private RemoveAttributeTransform(RemoveAttributeTransform other)
    {
        this.catRemove = new HashSet<Integer>(other.catRemove);
        this.numRemove = new HashSet<Integer>(other.numRemove);
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        int[] catVals = dp.getCategoricalValues();
        Vec numVals = dp.getNumericalValues();

        CategoricalData[] newCatData = new CategoricalData[catVals.length - catRemove.size()];
        int[] newCatVals = new int[newCatData.length];
        int newVecSize = numVals.length() - numRemove.size();
        Vec newNumVals;
        if (numVals.isSparse())
            if (numVals instanceof SparseVector)
                newNumVals = new SparseVector(newVecSize, ((SparseVector) numVals).nnz());
            else
                newNumVals = new SparseVector(newVecSize);
        else
            newNumVals = new DenseVector(newVecSize);

        int k = 0;//K is the new index
        for (int i = 0; i < catVals.length; i++)//i is the old index
        {
            if (catRemove.contains(i))
                continue;
            newCatVals[k] = catVals[i];
            newCatData[k] = dp.getCategoricalData()[i].clone();
            k++;
        }

        k = 0;

        Iterator<IndexValue> iter = numVals.getNonZeroIterator();
        if (iter.hasNext())//if all values are zero, nothing to do
        {
            IndexValue curIV = iter.next();
            for (int i = 0; i < numVals.length(); i++)//i is the old index
            {
                if (numRemove.contains(i))
                    continue;

                k++;
                if (numVals.isSparse())//log(n) insert and loopups to avoid!
                {
                    if (curIV == null)
                        continue;
                    if (i > curIV.getIndex())//We skipped a value that existed
                        while (i > curIV.getIndex() && iter.hasNext())
                            curIV = iter.next();
                    if (i < curIV.getIndex())//Index is zero, nothing to set
                        continue;
                    else if (i == curIV.getIndex())
                    {
                        newNumVals.set(k - 1, curIV.getValue());
                        if (iter.hasNext())
                            curIV = iter.next();
                        else
                            curIV = null;
                    }
                }
                else//All dense, just set them all
                    newNumVals.set(k - 1, numVals.get(i));
            }
        }
        return new DataPoint(newNumVals, newCatVals, newCatData, dp.getWeight());
    }

    @Override
    public RemoveAttributeTransform clone()
    {
        return new RemoveAttributeTransform(this);
    }
    
    public static class RemoveAttributeTransformFactory implements DataTransformFactory
    {
        private Set<Integer> catToRemove;
        private Set<Integer> numerToRemove;

        public RemoveAttributeTransformFactory(Set<Integer> catToRemove, Set<Integer> numerToRemove)
        {
            this.catToRemove = catToRemove;
            this.numerToRemove = numerToRemove;
        }
        
        @Override
        public DataTransform getTransform(DataSet dataset)
        {
            return new RemoveAttributeTransform(dataset, catToRemove, numerToRemove);
        }
    }
    
}
