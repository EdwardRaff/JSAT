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
    /*
     * Each index map maps the old indecies in the original data set to their 
     * new positions. The value in the array is old index, the index of the 
     * value is the index it would be when the attributes were removed. 
     * This means each is in sorted order, and is of the size of the resulting 
     * feature space
     */
    private int[] catIndexMap;
    private int[] numIndexMap;

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
        
        catIndexMap = new int[dataSet.getNumCategoricalVars()-categoricalToRemove.size()];
        numIndexMap = new int[dataSet.getNumNumericalVars()-numericalToRemove.size()];
        int k = 0;
        for(int i = 0; i < dataSet.getNumCategoricalVars(); i++)
        {
            if(categoricalToRemove.contains(i))
                continue;
            numIndexMap[k++] = i;
        }
        k = 0;
        for(int i = 0; i < dataSet.getNumNumericalVars(); i++)
        {
            if(numericalToRemove.contains(i))
                continue;
            numIndexMap[k++] = i;
        }
        
    }
    
    /**
     * Copy constructor
     * @param other the transform to copy
     */
    private RemoveAttributeTransform(RemoveAttributeTransform other)
    {
        this.catIndexMap = Arrays.copyOf(other.catIndexMap, other.catIndexMap.length);
        this.numIndexMap = Arrays.copyOf(other.numIndexMap, other.numIndexMap.length);
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        int[] catVals = dp.getCategoricalValues();
        Vec numVals = dp.getNumericalValues();

        CategoricalData[] newCatData = new CategoricalData[catIndexMap.length];
        int[] newCatVals = new int[newCatData.length];
        Vec newNumVals;
        if (numVals.isSparse())
            if (numVals instanceof SparseVector)
                newNumVals = new SparseVector(numIndexMap.length, ((SparseVector) numVals).nnz());
            else
                newNumVals = new SparseVector(numIndexMap.length);
        else
            newNumVals = new DenseVector(numIndexMap.length);

        for(int i = 0; i < catIndexMap.length; i++)
            newCatVals[i] = catVals[catIndexMap[i]];

        int k = 0;

        Iterator<IndexValue> iter = numVals.getNonZeroIterator();
        if (iter.hasNext())//if all values are zero, nothing to do
        {
            IndexValue curIV = iter.next();
            for (int i = 0; i < numIndexMap.length; i++)//i is the old index
            {
                if (numVals.isSparse())//log(n) insert and loopups to avoid!
                {
                    if (curIV == null)
                        continue;
                    if (numIndexMap[i] > curIV.getIndex())//We skipped a value that existed
                        while (numIndexMap[i] > curIV.getIndex() && iter.hasNext())
                            curIV = iter.next();
                    if (numIndexMap[i] < curIV.getIndex())//Index is zero, nothing to set
                        continue;
                    else if (numIndexMap[i] == curIV.getIndex())
                    {
                        newNumVals.set(i, curIV.getValue());
                        if (iter.hasNext())
                            curIV = iter.next();
                        else
                            curIV = null;
                    }
                }
                else//All dense, just set them all
                    newNumVals.set(i, numVals.get(numIndexMap[i]));
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
