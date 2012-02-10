package jsat.datatransform;

import java.util.HashSet;
import java.util.Set;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

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
    
    

    public DataPoint transform(DataPoint dp)
    {   
        int[] catVals = dp.getCategoricalValues();
        Vec numVals = dp.getNumericalValues();
        
        CategoricalData[] newCatData = new CategoricalData[catVals.length-catRemove.size()];
        int[] newCatVals = new int[newCatData.length];
        Vec newNumVals = new DenseVector(numVals.length()-numRemove.size());
        
        int k = 0;//K is the new index
        for(int i = 0; i < catVals.length; i++)//i is the old index
        {
            if(catRemove.contains(i))
                continue;
            newCatVals[k] = catVals[i];
            newCatData[k] = dp.getCategoricalData()[i].clone();
            k++;
        }
        
        k = 0;
        for(int i = 0; i < numVals.length(); i++)//i is the old index
        {
            if(numRemove.contains(i))
                continue;
            newNumVals.set(k, numVals.get(i));
            k++;
        }
        
        return new DataPoint(newNumVals, newCatVals, newCatData, dp.getWeight());
    }
    
}
