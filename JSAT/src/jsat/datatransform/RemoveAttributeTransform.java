package jsat.datatransform;

import java.util.*;

import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.*;
import jsat.utils.IntList;

/**
 * This Data Transform allows the complete removal of specific features from the
 * data set. 
 * 
 * @author Edward Raff
 */
public class RemoveAttributeTransform implements DataTransform
{   

    private static final long serialVersionUID = 2803223213862922734L;
    /*
     * Each index map maps the old indecies in the original data set to their 
     * new positions. The value in the array is old index, the index of the 
     * value is the index it would be when the attributes were removed. 
     * This means each is in sorted order, and is of the size of the resulting 
     * feature space
     */
    protected int[] catIndexMap;
    protected int[] numIndexMap;
    
    private Set<Integer> categoricalToRemove;
    private Set<Integer> numericalToRemove;
    
    /**
     * Empty constructor that may be used by extending classes. Transforms that 
     * extend this will need to call 
     * {@link #setUp(jsat.DataSet, java.util.Set, java.util.Set)  } once the 
     * attributes to remove have been selected
     */
    protected RemoveAttributeTransform()
    {
        
    }
    
    /**
     * Creates a new transform for removing specified features from a data set.
     * Needs to still call {@link #fit(jsat.DataSet) } before ready to be used.
     *
     * @param categoricalToRemove the set of categorical attributes to remove,
     * in the rage of [0, {@link DataSet#getNumCategoricalVars() }).
     * @param numericalToRemove the set of numerical attributes to remove, in
     * the rage of [0, {@link DataSet#getNumNumericalVars() }).
     */
    public RemoveAttributeTransform(Set<Integer> categoricalToRemove, Set<Integer> numericalToRemove)
    {
        this.categoricalToRemove =  categoricalToRemove;
        this.numericalToRemove = numericalToRemove;
        
    }
    /**
     * Creates a new transform for removing specified features from a data set
     * @param dataSet the data set that this transform is meant for
     * @param categoricalToRemove the set of categorical attributes to remove, in the rage of [0, {@link DataSet#getNumCategoricalVars() }). 
     * @param numericalToRemove the set of numerical attributes to remove, in the rage of [0, {@link DataSet#getNumNumericalVars() }). 
     */
    public RemoveAttributeTransform(DataSet dataSet, Set<Integer> categoricalToRemove, Set<Integer> numericalToRemove)
    {
        this.categoricalToRemove =  categoricalToRemove;
        this.numericalToRemove = numericalToRemove;
        setUp(dataSet, categoricalToRemove, numericalToRemove);
    }
    
    /**
     * Returns an unmodifiable list of the original indices of the numeric 
     * attributes that will be kept when this transform is applied. 
     * @return the numeric indices that are not removed by this transform
     */
    public List<Integer> getKeptNumeric()
    {
        return IntList.unmodifiableView(numIndexMap, numIndexMap.length);
    }
    
    /**
     * Returns a mapping from the numeric indices in the transformed space back 
     * to their original indices 
     * 
     * @return a mapping from the transformed numeric space to the original one
     */
    public Map<Integer, Integer> getReverseNumericMap()
    {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int newIndex = 0; newIndex < numIndexMap.length; newIndex++)
            map.put(newIndex, numIndexMap[newIndex]);
        return map;
    }
    
    /**
     * Returns an unmodifiable list of the original indices of the nominal 
     * attributes that will be kept when this transform is applied. 
     * @return the nominal indices that are not removed by this transform
     */
    public List<Integer> getKeptNominal()
    {
        return IntList.unmodifiableView(catIndexMap, catIndexMap.length);
    }
    
    /**
     * Returns a mapping from the nominal indices in the transformed space back 
     * to their original indices 
     * 
     * @return a mapping from the transformed nominal space to the original one
     */
    public Map<Integer, Integer> getReverseNominalMap()
    {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int newIndex = 0; newIndex < catIndexMap.length; newIndex++)
            map.put(newIndex, catIndexMap[newIndex]);
        return map;
    }
    
    @Override
    public void fit(DataSet data)
    {
        if (categoricalToRemove != null && numericalToRemove != null)
            setUp(data, categoricalToRemove, numericalToRemove);
    }
    
    /**
     * Sets up the Remove Attribute Transform properly
     * 
     * @param dataSet the data set to remove the attributes from
     * @param categoricalToRemove the categorical attributes to remove
     * @param numericalToRemove the numeric attributes to remove
     */
    protected final void setUp(DataSet dataSet, Set<Integer> categoricalToRemove, Set<Integer> numericalToRemove)
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
            catIndexMap[k++] = i;
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
    protected RemoveAttributeTransform(RemoveAttributeTransform other)
    {
        if(other.catIndexMap != null)
            this.catIndexMap = Arrays.copyOf(other.catIndexMap, other.catIndexMap.length);
        if(other.numIndexMap != null)
            this.numIndexMap = Arrays.copyOf(other.numIndexMap, other.numIndexMap.length);
    }
    
    /**
     * A serious of Remove Attribute Transforms may be learned and applied 
     * sequentially to a single data set. Instead of keeping all the transforms 
     * around indefinitely, a sequential series of Remove Attribute Transforms 
     * can be consolidated into a single transform object. <br>
     * This method mutates the this transform by providing it with the 
     * transform that would have been applied before this current object. Once
     * complete, this transform can be used two perform both removals in one 
     * step.<br><br>
     * Example: <br> 
     * An initial set of features <i>A</i> is transformed into <i>A'</i> by 
     * transform t<sub>1</sub><br>
     * <i>A'</i> is transformed into <i>A''</i> by transform t<sub>2</sub><br>
     * Instead, you can invoke t<sub>2</sub>.consolidate(t<sub>1</sub>). 
     * You can then transform <i>A</i> into <i>A''</i> by using only transform 
     * t<sub>2</sub>
     * 
     * 
     * @param preceding the DataTransform that immediately precedes this one in 
     * a sequential list of transforms
     */
    public void consolidate(RemoveAttributeTransform preceding)
    {
        for(int i = 0; i < catIndexMap.length; i++)
            catIndexMap[i] = preceding.catIndexMap[catIndexMap[i]];
        for(int i = 0; i < numIndexMap.length; i++)
            numIndexMap[i] = preceding.numIndexMap[numIndexMap[i]];
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
}
