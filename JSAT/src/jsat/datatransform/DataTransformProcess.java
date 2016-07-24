package jsat.datatransform;

import java.util.ArrayList;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;

/**
 * Performing a transform on the whole data set before training a classifier can
 * add bias to the results. For proper evaluation, the transforms must be 
 * learned from the training set and not contain any knowledge from the testing 
 * set. A DataTransformProcess aids in this by providing a mechanism to contain 
 * several different transforms to learn and then apply. 
 * <br><br>
 * The Parameters of the Data Transform Process are the parameters from the
 * individual transform factories that make up the whole process. The name 
 * "DataTransformProcess" will not be prefixed to the parameter names. 
 * 
 * @author Edward Raff
 */
public class DataTransformProcess implements DataTransform, Parameterized
{

    private static final long serialVersionUID = -2844495690944305885L;
    @ParameterHolder(skipSelfNamePrefix = true)
    private List<DataTransform> transformSource;
    private List<DataTransform> learnedTransforms;


    /**
     * Creates a new transform process that is empty. Transform factories must 
     * be added using 
     * {@link #addTransform(jsat.datatransform.DataTransformFactory) }.
     */
    public DataTransformProcess()
    {
        transformSource = new ArrayList<DataTransform>();
        learnedTransforms = new ArrayList<DataTransform>();   
    }
    
    /**
     * Creates a new transform process from the listed factories, which will be
     * applied in order by index. 
     * 
     * @param transforms the array of factories to apply as the data transform process
     */
    public DataTransformProcess(DataTransform... transforms)
    {
        this();
        for(DataTransform dt : transforms)
            this.addTransform(dt);
    }
    
    /**
     * Adds a transform to the list of transforms. Transforms are learned and 
     * applied in the order in which they are added. 
     * @param transform the factory for the transform to add
     */
    public void addTransform(DataTransform transform)
    {
        transformSource.add(transform);
    }
    
    /**
     * 
     * @return the number of transforms currently chained in this transform 
     * process
     */
    public int getNumberOfTransforms()
    {
        return transformSource.size();
    }
    
    /**
     * Consolidates transformation objects when possible. Currently only works with {@link RemoveAttributeTransform}
     */
    private void consolidateTransforms()
    {
        for(int i = 0; i < learnedTransforms.size()-1; i++)
        {
            DataTransform t1 = learnedTransforms.get(i);
            DataTransform t2 = learnedTransforms.get(i+1);
            if(!(t1 instanceof RemoveAttributeTransform && t2 instanceof RemoveAttributeTransform))
                continue;//They are not both RATs
            RemoveAttributeTransform r1 = (RemoveAttributeTransform) t1;
            RemoveAttributeTransform r2 = (RemoveAttributeTransform) t2;
            
            r2.consolidate(r1);
            learnedTransforms.remove(i);
            i--;
        }
    }

    @Override
    public void fit(DataSet data)
    {
        learnApplyTransforms(data);
    }
    
    /**
     * Learns the transforms for the given data set. The data set will not be 
     * altered. Once finished, <tt>this</tt> DataTransformProcess can be applied
     * to the dataSet to get the transformed data set. 
     * 
     * @param dataSet the data set to learn a series of transforms from
     */
    public void leanTransforms(DataSet dataSet)
    {
        learnApplyTransforms(dataSet.shallowClone());
    }
    
    /**
     * Learns the transforms for the given data set. The data set is then 
     * altered after each transform is learned so the next transform can be 
     * learned as well. <br> The results are equivalent to calling 
     * {@link #learnApplyTransforms(jsat.DataSet) } on the data set and then 
     * calling {@link DataSet#applyTransform(jsat.datatransform.DataTransform) }
     * with this DataTransformProces. 
     * 
     * @param dataSet the data set to learn a series of transforms from and 
     * alter into the final transformed form
     */
    public void learnApplyTransforms(DataSet dataSet)
    {
        learnedTransforms.clear();
        //used to keep track if we can start using in place transforms
        boolean vecSafe = false;
        boolean catSafe = false;
        int iter = 0;
        
        //copy original references so we can check saftey of inplace mutation later
        Vec[] origVecs = new Vec[dataSet.getSampleSize()];
        int[][] origCats = new int[dataSet.getSampleSize()][];
        for (int i = 0; i < origVecs.length; i++)
        {
            DataPoint dp = dataSet.getDataPoint(i);
            origVecs[i] = dp.getNumericalValues();
            origCats[i] = dp.getCategoricalValues();
        }

        for (DataTransform dtf : transformSource)
        {
            DataTransform transform = dtf.clone();
            transform.fit(dataSet);
            if(transform instanceof InPlaceTransform)
            {
                InPlaceTransform ipt = (InPlaceTransform) transform;
                //check if it is safe to apply mutations
                if(iter > 0 && !vecSafe || (ipt.mutatesNominal() && !catSafe))
                {
                    boolean vecClear = true, catClear = true;
                    for (int i = 0; i < origVecs.length && (vecClear || catClear); i++)
                    {
                        DataPoint dp = dataSet.getDataPoint(i);
                        vecClear = origVecs[i] != dp.getNumericalValues();
                        catClear = origCats[i] != dp.getCategoricalValues();
                    }
                    
                    vecSafe = vecClear;
                    catSafe = catClear;
                }
                
                //Now we know if we can apply the mutations or not
                if(vecSafe && (!ipt.mutatesNominal() || catSafe))
                    dataSet.applyTransform(ipt, true);
                else//go back to normal
                    dataSet.applyTransform(transform);
            }
            else
                dataSet.applyTransform(transform);
            
            learnedTransforms.add(transform);
            iter++;
        }
        consolidateTransforms();
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        final Vec origNum = dp.getNumericalValues();
        final int[] origCat = dp.getCategoricalValues();
        for(DataTransform dt : learnedTransforms)
        {
            if(dt instanceof InPlaceTransform)
            {
                InPlaceTransform it = (InPlaceTransform) dt;
                //check if we can safley mutableTransform instead of allocate
                if(origNum != dp.getNumericalValues() && (!it.mutatesNominal() || origCat != dp.getCategoricalValues()))
                {
                    it.mutableTransform(dp);
                    continue;
                }   
            }
            dp = dt.transform(dp);
        }
        return dp;
    }

    @Override
    public DataTransformProcess clone()
    {
        DataTransformProcess clone = new DataTransformProcess();
        
        for(DataTransform dtf : this.transformSource)
            clone.transformSource.add(dtf.clone());
        
        for(DataTransform dt : this.learnedTransforms)
            clone.learnedTransforms.add(dt.clone());
        
        return clone;
    }

    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
    
}
