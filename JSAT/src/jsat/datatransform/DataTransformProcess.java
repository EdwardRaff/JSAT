package jsat.datatransform;

import java.util.ArrayList;
import java.util.List;
import jsat.DataSet;
import jsat.classifiers.DataPoint;

/**
 * Performing a transform on the whole data set before training a classifier can
 * add bias to the results. For proper evaluation, the transforms must be 
 * learned from the training set and not contain any knowledge from the testing 
 * set. A DataTransformProcess aids in this by providing a mechanism to contain 
 * several different transforms to learn and then apply. 
 * 
 * @author Edward Raff
 */
public class DataTransformProcess implements DataTransform
{
    private List<DataTransformFactory> transformSource;
    private List<DataTransform> learnedTransforms;
    

    public DataTransformProcess()
    {
        transformSource = new ArrayList<DataTransformFactory>();
        learnedTransforms = new ArrayList<DataTransform>();   
    }
    
    public void addTransform(DataTransformFactory transform)
    {
        transformSource.add(transform);
    }
    
    public void leanTransforms(DataSet dataSet)
    {
        learnedTransforms.clear();
        for(DataTransformFactory dtf : transformSource)
            learnedTransforms.add(dtf.getTransform(dataSet));
    }

    @Override
    public DataPoint transform(DataPoint dp)
    {
        for(DataTransform dt : learnedTransforms)
            dp = dt.transform(dp);
        return dp;
    }

    @Override
    public DataTransform clone()
    {
        DataTransformProcess clone = new DataTransformProcess();
        
        for(DataTransformFactory dtf : this.transformSource)
            clone.transformSource.add(dtf);
        
        for(DataTransform dt : this.learnedTransforms)
            clone.learnedTransforms.add(dt.clone());
        
        return clone;
    }
    
}
