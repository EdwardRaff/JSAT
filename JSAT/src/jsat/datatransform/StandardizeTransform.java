
package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.Vec;

/**
 * This transform performs standardization of the data, which makes each column 
 * have a mean of zero and a variance of one. This assume the data comes from a 
 * normal distribution and scales it to the unit normal distribution. 
 * <br><br>
 * This transform is equivalent to applying {@link ZeroMeanTransform} followed 
 * by {@link UnitVarianceTransform}. 
 * 
 * @author Edward Raff
 */
public class StandardizeTransform implements InPlaceTransform
{

    private static final long serialVersionUID = -2349721113741805955L;
    private Vec means;
    private Vec stdDevs;

    /**
     * Creates a new object for Standardizing datasets
     */
    public StandardizeTransform()
    {
        
    }
    
    /**
     * Creates a new object for standaidizing datasets fit to the given dataset
     * @param dataset the dataset to learn how to standardize from
     */
    public StandardizeTransform(DataSet dataset)
    {
        fit(dataset);
    }

    @Override
    public void fit(DataSet dataset)
    {
        Vec[] vecs = dataset.getColumnMeanVariance();
        means = vecs[0];
        stdDevs = vecs[1];
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public StandardizeTransform(StandardizeTransform toCopy)
    {
        this.means = toCopy.means.clone();
        this.stdDevs = toCopy.stdDevs.clone();
    }
    
    @Override
    public DataPoint transform(DataPoint dp)
    {
        DataPoint newDP = dp.clone();
        mutableTransform(newDP);
        return newDP;
    }
    
    @Override
    public void mutableTransform(DataPoint dp)
    {
        Vec toAlter = dp.getNumericalValues();
        toAlter.mutableSubtract(means);
        toAlter.mutablePairwiseDivide(stdDevs);
    }

    @Override
    public boolean mutatesNominal()
    {
        return false;
    }

    @Override
    public StandardizeTransform clone()
    {
        return new StandardizeTransform(this);
    }
    
}
