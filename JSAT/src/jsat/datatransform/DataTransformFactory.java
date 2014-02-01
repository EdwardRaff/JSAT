package jsat.datatransform;

import jsat.DataSet;
import jsat.exceptions.FailedToFitException;

/**
 * It may be necessary to learn the same transform from several different 
 * versions of a data set, this interface facilitates that process. 
 * 
 * @author Edward Raff
 */
public interface DataTransformFactory
{
    /**
     * Creates a new transform that is inferred from the given data set
     * @param dataset the data set to learn the transform from
     * @return a new DataTransform that can be used
     * @throws FailedToFitException if the transform could not be constructed or
     * was inappropriate for the given data set. 
     */
    public DataTransform getTransform(DataSet dataset);
    
    public DataTransformFactory clone();
}
