package jsat.datatransform;

import java.util.List;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * This abstract class implements the Parameterized interface to ease the 
 * development of simple Data Transforms. If a more complicated set of 
 * parameters is needed then what is obtained from 
 * {@link Parameter#getParamsFromMethods(java.lang.Object) } than there is no 
 * reason to use this class. 
 * 
 * @author Edward Raff
 */
abstract public class DataTransformBase implements DataTransform, Parameterized
{

    @Override
    abstract public DataTransform clone();
    
}
