
package jsat.datatransform;

import jsat.classifiers.DataPoint;

/**
 *
 * @author Edward Raff
 */
public interface DataTransform
{
    public DataPoint transform(DataPoint dp);
}
