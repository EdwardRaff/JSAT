
package jsat.text.wordweighting;

import java.util.List;
import jsat.linear.SparceVector;
import jsat.linear.Vec;
import jsat.math.IndexFunction;

/**
 *
 * @author Edward Raff
 */
public abstract class WordWeighting extends IndexFunction
{

    /**
     * Prepares the word weighting to be performed on a data set. This should be called once before being applied to any vectors. 
     * @param toAlter the vector to apply weighting to 
     * @param totalDocuments
     * @param df 
     */
    abstract public void setWeight(int totalDocuments, List<Integer> df);

    /**
     * The user may want to pre compute come values based on the vector it is 
     * about to be applied to. This should be called once before 
     * {@link Vec#applyIndexFunction(jsat.math.IndexFunction) } is called. 
     * 
     * @param vec the vector to set up for and then call {@link Vec#applyIndexFunction(jsat.math.IndexFunction) } on
     */
    abstract public void applyTo(Vec vec);
}
