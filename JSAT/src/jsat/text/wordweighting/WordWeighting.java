
package jsat.text.wordweighting;

import java.util.List;
import jsat.linear.Vec;
import jsat.math.IndexFunction;

/**
 * WordWeighting is an index function specifically mean for modifying the values 
 * of a vectors used for a bag-of-words representation of text data. <br>
 * Some Word weighting schemes may need information about the document 
 * collection as a whole before constructing the weightings, and this class 
 * provides the facilities for this to be done in a standardized manner. 
 * 
 * @author Edward Raff
 */
public abstract class WordWeighting extends IndexFunction
{


	private static final long serialVersionUID = 2372760149718829334L;

	/**
     * Prepares the word weighting to be performed on a data set. This should be
     * called once before being applied to any vectors. Different WordWeightings
     * may require different amounts of computation to set up. 
     * 
     * @param allDocuments the list of all vectors that make up the set of 
     * documents. The word vectors should be unmodified, containing the value of
     * how many times a word appeared in the document for each index. 
     * @param df a list mapping each integer index of a word to how many times 
     * that word occurred in total
     */
    abstract public void setWeight(List<? extends Vec> allDocuments, List<Integer> df);

    /**
     * The implementation may want to pre compute come values based on the 
     * vector it is about to be applied to. This should be called in place of
     * {@link Vec#applyIndexFunction(jsat.math.IndexFunction) } . The vector 
     * should be in a bag-of-words form where each index value indicates how 
     * many times the word for that index occurred in the document represented 
     * by the vector. 
     * 
     * @param vec the vector to set up for and then alter by invoking 
     * {@link Vec#applyIndexFunction(jsat.math.IndexFunction) } on
     */
    abstract public void applyTo(Vec vec);
}
