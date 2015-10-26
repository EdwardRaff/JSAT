package jsat.text.wordweighting;

import java.util.List;
import jsat.linear.Vec;

/**
 * Provides a simple representation of bag-of-word vectors by simply using the 
 * number of occurrences for a word in a document as the weight for said word.
 * <br>
 * <br>
 * WordCount needs no initialization, and can be applied as soon as the object
 * is created. 
 * 
 * @author Edward Raff
 */
public class WordCount extends WordWeighting
{


	private static final long serialVersionUID = 4665749166722300326L;

	@Override
    public void setWeight(final List<? extends Vec> allDocuments, final List<Integer> df)
    {
        //No work needed
    }

    @Override
    public void applyTo(final Vec vec)
    {
        vec.applyIndexFunction(this);
    }

    @Override
    public double indexFunc(final double value, final int index)
    {
        if(index < 0) {
          return 0.0;
        } else if(value > 0.0) {
          return value;
        } else {
          return 0.0;
        }
    }
    
}
