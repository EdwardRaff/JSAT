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

    @Override
    public void setWeight(List<? extends Vec> allDocuments, List<Integer> df)
    {
        //No work needed
    }

    @Override
    public void applyTo(Vec vec)
    {
        vec.applyIndexFunction(this);
    }

    @Override
    public double indexFunc(double value, int index)
    {
        if(index < 0)
            return 0.0;
        else if(value > 0.0)
            return value;
        else
            return 0.0;
    }
    
}
