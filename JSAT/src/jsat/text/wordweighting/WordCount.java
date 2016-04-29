package jsat.text.wordweighting;

import java.util.List;
import jsat.linear.Vec;

/**
 * Provides a simple representation of bag-of-word vectors by simply using the
 * number of occurrences for a word in a document as the weight for said word.
 * <br><br>
 * This class does not require any state or configuration, so it can be used
 * without calling {@link #setWeight(java.util.List, java.util.List) }.
 * 
 * @author Edward Raff
 */
public class WordCount extends WordWeighting
{
    private static final long serialVersionUID = 4665749166722300326L;

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
