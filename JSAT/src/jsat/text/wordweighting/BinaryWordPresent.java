package jsat.text.wordweighting;

import java.util.List;
import jsat.linear.Vec;

/**
 * Provides a simple binary representation of bag-of-word vectors by simply 
 * marking a value 1.0 if the token is present, and 0.0 if the token is not 
 * present. Nothing else is taken into account.<br>
 * <br>
 * This class does not require any state or configuration, so it can be used
 * without calling {@link #setWeight(java.util.List, java.util.List) }.
 *
 * 
 * @author Edward Raff
 */
public class BinaryWordPresent extends WordWeighting
{

    private static final long serialVersionUID = 5633647387188363706L;

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
            return 1.0;
        else
            return 0.0;
    }
    
}
