package jsat.text.wordweighting;

import java.util.List;
import jsat.linear.Vec;

/**
 * Provides a simple binary representation of bag-of-word vectors by simply 
 * marking a value 1.0 if the token is present, and 0.0 if the token is not 
 * present. Nothing else is taken into account. 
 * 
 * @author Edward Raff
 */
public class BinaryWordPresent extends WordWeighting
{


	private static final long serialVersionUID = 5633647387188363706L;

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
          return 1.0;
        } else {
          return 0.0;
        }
    }
    
}
