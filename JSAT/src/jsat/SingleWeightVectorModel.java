package jsat;

import jsat.linear.Vec;

/**
 * This interface is for binary classification and regression problems where the
 * solution can be represented as a single weight vector. 
 * 
 * @author Edward Raff
 */
public interface SingleWeightVectorModel extends SimpleWeightVectorModel
{
    /**
     * Returns the only weight vector used for the model
     * @return the only weight vector used for the model
     */
    public Vec getRawWeight();
    
    /**
     * Returns the bias term used for the model, or 0 of the model does not 
     * support or was not trained with a bias term. 
     * 
     * @return the bias term for the model
     */
    public double getBias();
    
    @Override
    default public Vec getRawWeight(int index)
    {
	if(index == 0)
	    return getRawWeight();
	else 
	    throw new IndexOutOfBoundsException("SingleWeightVectorModel has only a single weight vector at index 0, index " + index + " is not valid");
    }
    
    @Override
    default public double getBias(int index)
    {
	if(index == 0)
	    return getBias();
	else 
	    throw new IndexOutOfBoundsException("SingleWeightVectorModel has only a single weight vector at index 0, index " + index + " is not valid");
    }
    
    @Override
    default public int numWeightsVecs()
    {
	return 1;
    }
    
    public SingleWeightVectorModel clone();
}
