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
}
