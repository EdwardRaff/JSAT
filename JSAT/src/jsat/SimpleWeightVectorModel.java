package jsat;

import jsat.linear.ConstantVector;
import jsat.linear.Vec;

/**
 * This interface is for multi-class classification problems where there may be 
 * <i>K</i> or <i>K-1</i> weight vectors for <i>K</i> classes. For regression 
 * problems it is treated as <i>K = 1</i> and there should be only one weight 
 * vector.
 * 
 * @author Edward Raff
 */
public interface SimpleWeightVectorModel
{
    /**
     * Returns the raw weight vector associated with the given class index. If 
     * the given class is an implicit zero vector, a {@link ConstantVector} 
     * object may be returned. <br>
     * Do not alter the returned weight vector, as it will change the model's 
     * values. <br>
     * <br>
     * If a regression problem, only {@code index = 0} should be used
     * 
     * @param index the class index to get the weight vector for
     * @return the weight vector used for the specified class
     */
    public Vec getRawWeight(int index);
    
    /**
     * Returns the bias term used with the weight vector for the given class 
     * index. If the model does not support or was not trained with  bias
     * weights, {@code 0} will be returned.<br>
     * <br>
     * If a regression problem, only {@code index = 0} should be used
     * 
     * @param index the class index to get the weight vector for
     * @return the bias term for the specified class
     */
    public double getBias(int index);
    
    /**
     * Returns the number of weight vectors that can be returned. For binary 
     * classification problems the value may be 1 if only a single weight 
     * vector's sign is used to determine the class. For multi-class problems, 
     * the weight vector count includes the implicit zero vector (if one is 
     * being used). 
     * @return the number of weight vectors for which 
     * {@link #getRawWeight(int) } can be called. 
     */
    public int numWeightsVecs();
}
