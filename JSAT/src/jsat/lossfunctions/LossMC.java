package jsat.lossfunctions;

import jsat.classifiers.CategoricalResults;
import jsat.linear.Vec;

/**
 * Specifies a loss function for multi-class problems. A multi-class loss 
 * function must support a raw vector of scores for each class, where positive 
 * values indicate preference for the class associated with the same index. <br>
 * <br>
 * Calling {@link #process(jsat.linear.Vec, jsat.linear.Vec) } on the raw
 * scores is a mandatory first step, and will transform the raw scores into a 
 * usable form for the loss function. <br>
 * <br>
 * @author Edward Raff
 */
public interface LossMC extends LossC
{
    /**
     * Computes the scalar loss for on the given example
     * @param processed the vector of raw predictions. 
     * @param y the true class label in [0, k-1] for <i>k</i> classes
     * @return the loss in [0, Inf)
     */
    public double getLoss(Vec processed, int y);
    
    /**
     * Given the vector of raw outputs for each class, transform it into a new 
     * vector. 
     * <br>
     * {@code processed} and {@code derivs} may be the same object, and will 
     * simply have all its values altered if so. 
     * @param pred the vector of raw predictions 
     * @param processed the location to store the processed predictions. 
     */
    public void process(Vec pred, Vec processed);
    
    /**
     * Computes the derivatives with respect to each output
     * <br>
     * {@code processed} and {@code derivs} may be the same object, and will 
     * simply have all its values altered if so. 
     * @param processed the processed predictions
     * @param derivs the vector to place the derivative of the loss to.
     * @param y the true class label in [0, k-1] for <i>k</i> classes
     */
    public void deriv(Vec processed, Vec derivs, int y);
    
    /**
     * Given the {@link #process(jsat.linear.Vec, jsat.linear.Vec) processed} 
     * predictions, returns the classification results for said predictions. 
     * @param processed the processed score/prediction vector
     * @return the classification results
     */
    public CategoricalResults getClassification(Vec processed);
}
