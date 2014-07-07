
package jsat.classifiers.neuralnetwork.regularizers;

import java.io.Serializable;
import java.util.concurrent.ExecutorService;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This interface defines the contract for applying a regularization scheme to 
 * the weight and bias values of a laying in a neural network. 
 * 
 * @author Edward Raff
 */
public interface WeightRegularizer extends Serializable
{
    /**
     * Applies regularization to one matrix, where the rows of the matrix 
     * correspond tot he weights associated to one neuron's input. The vector of
     * bias terms must then have the same length as the number of rows in the 
     * given matrix. 
     * @param W the matrix to apply regularization to
     * @param b the vector of bias terms to apply regularization to
     */
    public void applyRegularization(Matrix W, Vec b);
    
    /**
     * Applies regularization to one matrix, where the rows of the matrix 
     * correspond tot he weights associated to one neuron's input. The vector of
     * bias terms must then have the same length as the number of rows in the 
     * given matrix. 
     * @param W the matrix to apply regularization to
     * @param b the vector of bias terms to apply regularization to
     * @param ex the source of threads for parallel computation
     */
    public void applyRegularization(Matrix W, Vec b, ExecutorService ex);
    
    /**
     * Applies the regularization to one row of the weight matrix, where the row
     * corresponds to the weights into one neuron. 
     * 
     * @param w the weight row to be altered depending on the regularization method
     * @param b the original bias input to this row
     * @return the new bias value, or the same value if no change in the bias has occurred
     */
    public double applyRegularizationToRow(Vec w, double b);
    
    public WeightRegularizer clone();
}
