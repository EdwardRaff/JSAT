
package jsat.classifiers.neuralnetwork.initializers;

import java.io.Serializable;
import java.util.Random;
import jsat.linear.Matrix;

/**
 * This interface specifies the method of initializing the weight connections in
 * a neural network. 
 * 
 * @author Edward Raff
 */
public interface WeightInitializer extends Serializable
{
    /**
     * Initializes the values of the given weight matrix
     * @param w the matrix to initialize
     * @param rand the source of randomness for the initialization 
     */
    public void init(Matrix w, Random rand);
    
    public WeightInitializer clone();
}
