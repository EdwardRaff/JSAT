package jsat.classifiers.neuralnetwork.initializers;

import java.io.Serializable;
import java.util.Random;
import jsat.linear.Vec;

/**
 * This interface specifies the method of initializing the bias connections in a
 * neural network. 
 * @author Edward Raff
 */
public interface BiastInitializer extends Serializable
{
    /**
     * Performs the initialization of the given vector of bias values
     * @param b the vector to store the biases in
     * @param fanIn the number of connections coming into the layer that these 
     * biases are for. 
     * @param rand the source of randomness for initialization 
     */
    public void init(Vec b, int fanIn, Random rand);
    
    public BiastInitializer clone();
}
