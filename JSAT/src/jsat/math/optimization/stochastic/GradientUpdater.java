package jsat.math.optimization.stochastic;

import java.io.Serializable;
import jsat.linear.Vec;

/**
 * This interface defines the method of updating some weight vector using a 
 * gradient and a learning rate. The method may then apply its own set of 
 * learning rates on top of the given learning rate in order to accelerate 
 * convergence in general or for specific conditions / methods. 
 * 
 * @author Edward Raff
 */
public interface GradientUpdater extends Serializable
{
    /**
     * Updates the weight vector {@code x} such that <i> x = x-&eta;f(grad)</i>,
     * where f(grad) is some function on the gradient that effectively returns a
     * new vector. It is not necessary for the internal implementation to ever 
     * explicitly form any of these objects, so long as {@code x} is mutated to 
     * have the correct result. 
     * @param w the vector to mutate such that is has been updated by the 
     * gradient
     * @param grad the gradient to update the weight vector {@code x} from
     * @param eta the learning rate to apply
     */
    public void update(Vec w, Vec grad, double eta);
    
    /**
     * Updates the weight vector {@code x} such that <i> x = x-&eta;f(grad)</i>,
     * where f(grad) is some function on the gradient that effectively returns a
     * new vector. It is not necessary for the internal implementation to ever 
     * explicitly form any of these objects, so long as {@code x} is mutated to 
     * have the correct result. <br>
     * <br>
     * This version of the update method includes two extra parameters to make 
     * it easer to use when a scalar bias term is also used
     * 
     * @param w the vector to mutate such that is has been updated by the 
     * gradient
     * @param grad the gradient to update the weight vector {@code x} from
     * @param eta the learning rate to apply
     * @param bias the bias term of the vector
     * @param biasGrad the gradient for the bias term
     * @return the value to change the bias by, the update being 
     * {@code bias = bias - returnValue}
     */
    public double update(Vec w, Vec grad, double eta, double bias, double biasGrad);
    
    /**
     * Sets up this updater to update a weight vector of dimension {@code d} 
     * by a gradient of the same dimension 
     * @param d the dimension of the weight vector that will be updated
     */
    public void setup(int d);
    
    public GradientUpdater clone();
}
