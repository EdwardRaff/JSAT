package jsat.classifiers.neuralnetwork.activations;

import java.io.Serializable;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * This interface defines a type of activation layer for use in a Neural Network
 * @author Edward Raff
 */
public interface ActivationLayer extends Serializable
{
    /**
     * Computes the activation function of this layer on the given input. 
     * @param input the raw input to compute the activation for
     * @param output the location to store the activation in
     */
    public void activate(Vec input, Vec output);
    
    /**
     * Computes the activation function of this layer on the given input. 
     * @param input the raw input to compute the activation for
     * @param output the location to store the activation in
     * @param rowMajor {@code true} if the information per input is stored in 
     * rows, {@code false} if the inputs were stored by column. This parameter 
     * does not indicate if the matrices themselves are backed by a row or 
     * column major implementation
     */
    public void activate(Matrix input, Matrix output, boolean rowMajor);
    
    /**
     * This method computes the backpropagated error to a given layer. Often 
     * denoted as &delta;<sup>l</sup> = w<sup><small>l+1</small> <b>T</b></sup> 
     * &delta;<sup>l+1</sup> &otimes; &part; f(x<sup>l</sup>), where &part; is 
     * the Hadamard product and &part; f(x<sup>l</sup>) is the derivative of 
     * this activation function on the input that was feed into this activation. 
     * <br>
     * {@code delta_partial} and {@code errout} may point to the same vector 
     * object
     * @param input the input to this layer that was feed in to be activated
     * @param output the activation that was produced for this layer
     * @param delta_partial the error assigned to this layer from the above
     * layer, sans the hamard product with the derivative of the layer 
     * activation. Often denoted as w<sup><small>l+1</small> <b>T</b></sup> &delta;<sup>l+1</sup>
     * @param errout the delta value or error produced for this layer
     */
    public void backprop(Vec input, Vec output, Vec delta_partial, Vec errout);
    
    /**
     * This method computes the backpropagated error to a given layer. Often 
     * denoted as &delta;<sup>l</sup> = w<sup><small>l+1</small> <b>T</b></sup> 
     * &delta;<sup>l+1</sup> &otimes; &part; f(x<sup>l</sup>), where &part; is 
     * the Hadamard product and &part; f(x<sup>l</sup>) is the derivative of 
     * this activation function on the input that was feed into this activation. 
     * <br>
     * {@code delta_partial} and {@code errout} may point to the same vector 
     * object
     * @param input the input to this layer that was feed in to be activated
     * @param output the activation that was produced for this layer
     * @param delta_partial the error assigned to this layer from the above
     * layer, sans the hamard product with the derivative of the layer 
     * activation. Often denoted as w<sup><small>l+1</small> <b>T</b></sup> &delta;<sup>l+1</sup>
     * @param errout the delta value or error produced for this layer
     * @param rowMajor {@code true} if the information per input is stored in 
     * rows, {@code false} if the inputs were stored by column. This parameter 
     * does not indicate if the matrices themselves are backed by a row or 
     * column major implementation
     */
    public void backprop(Matrix input, Matrix output, Matrix delta_partial, Matrix errout, boolean rowMajor);
    
    public ActivationLayer clone();
}
