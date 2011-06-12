
package jsat.distributions.kernels;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public interface KernelTrick
{
    /**
     * Evaluate this kernel function for the two given vectors. 
     * @param a the first vector
     * @param b the first vector
     * @return the evaluation
     */
    public double eval(Vec a, Vec b);
    
    /**
     * 
     * @return a descriptive name for the type of KernelFunction 
     */
    @Override
    public String toString();
}
