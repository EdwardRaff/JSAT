
package jsat.distributions.kernels;

import java.io.Serializable;
import jsat.linear.Vec;
import jsat.parameters.Parameterized;

/**
 * The KernelTrick is a method can can be used to alter an algorithm to do its 
 * calculations in a projected feature space, without explicitly forming the 
 * features. If an algorithm uses only dot products, the Kernel trick can be 
 * used in place of these dot products, and computes the inner product in a 
 * different feature space. 
 * <br><br>
 * All KerenlTrick objects are {@link Parameterized parameterized} so that the 
 * values of the kernel can be exposed by the algorithm that makes use of these 
 * parameters. To avoid conflicts in parameter names, the parameters of a 
 * KernelTrick should be of the form:<br>
 * &lt; SimpleClassName &gt;_&lt; Variable Name &gt;
 * 
 * @author Edward Raff
 */
public interface KernelTrick extends Parameterized, Cloneable, Serializable
{
    /**
     * Evaluate this kernel function for the two given vectors. 
     * @param a the first vector
     * @param b the first vector
     * @return the evaluation
     */
    public double eval(Vec a, Vec b);
    
    /**
     * A descriptive name for the type of KernelFunction 
     * @return a descriptive name for the type of KernelFunction 
     */
    @Override
    public String toString();
    
    public KernelTrick clone();
}
