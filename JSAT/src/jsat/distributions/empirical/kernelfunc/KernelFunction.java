
package jsat.distributions.empirical.kernelfunc;

/**
 *
 * See http://en.wikipedia.org/wiki/Kernel_(statistics)
 * 
 * @author Edward Raff
 */
public interface KernelFunction
{
    public double k(double u);
    /**
     * Computes the value of the finite integral from -Infinity up to the value u, of the function given by {@link #k(double) }
     * @param u
     * @return 
     */
    public double intK(double u);
    
    public double k2();
    
    /**
     * As the value of |u| for the kernel function approaches infinity, the
     * value of k(u) approaches zero. This function returns the minimal 
     * absolute value of u for which k(u) returns 0
     * 
     * @return the first value for which k(u) = 0
     */
    public double cutOff();
}
