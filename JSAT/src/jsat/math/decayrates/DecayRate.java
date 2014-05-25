package jsat.math.decayrates;

import java.io.Serializable;

/**
 * Many algorithms use a learning rate to adjust the step size by which the 
 * search space is covered. In practice, it is often useful to reduce this 
 * learning rate over time. In this way, large steps can be taken in the 
 * beginning when we are far from the solution, and smaller steps when we have 
 * gotten closer to the solution and do not want to step too far away. 
 * 
 * @author Edward Raff
 */
public interface DecayRate extends Serializable
{
    /**
     * Decays the initial value over time. 
     * 
     * @param time the current time through the algorithm in the range 
     * [0, <tt>maxTime</tt>]
     * @param maxTime the maximum time step that will be seen
     * @param initial the initial value 
     * @return the decayed value over time of the <tt>initial</tt> value
     * @throws ArithmeticException if the time is negative
     */
    public double rate(double time, double maxTime, double initial);
    
    /**
     * Decays the initial value over time. 
     * 
     * @param time the current time step to return a value for
     * @param initial the initial learning rate
     * @return the decayed value 
     */
    public double rate(double time, double initial);
    
    public DecayRate clone();
}
