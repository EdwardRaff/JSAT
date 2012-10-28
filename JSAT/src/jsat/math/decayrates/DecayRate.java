package jsat.math.decayrates;

/**
 * Many algorithms use a learning rate to adjust the step size by which the 
 * search space is covered. In practice, it is often useful to reduce this 
 * learning rate over time. In this way, large steps can be taken in the 
 * beginning when we are far from the solution, and smaller steps when we have 
 * gotten closer to the solution and do not want to step too far away. 
 * 
 * @author Edward Raff
 */
public interface DecayRate
{
    /**
     * Decays the initial value. The value returned will be a value in the range
     * (0, <tt>initial</tt>]. It will always be non zero, though it may be very 
     * small in the edges of the algorithm. 
     * 
     * @param time the current time through the algorithm in the range 
     * [0, <tt>maxTime</tt>]
     * @param maxTime the maximum amount of time that can be progressed
     * @param initial the initial value 
     * @return the decayed value over time of the <tt>initial</tt> value
     * @throws ArithmeticException if the time is negative
     */
    public double rate(double time, double maxTime, double initial);
}
