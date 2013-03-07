package jsat.math.decayrates;

/**
 *
 * Decays an input by the inverse of the amount of time that has occurred, the 
 * max time being irrelevant. More specifically as 1 / (1 + time/rate)
 * 
 * @author Edward Raff
 */
public class InverseDecay implements DecayRate
{
    private double rate;

    /**
     * Creates a new inverse decay object
     * @param rate the positive decay rate. Higher values decay slower
     */
    public InverseDecay(double rate)
    {
        if(rate <= 0 || Double.isNaN(rate) || Double.isInfinite(rate))
            throw new ArithmeticException("Decay rate must be positive, not " + rate);
        this.rate = rate;
    }

    /**
     * Creates a new inverse decay object with a default rate of 2.0
     */
    public InverseDecay()
    {
        this(2.0);
    }

    @Override
    public double rate(double time, double maxTime, double initial)
    {
        return initial/(1.0+time/rate);
    }

    @Override
    public String toString()
    {
        return "Inverse Decay";
    }
}
