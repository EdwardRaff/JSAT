package jsat.math.decayrates;

/**
 * A possible value for a decaying learning rate. The initial value will be 
 * reduced as a linear function of time. 
 * 
 * @author Edward Raff
 */
public class LinearDecay implements DecayRate
{

    public double rate(double time, double maxTime, double initial)
    {
        if(time < 0)
            throw new ArithmeticException("Negative time value given");
        return initial*(1.0-time/maxTime);
    }

    @Override
    public String toString()
    {
        return "Linear Decay";
    }
}
