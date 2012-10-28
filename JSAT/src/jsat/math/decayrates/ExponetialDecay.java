package jsat.math.decayrates;

/**
 * A possible value for a decaying learning rate. The initial value will be 
 * decayed at an exponential rate. 
 * 
 * @author Edward Raff
 */
public class ExponetialDecay implements DecayRate
{

    public double rate(double time, double maxTime, double initial)
    {
        if(time < 0)
            throw new ArithmeticException("Negative time value given");
        return initial*Math.exp(  -time / (maxTime / Math.log(maxTime)));
    }

    @Override
    public String toString()
    {
        return "Exponetial Decay";
    }
}
