package jsat.math.decayrates;

/**
 * An Exponential decay that drives the initial learning rate down to a fixed 
 * value after a fixed number of iterations. The initial value will be 
 * decayed at an exponential rate. 
 * 
 * @author Edward Raff
 */
public class ExponetialDecay implements DecayRate
{
    private double min;

    /**
     * Sets the minimum amount of learning rate that can occur. The decay will 
     * decay down to and stop at the minimum. 
     * 
     * @param min a value less than the learning rate that, that will be the 
     * minimum returned value 
     */
    public ExponetialDecay(double min)
    {
        this.min = min;
    }

    /**
     * Creates a new decay rate that decays down to zero
     */
    public ExponetialDecay()
    {
        this(0.0);
    }
    
    

    @Override
    public double rate(double time, double maxTime, double initial)
    {
        if(time < 0)
            throw new ArithmeticException("Negative time value given");
        return (initial-min)*Math.exp(  -time / (maxTime / Math.log(maxTime)))+min;
    }

    @Override
    public String toString()
    {
        return "Exponetial Decay";
    }
}
