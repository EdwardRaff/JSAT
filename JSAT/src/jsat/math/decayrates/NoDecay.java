package jsat.math.decayrates;

/**
 * A possible value for a decaying learning rate. NoDecay will perform no 
 * decaying of the initial value, the initial value will always be returned 
 * regardless of the input.
 * 
 * @author Edward Raff
 */
public class NoDecay implements DecayRate
{

	private static final long serialVersionUID = -4502356199281880268L;

	@Override
    public double rate(double time, double maxTime, double initial)
    {
        return rate(time, initial);
    }
    
    @Override
    public double rate(double time, double initial)
    {
        if(time < 0)
            throw new ArithmeticException("Negative time value given");
        return initial;
    }

    @Override
    public DecayRate clone()
    {
        return new NoDecay();
    }

    @Override
    public String toString()
    {
        return "NoDecay";
    }
}
