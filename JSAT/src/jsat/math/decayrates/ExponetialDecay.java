package jsat.math.decayrates;

import java.util.List;
import jsat.math.FastMath;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * The Exponential Decay requires the maximum time step to be explicitly known ahead 
 * of time. Provided either in the call to 
 * {@link #rate(double, double, double) }, or internal by 
 * {@link #setMinRate(double) }. <br>
 * <br>
 * The Exponential Decay will decay at an exponential rate from the initial value
 * until the specified {@link #setMinRate(double) } is reached. 
 * 
 * @author Edward Raff
 */
public class ExponetialDecay implements DecayRate, Parameterized
{

	private static final long serialVersionUID = 4762109043998381095L;
	private double min;
    private double maxTime;

    /**
     * Creates a new Exponential Decay
     * 
     * @param min a value less than the learning rate that, that will be the 
     * minimum rate returned once the maximum time is reached
     */
    public ExponetialDecay(double min)
    {
        this(min, 1000000);
    }
    
    /**
     * Creates a new Exponential Decay
     * <br>
     * <br>
     * Note that when using {@link #rate(double, double, double) }, the maxTime 
     * is always superceded by the value given to the function. 
     * @param min a value less than the learning rate that, that will be the 
     * minimum returned value 
     * @param maxTime the maximum amount of time
     */
    public ExponetialDecay(double min, double maxTime)
    {
        setMinRate(min);
        setMaxTime(maxTime);
    }

    /**
     * Creates a new decay rate that decays down to 1e-4
     */
    public ExponetialDecay()
    {
        this(1e-4);
    }
    
        /**
     * Sets the minimum learning rate to return
     * @param min the minimum learning rate to return
     */
    public void setMinRate(double min)
    {
        if(min <= 0 || Double.isNaN(min) || Double.isInfinite(min))
            throw new RuntimeException("minRate should be positive, not " + min);
        this.min = min;
    }

    /**
     * Returns the minimum value to return from he <i>rate</i> methods
     * @return the minimum value to return 
     */
    public double getMinRate()
    {
        return min;
    }

    /**
     * Sets the maximum amount of time to allow in the rate decay. Any time 
     * value larger will be treated as the set maximum.<br>
     * <br>
     * Any calls to {@link #rate(double, double, double) } will use the value 
     * provided in that method call instead. 
     * @param maxTime the maximum amount of time to allow
     */
    public void setMaxTime(double maxTime)
    {
        if(maxTime <= 0 || Double.isInfinite(maxTime) || Double.isNaN(maxTime))
            throw new RuntimeException("maxTime should be positive, not " + maxTime);
        this.maxTime = maxTime;
    }

    /**
     * Returns the maximum time to use in the rate decay
     * @return the maximum time to use in the rate decay
     */
    public double getMaxTime()
    {
        return maxTime;
    }

    @Override
    public double rate(double time, double maxTime, double initial)
    {
        if(time < 0)
            throw new ArithmeticException("Negative time value given");
        return (initial-min)* FastMath.pow(maxTime, -Math.min(time, maxTime)/maxTime)+min;
    }

    @Override
    public double rate(double time, double initial)
    {
        return rate(time, maxTime, initial);
    }

    @Override
    public DecayRate clone()
    {
        return new ExponetialDecay(min, maxTime);
    }

    @Override
    public String toString()
    {
        return "Exponetial Decay";
    }
    
    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
