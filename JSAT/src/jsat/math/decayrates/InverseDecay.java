package jsat.math.decayrates;

import java.util.List;
import jsat.classifiers.svm.Pegasos;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 *
 * Decays an input by the inverse of the amount of time that has occurred, the 
 * max time being irrelevant. More specifically as 
 * &eta; / ({@link #setAlpha(double) &alpha;}({@link #setTau(double) &tau;} + time))<br>
 * <br>
 * This decay rate can be used to create the same rate used by {@link Pegasos}, 
 * by using an initial rate of 1 and setting &tau; = 1 and &alpha; = &lambda;, 
 * where &lambda; is the regularization term used by the method calling the 
 * decay rate. 
 * 
 * @author Edward Raff
 */
public class InverseDecay implements DecayRate, Parameterized
{

	private static final long serialVersionUID = 2756825625752543664L;
	private double tau;
    private double alpha;

    /**
     * Creates a new Inverse decay rate
     * @param tau the initial time offset
     * @param alpha the time scaling 
     */
    public InverseDecay(double tau, double alpha)
    {
        setTau(tau);
        setAlpha(alpha);
    }

    /**
     * Creates a new Inverse Decay rate
     */
    public InverseDecay()
    {
        this(1, 1);
    }
    
    

    /**
     * Controls the scaling of the divisor, increasing &alpha; dampens the 
     * whole range of values. Increasing it increases the values.
     * value.
     * @param alpha the scaling parameter
     */
    public void setAlpha(double alpha)
    {
        if(alpha <= 0 || Double.isInfinite(alpha) || Double.isNaN(alpha))
            throw new IllegalArgumentException("alpha must be a positive constant, not " + alpha);
        this.alpha = alpha;
    }

    /**
     * Returns the scaling parameter
     * @return the scaling parameter
     */
    public double getAlpha()
    {
        return alpha;
    }
    
    /**
     * Controls the rate early in time, but has a decreasing impact on the rate 
     * returned as time goes forward. Larger values of &tau; dampen the initial 
     * rates returned, while lower values let the initial rates start higher.  
     * 
     * @param tau the early rate dampening parameter
     */
    public void setTau(double tau)
    {
        if(tau <= 0 || Double.isInfinite(tau) || Double.isNaN(tau))
            throw new IllegalArgumentException("tau must be a positive constant, not " + tau);
        this.tau = tau;
    }

    /**
     * Returns the early rate dampening parameter
     * @return the early rate dampening parameter
     */
    public double getTau()
    {
        return tau;
    }
    
    @Override
    public double rate(double time, double maxTime, double initial)
    {
        return rate(time, initial);
    }
    
    @Override
    public double rate(double time, double initial)
    {
        return initial/(alpha*(tau+time));
    }

    @Override
    public DecayRate clone()
    {
        return new InverseDecay(tau, alpha);
    }

    @Override
    public String toString()
    {
        return "Inverse Decay";
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
