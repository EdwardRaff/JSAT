package jsat.parameters;

import jsat.DataSet;
import jsat.distributions.Distribution;

/**
 * An integer parameter that may be altered. 
 * 
 * @author Edward Raff
 */
public abstract class IntParameter extends Parameter
{

    private static final long serialVersionUID = -8467918069240345315L;

    /**
     * Returns the current value for the parameter.
     *
     * @return the value for this parameter.
     */
    abstract public int getValue();
    
    /**
     * Sets the value for this parameter. 
     * @return <tt>true</tt> if the value was set, <tt>false</tt> if the value 
     * was invalid, and thus ignored. 
     */
    abstract public boolean setValue(int val);
    
    /**
     * This method allows one to obtain a distribution that represents a
     * reasonable "guess" at the range of values that would work for this
     * parameter. If the DataSet is an instance of {@link ClassificationDataSet}
     * or {@link RegressionDataSet}, the method may choose to assume that the
     * value is being guessed for the specified task and change its behavior<br>
     * <br>
     * Providing a getGuess is not required, and returns {@code null} if
     * guessing is not supported.
     *
     * @param data the data with which we want a reasonable guess for this
     * parameter
     * @return a distribution that represents a reasonable guess of a good value
     * for this parameter given the input data
     */
    public Distribution getGuess(DataSet data)
    {
        return null;
    }

    @Override
    public String getValueString() 
    {
        return Integer.toString(getValue());
    }
}
