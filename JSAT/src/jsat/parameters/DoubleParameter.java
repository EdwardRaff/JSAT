package jsat.parameters;

/**
 * A double parameter that may be altered. 
 * 
 * @author Edward Raff
 */
public abstract class DoubleParameter extends Parameter
{
    /**
     * Returns the current value for the parameter. 
     * @return the value for this parameter. 
     */
    abstract public double getValue();
    
    /**
     * Sets the value for this parameter. 
     * @return <tt>true</tt> if the value was set, <tt>false</tt> if the value 
     * was invalid, and thus ignored. 
     */
    abstract public boolean setValue(double val);
}
