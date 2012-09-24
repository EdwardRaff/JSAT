package jsat.parameters;

/**
 * An integer parameter that may be altered. 
 * 
 * @author Edward Raff
 */
public abstract class IntParameter extends Parameter
{
    /**
     * Returns the current value for the parameter. 
     * @return the value for this parameter. 
     */
    abstract public int getValue();
    
    /**
     * Sets the value for this parameter. 
     * @return <tt>true</tt> if the value was set, <tt>false</tt> if the value 
     * was invalid, and thus ignored. 
     */
    abstract public boolean setValue(int val);

    @Override
    public String getValueString() 
    {
        return Integer.toString(getValue());
    }
}
