package jsat.parameters;

/**
 * A boolean parameter that may be altered. 
 * 
 * @author Edward Raff
 */
public abstract class BooleanParameter extends Parameter
{

	private static final long serialVersionUID = 4961692453234546675L;

	/**
     * Returns the current value for the parameter. 
     * @return the value for this parameter. 
     */
    abstract public boolean getValue();
    
    /**
     * Sets the value for this parameter. 
     * @return <tt>true</tt> if the value was set, <tt>false</tt> if the value 
     * was invalid, and thus ignored. 
     */
    abstract public boolean setValue(boolean val);
    
    @Override
    public String getValueString() 
    {
        return Boolean.toString(getValue());
    }
}
