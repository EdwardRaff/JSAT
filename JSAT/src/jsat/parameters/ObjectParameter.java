package jsat.parameters;

import java.util.List;

/**
 * A parameter that could be one of a finite number of possible objects. 
 * 
 * @author Edward Raff
 */
public abstract class ObjectParameter<T> extends Parameter
{

	private static final long serialVersionUID = 7639067170001873762L;

	/**
     * Returns the current object value
     * @return the current object set for the parameter
     */
    abstract public T getObject();
    
    /**
     * Sets the parameter to the given object
     * @param obj the new parameter value
     * @return <tt>true</tt> if the value was set, <tt>false</tt> if the value 
     * was invalid, and thus ignored. 
     */
    abstract public boolean setObject(T obj);
    
    /**
     * Returns a list of all possible objects that may be used as a parameter. 
     * @return  a list of all possible objects that may be used as a parameter. 
     */
    abstract public List<T> parameterOptions();
    
    @Override
    public String getValueString() 
    {
        return getObject().toString();
    }
    
}
