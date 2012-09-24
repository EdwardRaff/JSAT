package jsat.parameters;

import java.util.*;

/**
 * This interface provides a programmable manner in which the parameters of an 
 * algorithm may be altered and adjusted. 
 * 
 * @author Edward Raff
 */
public abstract class Parameter
{
    /**
     * Some variables of a learning method may be adjustable without having to 
     * re-train the whole data set. <tt>false</tt> is returned if this is such a 
     * parameter, <tt>true</tt> if the learning method will need to be 
     * retrained after the parameter has changed. <br><br>
     * By default, this method returns <tt>true</tt> unless overwritten, as it 
     * is always safe to retrain the classifier if a parameter was changed. 
     * @return <tt>true</tt> if changing this parameter requires a re-training 
     * of the algorithm, or <tt>false</tt> if no-retraining is needed to take
     * effect. 
     */
    public boolean requiresRetrain(){
        return true;
    };
   
    /**
     * Returns the name of this parameter using only valid ACII characters. 
     * @return the ACII name 
     */
    abstract public String getASCIIName();
    
    /**
     * Returns the display name of this parameter. By default, this returns the 
     * {@link #getASCIIName() ASCII name} of the parameter. If one exists, a 
     * name using Unicode characters may be returned instead. 
     * 
     * @return the name of this parameter
     */
    public String getName()
    {
        return getASCIIName();
    }

    @Override
    public String toString()
    {
        return getName();
    }

    @Override
    public int hashCode()
    {
        return getName().hashCode();
    }
    
    /**
     * Returns a string indicating the value currently held by the Parameter 
     * 
     * @return a string representation of the parameter's value
     */
    abstract public String getValueString();
    
    @Override
    public boolean equals(Object obj)
    {
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        final Parameter other = (Parameter) obj;
        return this.getName().equals(other.getName());
    }
        
    /**
     * Creates a map of all possible parameter names to their corresponding object. No two parameters may have the same name. 
     * @param params the list of parameters to create a map for
     * @return a map of string names to their parameters
     * @throws RuntimeException if two parameters have the same name
     */
    public static Map<String, Parameter> toParameterMap(List<Parameter> params)
    {
        Map<String, Parameter> map = new HashMap<String, Parameter>(params.size());
        for(Parameter param : params)
        {
            if(map.put(param.getASCIIName(), param) != null)
                throw new RuntimeException("Name collision, two parameters use the name '" + param.getASCIIName() + "'");
            if(!param.getName().equals(param.getASCIIName()))//Dont put it in again
                if(map.put(param.getName(), param) != null)
                    throw new RuntimeException("Name collision, two parameters use the name '" + param.getName() + "'");
        }
        return map;
    }
}
