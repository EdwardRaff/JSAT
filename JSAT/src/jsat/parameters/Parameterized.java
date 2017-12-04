package jsat.parameters;

import java.util.List;

/**
 * An algorithm may be Parameterized, meaning it has one or more parameters that
 * can be tuned or alter the results of the algorithm in question. 
 * 
 * @author Edward Raff
 */
public interface Parameterized
{
    /**
     * Returns the list of parameters that can be altered for this learner. 
     * @return the list of parameters that can be altered for this learner. 
     */
    default public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }
    
    /**
     * Returns the parameter with the given name. Two different strings may map 
     * to a single Parameter object. An ASCII only string, and a Unicode style 
     * string. 
     * @param paramName the name of the parameter to obtain
     * @return the Parameter in question, or null if no such named Parameter exists. 
     */
    default public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
