package jsat.parameters;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.*;
import jsat.distributions.empirical.kernelfunc.KernelFunction;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.math.decayrates.DecayRate;

/**
 * This interface provides a programmable manner in which the parameters of an 
 * algorithm may be altered and adjusted. 
 * 
 * @author Edward Raff
 */
public abstract class Parameter implements Serializable
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
    
    /**
     * Given an object, this method will use reflection to automatically find 
     * getter and setter method pairs, and create Parameter object for each 
     * getter setter pair.<br>
     * Getters are found by searching for no argument methods that start with 
     * "get" or "is". Setters are found by searching for one argument methods 
     * that start with "set". 
     * A getter and setter are a pair only if everything after the prefix is the 
     * same in the method's name, and the return type of the getter is the same
     * class as the argument for the setter. <br>
     * Current types supported are:
     * <ul>
     * <li>integer</li>
     * <li>doubles</li>
     * <li>booleans</li>
     * <li>{@link KernelFunction Kernel Functions}</li>
     * <li>{@link DistanceMetric Distance Metrics}</li>
     * <li>{@link Enum Enums}</li>
     * </ul>
     * 
     * @param obj
     * @return 
     */
    public static List<Parameter> getParamsFromMethods(final Object obj)
    {
        Map<String, Method> getMethods = new HashMap<String, Method>();
        Map<String, Method> setMethods = new HashMap<String, Method>();
        
        //Collect potential get/set method pairs
        for(Method method : obj.getClass().getMethods())
        {
            int paramCount = method.getParameterTypes().length;
            if(method.isVarArgs() || paramCount > 1)
                continue;
            String name = method.getName();
            if(name.startsWith("get") && paramCount == 0)
                getMethods.put(name.substring(3), method);
            else if(name.startsWith("is") && paramCount == 0)
                getMethods.put(name.substring(2), method);
            else if(name.startsWith("set") && paramCount == 1)
                setMethods.put(name.substring(3), method);
        }
        
        //Find pairings and add to list
        List<Parameter> params = new ArrayList<Parameter>(Math.min(getMethods.size(), setMethods.size()));
        for(Map.Entry<String, Method> entry : setMethods.entrySet())
        {
            final Method setMethod = entry.getValue();
            final Method getMethod = getMethods.get(entry.getKey());
            if(getMethod == null)
                continue;
            
            final Class retClass = getMethod.getReturnType();
            final Class argClass = entry.getValue().getParameterTypes()[0];
            if(!retClass.equals(argClass))
                continue;
            final String name = spaceCamelCase(entry.getKey());
            //Found a match do we know how to handle it?
            Parameter param = getParam(obj, argClass, getMethod, setMethod, name);
            
            
            if(param != null)
                params.add(param);
        }
        
        
        return params;
    }

    private static Parameter getParam(final Object targetObject, final Class varClass, final Method getMethod, final Method setMethod, final String asciiName)
    {
        return getParam(targetObject, varClass, getMethod, setMethod, asciiName, null);
    }
    
    private static Parameter getParam(final Object targetObject, final Class varClass, final Method getMethod, final Method setMethod, final String asciiName, final String uniName)
    {
        Parameter param = null;
            if(varClass.equals(double.class) || varClass.equals(Double.class))
            {
                param = new DoubleParameter() 
                {

                    @Override
                    public double getValue()
                    {
                        try
                        {
                            return (Double) getMethod.invoke(targetObject);
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        return Double.NaN;
                    }

                    @Override
                    public boolean setValue(double val)
                    {
                        try
                        {
                            setMethod.invoke(targetObject, val);
                            return true;
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        
                        return false;
                    }

                    @Override
                    public String getASCIIName()
                    {
                        return asciiName;
                    }

                    @Override
                    public String getName()
                    {
                        if(uniName == null)
                            return super.getName();
                        else
                            return uniName;
                    }
                    
                    
                };
            }
            else if(varClass.equals(int.class) || varClass.equals(Integer.class))
            {
                param = new IntParameter() 
                {

                    @Override
                    public int getValue()
                    {
                        try
                        {
                            return (Integer) getMethod.invoke(targetObject);
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        return -1;
                    }

                    @Override
                    public boolean setValue(int val)
                    {
                        try
                        {
                            setMethod.invoke(targetObject, val);
                            return true;
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        
                        return false;
                    }

                    @Override
                    public String getASCIIName()
                    {
                        return asciiName;
                    }
                    
                    @Override
                    public String getName()
                    {
                        if(uniName == null)
                            return super.getName();
                        else
                            return uniName;
                    }
                };
            }
            else if(varClass.equals(boolean.class) || varClass.equals(Boolean.class))
            {
                param = new BooleanParameter() 
                {

                    @Override
                    public boolean getValue()
                    {
                        try
                        {
                            return (Boolean) getMethod.invoke(targetObject);
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        return false;
                    }

                    @Override
                    public boolean setValue(boolean val)
                    {
                        try
                        {
                            setMethod.invoke(targetObject, val);
                            return true;
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        
                        return false;
                    }

                    @Override
                    public String getASCIIName()
                    {
                        return asciiName;
                    }
                    
                    @Override
                    public String getName()
                    {
                        if(uniName == null)
                            return super.getName();
                        else
                            return uniName;
                    }
                };
            }
            else if(varClass.equals(KernelFunction.class))
            {
                param = new KernelFunctionParameter() 
                {
                    @Override
                    public KernelFunction getObject()
                    {
                        try
                        {
                            return (KernelFunction) getMethod.invoke(targetObject);
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        return null;
                    }

                    @Override
                    public boolean setObject(KernelFunction val)
                    {
                        try
                        {
                            setMethod.invoke(targetObject, val);
                            return true;
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        
                        return false;
                    }
                };
            }
            else if(varClass.equals(DistanceMetric.class))
            {
                param = new MetricParameter() 
                {
                    @Override
                    public DistanceMetric getMetric()
                    {
                        try
                        {
                            return (DistanceMetric) getMethod.invoke(targetObject);
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        return null;
                    }

                    @Override
                    public boolean setMetric(DistanceMetric val)
                    {
                        try
                        {
                            setMethod.invoke(targetObject, val);
                            return true;
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        
                        return false;
                    }
                };
            }
            else if(varClass.equals(DecayRate.class))
            {
                param = new DecayRateParameter() {

                    @Override
                    public DecayRate getObject()
                    {
                        try
                        {
                            return (DecayRate) getMethod.invoke(targetObject);
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        
                        return null;
                    }

                    @Override
                    public boolean setObject(DecayRate obj)
                    {
                        try
                        {
                            setMethod.invoke(targetObject, obj);
                            return true;
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        
                        return false;
                    }
                    
                    @Override
                    public String getASCIIName()
                    {
                        return asciiName;
                    }
                    
                    @Override
                    public String getName()
                    {
                        if(uniName == null)
                            return super.getName();
                        else
                            return uniName;
                    }
                };
            }
            else if(varClass.isEnum())//We can create an ObjectParameter for enums
            {
                param = new ObjectParameter() {

                    @Override
                    public Object getObject()
                    {
                        try
                        {
                            return getMethod.invoke(targetObject);
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        return null;
                    }

                    @Override
                    public boolean setObject(Object val)
                    {
                        try
                        {
                            setMethod.invoke(targetObject, val);
                            return true;
                        }
                        catch (Exception ex)
                        {
                            
                        }
                        
                        return false;
                    }

                    @Override
                    public List parameterOptions()
                    {
                        return Collections.unmodifiableList(Arrays.asList(varClass.getEnumConstants()));
                    }

                    @Override
                    public String getASCIIName()
                    {
                        return asciiName;
                    }
                    
                    @Override
                    public String getName()
                    {
                        if(uniName == null)
                            return super.getName();
                        else
                            return uniName;
                    }
                };
            }
            return param;
    }
    
    /**
     * Returns a version of the same string that has spaced inserted before each
     * capital letter 
     * @param in the CamelCase string
     * @return the spaced Camel Case string
     */
    private static String spaceCamelCase(String in)
    {
        StringBuilder sb = new StringBuilder(in.length()+5);
        for(int i = 0; i < in.length(); i++)
        {
            char c = in.charAt(i);
            if(Character.isUpperCase(c))
                sb.append(' ');
            sb.append(c);
        }
        return sb.toString();
    }
}
