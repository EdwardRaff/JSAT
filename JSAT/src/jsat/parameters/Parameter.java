package jsat.parameters;

import java.io.Serializable;
import java.lang.annotation.*;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.distributions.Distribution;
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

    private static final long serialVersionUID = -6903844587637472657L;

    /**
     * Adding this annotation to a field tells the 
     * {@link #getParamsFromMethods(java.lang.Object)} method to search this 
     * object recursively for more parameter get/set
     * pairs.<br><br>
     * Placing this annotation on a {@link Collection} will cause the search to 
     * be done recursively over each item in the collection. 
     */
    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.FIELD)
    public static @interface ParameterHolder
    {
        boolean skipSelfNamePrefix() default false;
    }
    
    /**
     * Adding this annotation to a method tells the {@link #getParamsFromMethods(java.lang.Object)
     * } method to consider the Parameter object generated for that method a
     * preferred parameter for warm starting.
     */
    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    public static @interface WarmParameter
    {
        /**
         * Indicates the ordering that would be preferred by the algorithm when 
         * training. <br>
         * <br>
         * It may be the case that the model has no preference for models to be
         * trained from low to high or high to low, in which case any arbitrary
         * value may be returned - so long as it consistently returns the same
         * value.
         * 
         * @return {@code true} if it is preferred to train from low values of
         * the parameter to high values of the parameter. {@code false} is
         * returned if it is preferred to go from high to low values.
         */
        boolean prefLowToHigh();
    }
    
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
     * If {@code true}, that means this parameter is the preferred parameter to 
     * be altered when warm starting from a previous version of the same class. 
     * Being the preferred parameter means that using warms started training on 
     * a sequence of changes to this parameter should be faster than simply 
     * training normally for every combination of values.<br>
     * <br>
     * If more than one parameter would have this impact on training, the one 
     * that has the largest and most consistent impact should be selected.<br>
     * <br>
     * By default, this method will return {@code false}. 
     * 
     * @return {@code true} if warm starting on changes in this parameter has a 
     * significant impact on training speed, {@code false} otherwise. 
     */
    public boolean isWarmParameter(){
        return false;
    };
    
    /**
     * This value is meaningless if {@link #isWarmParameter() } is {@code false}
     * , and by default returns {@code false}. <br>
     * <br>
     * This should return the preferred order of warm start value training in 
     * the progression of the warm parameter, either from low values to high 
     * values (ie: the model being trained has a higher value for this parameter
     * than the warm model being trained from). 
     * 
     * @return {@code true} if warm starting on this parameter should occur from 
     * low values to high values, and {@code false} if it should go from high
     * values to low. 
     */
    public boolean preferredLowToHigh() {
        return false;
    }
   
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
     * @return a list of parameter objects generated from the given object
     */
    public static List<Parameter> getParamsFromMethods(final Object obj)
    {
        return getParamsFromMethods(obj, "");
    }
    
    private static List<Parameter> getParamsFromMethods(final Object obj, String prefix)
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
            Parameter param = getParam(obj, argClass, getMethod, setMethod, prefix + name);
            
            
            if(param != null)
                params.add(param);
        }
        
        //Find params from field objects
        
        //first get all fields of this object
        List<Field> fields = new ArrayList<Field>();
        Class curClassLevel = obj.getClass();
        while(curClassLevel != null)
        {
            fields.addAll(Arrays.asList(curClassLevel.getDeclaredFields()));
            curClassLevel = curClassLevel.getSuperclass();
        }
        
        final String simpleObjName = obj.getClass().getSimpleName();
        //For each field, check if it has our magic annotation 
        for(Field field : fields)
        {
            Annotation[] annotations = field.getAnnotations();

            for(Annotation annotation : annotations)
            {
                if(annotation.annotationType().equals(ParameterHolder.class))
                {
                    ParameterHolder annotationPH = (ParameterHolder) annotation;
                    //get the field value fromt he object passed in
                    try
                    {
                        //If its private/protected we are not int he same object chain
                        field.setAccessible(true);
                        Object paramHolder = field.get(obj);
                        
                        if(paramHolder instanceof Collection)//serach for each item in the collection 
                        {
                            Collection toSearch = (Collection) paramHolder;
                            for(Object paramHolderSub : toSearch)
                            {
                                String subPreFix = paramHolderSub.getClass().getSimpleName() + "_";
                                
                                if(annotationPH.skipSelfNamePrefix())
                                    subPreFix = prefix.replace(simpleObjName+"_", "") + subPreFix;
                                else
                                    subPreFix = prefix + subPreFix;

                                params.addAll(Parameter.getParamsFromMethods(paramHolderSub, subPreFix));
                            }
                        }
                        else if(paramHolder != null)//search the item directly
                        {
                            String subPreFix = paramHolder.getClass().getSimpleName() + "_";

                            if (annotationPH.skipSelfNamePrefix())
                                subPreFix = prefix.replace(simpleObjName + "_", "") + subPreFix;
                            else
                                subPreFix = prefix + subPreFix;

                            params.addAll(Parameter.getParamsFromMethods(paramHolder, subPreFix));
                        }
                    }
                    catch (IllegalArgumentException ex)
                    {
                        Logger.getLogger(Parameter.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    catch (IllegalAccessException ex)
                    {
                        Logger.getLogger(Parameter.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
            }
            
        }
        
        
        return params;
    }

    private static Parameter getParam(final Object targetObject, final Class varClass, final Method getMethod, final Method setMethod, final String asciiName)
    {
        return getParam(targetObject, varClass, getMethod, setMethod, asciiName, null);
    }
    
    private static Parameter getParam(final Object targetObject, final Class varClass, final Method getMethod, final Method setMethod, final String asciiName, final String uniName)
    {
        final boolean warm;
        final boolean lowToHigh;
        //lets find out if this paramter is a "warm" parameter
        Parameter.WarmParameter warmAna = null;
        warmAna = setMethod.getAnnotation(Parameter.WarmParameter.class);
        if(warmAna == null)
            warmAna = getMethod.getAnnotation(Parameter.WarmParameter.class);
        if(warmAna != null)
        {
            warm = true;
            lowToHigh = warmAna.prefLowToHigh();
        }
        else
        {
            warm = false;
            lowToHigh = false;
        }
        //lets see if we can find a method that corresponds to "guess" for this parameter
        final Method guessMethod;
        Method tmp = null;
        try
        {
            tmp = targetObject.getClass().getMethod("guess" + setMethod.getName().replaceFirst("set", ""), DataSet.class);
        }
        catch (NoSuchMethodException ex)
        {
        }
        catch (SecurityException ex)
        {
        }
        guessMethod = tmp;//ugly hack so that I can reference a final guess method in anon class below
        //ok, now lets go create the correct object type
        Parameter param = null;
        if (varClass.equals(double.class) || varClass.equals(Double.class))
        {
            param = new DoubleParameter()
            {
                private static final long serialVersionUID = -4741218633343565521L;

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
                public boolean isWarmParameter()
                {
                    return warm;
                }

                @Override
                public boolean preferredLowToHigh()
                {
                    return lowToHigh;
                }

                @Override
                public String getASCIIName()
                {
                    return asciiName;
                }

                @Override
                public String getName()
                {
                    if (uniName == null)
                        return super.getName();
                    else
                        return uniName;
                }

                @Override
                public Distribution getGuess(DataSet data)
                {
                    if (guessMethod == null)
                        return null;
                    try
                    {
                        return (Distribution) guessMethod.invoke(targetObject, data);
                    }
                    catch (Exception ex)
                    {
                    }
                    return null;
                }

            };
        }
        else if (varClass.equals(int.class) || varClass.equals(Integer.class))
        {
            param = new IntParameter()
            {

                private static final long serialVersionUID = 693593136858174197L;

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
                public Distribution getGuess(DataSet data)
                {
                    if (guessMethod == null)
                        return null;
                    try
                    {
                        return (Distribution) guessMethod.invoke(targetObject, data);
                    }
                    catch (Exception ex)
                    {
                    }
                    return null;
                }

                @Override
                public boolean isWarmParameter()
                {
                    return warm;
                }

                @Override
                public boolean preferredLowToHigh()
                {
                    return lowToHigh;
                }

                @Override
                public String getASCIIName()
                {
                    return asciiName;
                }

                @Override
                public String getName()
                {
                    if (uniName == null)
                        return super.getName();
                    else
                        return uniName;
                }
            };
        }
        else if (varClass.equals(boolean.class) || varClass.equals(Boolean.class))
        {
            param = new BooleanParameter()
            {

                private static final long serialVersionUID = 8356074301252766754L;

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
                    if (uniName == null)
                        return super.getName();
                    else
                        return uniName;
                }
            };
        }
        else if (varClass.equals(KernelFunction.class))
        {
            param = new KernelFunctionParameter()
            {
                private static final long serialVersionUID = -482809259476649959L;

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
        else if (varClass.equals(DistanceMetric.class))
        {
            param = new MetricParameter()
            {
                private static final long serialVersionUID = -2823576782267398656L;

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
        else if (varClass.equals(DecayRate.class))
        {
            param = new DecayRateParameter()
            {

                private static final long serialVersionUID = 5348280386363008701L;

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
                    if (uniName == null)
                        return super.getName();
                    else
                        return uniName;
                }
            };
        }
        else if (varClass.isEnum())//We can create an ObjectParameter for enums
        {
            param = new ObjectParameter()
            {
                private static final long serialVersionUID = -6245401198404522216L;

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
                    if (uniName == null)
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
        return sb.toString().trim();
    }
}
