
package jsat.parameters;

import java.util.*;
import jsat.distributions.empirical.kernelfunc.*;

/**
 * A default Parameter semi-implementation for classes that require a 
 * {@link KernelFunction} to be specified. 
 * 
 * @author Edward Raff
 */
public abstract class KernelFunctionParameter extends ObjectParameter<KernelFunction>
{
    private final static List<KernelFunction> kernelFuncs = Collections.unmodifiableList(new ArrayList<KernelFunction>()
    {{
        add(UniformKF.getInstance());
        add(EpanechnikovKF.getInstance());
        add(GaussKF.getInstance());
        add(BiweightKF.getInstance());
        add(TriweightKF.getInstance());
    }});

    @Override
    public List<KernelFunction> parameterOptions()
    {
        return kernelFuncs;
    }

    @Override
    public String getASCIIName()
    {
        return "Kernel Function";
    }
}
