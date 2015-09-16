package jsat.parameters;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import jsat.distributions.empirical.kernelfunc.BiweightKF;
import jsat.distributions.empirical.kernelfunc.EpanechnikovKF;
import jsat.distributions.empirical.kernelfunc.GaussKF;
import jsat.distributions.empirical.kernelfunc.KernelFunction;
import jsat.distributions.empirical.kernelfunc.TriweightKF;
import jsat.distributions.empirical.kernelfunc.UniformKF;

/**
 * A default Parameter semi-implementation for classes that require a
 * {@link KernelFunction} to be specified.
 *
 * @author Edward Raff
 */
public abstract class KernelFunctionParameter extends ObjectParameter<KernelFunction> {

  private static final long serialVersionUID = 2100826688956817533L;
  private final static List<KernelFunction> kernelFuncs = Collections.unmodifiableList(new ArrayList<KernelFunction>() {
    /**
     *
     */
    private static final long serialVersionUID = 4910454799262834767L;

    {
      add(UniformKF.getInstance());
      add(EpanechnikovKF.getInstance());
      add(GaussKF.getInstance());
      add(BiweightKF.getInstance());
      add(TriweightKF.getInstance());
    }
  });

  @Override
  public String getASCIIName() {
    return "Kernel Function";
  }

  @Override
  public List<KernelFunction> parameterOptions() {
    return kernelFuncs;
  }
}
