package jsat.distributions.empirical.kernelfunc;

/**
 *
 * @author Edward Raff
 */
public class EpanechnikovKF implements KernelFunction {

  private static class SingletonHolder {

    public static final EpanechnikovKF INSTANCE = new EpanechnikovKF();
  }

  private static final long serialVersionUID = 8688942176576932932L;

  /**
   * Returns the singleton instance of this class
   *
   * @return the instance of this class
   */
  public static EpanechnikovKF getInstance() {
    return SingletonHolder.INSTANCE;
  }

  private EpanechnikovKF() {
  }

  @Override
  public double cutOff() {
    return Math.ulp(1) + 1;
  }

  @Override
  public double intK(final double u) {
    if (u < -1) {
      return 0;
    }
    if (u > 1) {
      return 1;
    }
    return (-u * u * u + 3 * u + 2) / 4;
  }

  @Override
  public double k(final double u) {
    if (Math.abs(u) > 1) {
      return 0;
    }
    return (1 - u * u) * (3.0 / 4.0);
  }

  @Override
  public double k2() {
    return 1.0 / 5.0;
  }

  @Override
  public double kPrime(final double u) {
    if (Math.abs(u) > 1) {
      return 0;
    }
    return -u * (3.0 / 2.0);
  }

  @Override
  public String toString() {
    return "Epanechnikov Kernel";
  }
}
