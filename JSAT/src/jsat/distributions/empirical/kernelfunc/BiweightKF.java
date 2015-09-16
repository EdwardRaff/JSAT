package jsat.distributions.empirical.kernelfunc;

/**
 *
 * @author Edward Raff
 */
public class BiweightKF implements KernelFunction {

  private static class SingletonHolder {

    public static final BiweightKF INSTANCE = new BiweightKF();
  }

  private static final long serialVersionUID = -7199542934997154186L;

  /**
   * Returns the singleton instance of this class
   *
   * @return the instance of this class
   */
  public static BiweightKF getInstance() {
    return SingletonHolder.INSTANCE;
  }

  private BiweightKF() {
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
    return Math.pow(u + 1, 3) / 16.0 * (3 * u * u - 9 * u + 8);
  }

  @Override
  public double k(final double u) {
    if (Math.abs(u) > 1) {
      return 0;
    }
    return Math.pow(1 - u * u, 2) * (15.0 / 16.0);
  }

  @Override
  public double k2() {
    return 1.0 / 7.0;
  }

  @Override
  public double kPrime(final double u) {
    if (Math.abs(u) > 1) {
      return 0;
    }
    return 15.0 / 4.0 * u * (u * u - 1);
  }

  @Override
  public String toString() {
    return "Biweight Kernel";
  }
}
