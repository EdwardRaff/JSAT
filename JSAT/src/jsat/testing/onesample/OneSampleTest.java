package jsat.testing.onesample;

import jsat.linear.Vec;
import jsat.testing.StatisticTest;

/**
 *
 * @author Edward Raff
 */
public interface OneSampleTest extends StatisticTest {

  public String getAltVar();

  public String getNullVar();

  public String[] getTestVars();

  public void setAltVar(double altVar);

  /**
   * Sets the statistics that will be tested against an alternate hypothesis.
   *
   * @param data
   */
  public void setTestUsingData(Vec data);

  public void setTestVars(double[] testVars);

}
