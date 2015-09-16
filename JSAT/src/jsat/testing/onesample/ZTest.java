package jsat.testing.onesample;

import jsat.distributions.Normal;
import jsat.linear.Vec;
import jsat.text.GreekLetters;

/**
 *
 * @author Edward Raff
 */
public class ZTest implements OneSampleTest {

  private final Normal norm;

  private double sampleMean;
  private double sampleDev;
  private int sampleSize;
  private H1 h1;

  /**
   * The mean of the null hypothesis
   */
  private double hypoMean;

  public ZTest() {
    this(0, 1, 1);
  }

  public ZTest(final double sampleMean, final double sampleDev, final int sampleSize) {
    this(H1.NOT_EQUAL, sampleMean, sampleDev, sampleSize);
  }

  public ZTest(final H1 h1, final double sampleMean, final double sampleDev, final int sampleSize) {
    this.h1 = h1;
    hypoMean = 0;
    this.sampleMean = sampleMean;
    this.sampleDev = sampleDev;
    this.sampleSize = sampleSize;
    norm = new Normal();
  }

  public ZTest(final H1 h1, final Vec data) {
    this(h1, data.mean(), data.standardDeviation(), data.length());
  }

  public ZTest(final Vec data) {
    this(data.mean(), data.standardDeviation(), data.length());
  }

  @Override
  public String getAltVar() {
    return GreekLetters.mu + "0";
  }

  @Override
  public String getNullVar() {
    return GreekLetters.mu;
  }

  @Override
  public String[] getTestVars() {
    return new String[] { GreekLetters.bar("x"), GreekLetters.sigma, "n" };
  }

  @Override
  public double pValue() {
    final double se = sampleDev / Math.sqrt(sampleSize);

    final double zScore = (sampleMean - hypoMean) / se;

    if (h1 == H1.NOT_EQUAL) {
      return norm.cdf(-Math.abs(zScore)) * 2;
    } else if (h1 == H1.LESS_THAN) {
      return norm.cdf(zScore);
    } else {
      return 1 - norm.cdf(zScore);
    }
  }

  @Override
  public void setAltHypothesis(final H1 h1) {
    this.h1 = h1;
  }

  @Override
  public void setAltVar(final double altVar) {
    hypoMean = altVar;
  }

  @Override
  public void setTestUsingData(final Vec data) {
    sampleMean = data.mean();
    sampleDev = data.standardDeviation();
    sampleSize = data.length();
  }

  @Override
  public void setTestVars(final double[] testVars) {
    sampleMean = testVars[0];
    sampleDev = testVars[1];
    sampleSize = (int) testVars[2];
  }

  @Override
  public String testName() {
    return "One Sample Z-Test";
  }

  @Override
  public H1[] validAlternate() {
    return new H1[] { H1.LESS_THAN, H1.NOT_EQUAL, H1.GREATER_THAN };
  }

}
