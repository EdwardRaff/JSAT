/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.math.rootfinding;

import static java.lang.Math.PI;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.linear.Vec;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class BisectionTest {

  @BeforeClass
  public static void setUpClass() throws Exception {
  }

  @AfterClass
  public static void tearDownClass() throws Exception {
  }

  /**
   * Root at 0
   */
  Function sinF = new Function() {

    /**
     *
     */
    private static final long serialVersionUID = -5830265331948124543L;

    @Override
    public double f(final double... x) {
      return sin(x[0]);
    }

    @Override
    public double f(final Vec x) {
      return f(x.arrayCopy());
    }
  };

  /**
   * Root at 0 + 2nd param
   */
  Function sinFp1 = new Function() {

    /**
     *
     */
    private static final long serialVersionUID = -8881811957419055036L;

    @Override
    public double f(final double... x) {
      return sin(x[0] + x[1]);
    }

    @Override
    public double f(final Vec x) {
      return f(x.arrayCopy());
    }
  };

  /**
   * Root at approx -4.87906
   */
  Function polyF = new Function() {

    /**
     *
     */
    private static final long serialVersionUID = -6863811831444846780L;

    @Override
    public double f(final double... x) {
      final double xp = x[0];

      return pow(xp, 3) + 5 * pow(xp, 2) + xp + 2;
    }

    @Override
    public double f(final Vec x) {
      return f(x.arrayCopy());
    }
  };

  public BisectionTest() {
  }

  @Before
  public void setUp() {
  }

  /**
   * Test of root method, of class Bisection.
   */
  @Test
  public void testRoot_4args() {
    System.out.println("root");
    final double eps = 1e-15;
    double result = Bisection.root(-PI / 2, PI / 2, sinF);
    assertEquals(0, result, eps);

    result = Bisection.root(-6, 6, polyF);
    assertEquals(-4.8790576334840479813, result, eps);

    result = Bisection.root(-6, 6, polyF, 0);
    assertEquals(-4.8790576334840479813, result, eps);

    result = Bisection.root(-PI / 2, PI / 2, sinFp1, 0.0, 1.0);
    assertEquals(-1.0, result, eps);

    try {
      result = Bisection.root(-PI / 2, PI / 2, sinFp1);
      fail("Should not have run");
    } catch (final Exception ex) {
    }
  }

  /**
   * Test of root method, of class Bisection.
   */
  @Test
  public void testRoot_5args() {
    System.out.println("root");
    final double eps = 1e-15;
    double result = Bisection.root(eps, -PI / 2, PI / 2, sinF);
    assertEquals(0, result, eps);

    result = Bisection.root(eps, -6, 6, polyF);
    assertEquals(-4.8790576334840479813, result, eps);

    result = Bisection.root(eps, -6, 6, 0, polyF);
    assertEquals(-4.8790576334840479813, result, eps);

    result = Bisection.root(eps, -PI / 2, PI / 2, sinFp1, 0.0, 1.0);
    assertEquals(-1.0, result, eps);

    try {
      result = Bisection.root(eps, -PI / 2, PI / 2, sinFp1);
      fail("Should not have run");
    } catch (final Exception ex) {
    }
  }

  @Test
  public void testRoot_6args() {
    System.out.println("root");
    final double eps = 1e-15;
    double result = Bisection.root(eps, -PI / 2, PI / 2, 0, sinF);
    assertEquals(0, result, eps);

    result = Bisection.root(eps, -PI / 2, PI / 2, 0, sinFp1, 0.0, 1.0);
    assertEquals(-1.0, result, eps);

    result = Bisection.root(eps, -PI / 2, PI / 2, 1, sinFp1, 3.0, 0.0);
    assertEquals(PI - 3.0, result, eps);

    result = Bisection.root(eps, -6, 6, 0, polyF);
    assertEquals(-4.8790576334840479813, result, eps);
  }

  /**
   * Test of root method, of class Bisection.
   */
  @Test
  public void testRoot_7args() {
    System.out.println("root");
    final double eps = 1e-13;
    final int maxIterations = 1000;
    double result = Bisection.root(eps, maxIterations, -PI / 2, PI / 2, 0, sinF);
    assertEquals(0, result, eps);

    result = Bisection.root(eps, maxIterations, -PI / 2, PI / 2, 0, sinFp1, 0.0, 1.0);
    assertEquals(-1.0, result, eps);

    result = Bisection.root(eps, maxIterations, -PI / 2, PI / 2, 1, sinFp1, 3.0, 0.0);
    assertEquals(PI - 3.0, result, eps);

    result = Bisection.root(eps, maxIterations, -6, 6, 0, polyF);
    assertEquals(-4.8790576334840479813, result, eps);
  }
}
