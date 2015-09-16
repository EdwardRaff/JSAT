package jsat.math.optimization;

import java.util.Random;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.FunctionVec;
import org.junit.After;
import org.junit.AfterClass;
import static org.junit.Assert.*;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Edward Raff
 */
public class LBFGSTest {

  public LBFGSTest() {
  }

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of optimize method, of class LBFGS.
   */
  @Test
  public void testOptimize() {
    System.out.println("optimize");
    Random rand = new Random();
    Vec x0 = new DenseVector(20);
    for (int i = 0; i < x0.length(); i++) {
      x0.set(i, rand.nextDouble());
    }

    RosenbrockFunction f = new RosenbrockFunction();
    FunctionVec fp = f.getDerivative();
    LBFGS instance = new LBFGS();

    for (LineSearch lineSearch : new LineSearch[]{new BacktrackingArmijoLineSearch(), new WolfeNWLineSearch()}) {
      instance.setLineSearch(lineSearch);
      Vec w = new DenseVector(x0.length());
      instance.optimize(1e-4, w, x0, f, fp, null);

      for (int i = 0; i < w.length(); i++) {
        assertEquals(1.0, w.get(i), 1e-4);
      }
      assertEquals(0.0, f.f(w), 1e-4);
    }
  }
}
