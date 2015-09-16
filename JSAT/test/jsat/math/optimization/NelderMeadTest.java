package jsat.math.optimization;

import static org.junit.Assert.assertEquals;
import java.util.ArrayList;
import java.util.List;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public class NelderMeadTest {

  @BeforeClass
  public static void setUpClass() throws Exception {
  }

  @AfterClass
  public static void tearDownClass() throws Exception {
  }

  public NelderMeadTest() {
  }

  @Before
  public void setUp() {
  }

  /**
   * Test of optimize method, of class NelderMead.
   */
  @Test
  public void testOptimize_4args() {
    System.out.println("optimize");

    final NelderMead instance = new NelderMead();
    final Function banana = new RosenbrockFunction();
    final DenseVector search = DenseVector.toDenseVec(1.05, 0.95, 1.05);
    final List<Vec> input = new ArrayList<Vec>();
    input.add(search);
    final Vec x = instance.optimize(1e-6, 1000, banana, input);
    assertEquals(0.0, banana.f(x), 1e-3);// Its a hard function to get, we often
                                         // get close
  }

  /**
   * Test of optimize method, of class NelderMead.
   */
  @Test
  public void testOptimize_7args() {
    System.out.println("optimize");
    final NelderMead instance = new NelderMead();
    final Function banana = new RosenbrockFunction();
    final DenseVector search = DenseVector.toDenseVec(1.05, 0.95, 1.05);
    final Vec x = instance.optimize(1e-6, 10000, banana, banana, search, null, null);
    assertEquals(0.0, banana.f(x), 1e-3);// Its a hard function to get, we often
                                         // get close
  }

  /**
   * Test of optimize method, of class NelderMead.
   */
  @Test
  public void testOptimize_8args() {
    System.out.println("optimize");

    final NelderMead instance = new NelderMead();
    final Function banana = new RosenbrockFunction();
    final DenseVector search = DenseVector.toDenseVec(1.05, 0.95, 1.05);
    final Vec x = instance.optimize(1e-6, 1000, banana, banana, search, null, null, null);
    assertEquals(0.0, banana.f(x), 1e-3);// Its a hard function to get, we often
                                         // get close
  }
}
