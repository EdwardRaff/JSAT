package jsat.regression.evaluation;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Edward Raff
 */
public class RelativeSquaredErrorTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public RelativeSquaredErrorTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of getScore method, of class RelativeSquaredError.
   */
  @Test
  public void testGetScore() {
    System.out.println("getScore");
    final RelativeSquaredError scorer = new RelativeSquaredError();
    final RelativeSquaredError otherHalf = scorer.clone();

    assertEquals(scorer, otherHalf);
    assertEquals(scorer.hashCode(), otherHalf.hashCode());
    assertTrue(scorer.lowerIsBetter());

    assertFalse(scorer.equals(""));
    assertFalse(scorer.hashCode() == "".hashCode());

    final double[] pred = new double[] { 0, 2, 4, 6, 8, 9 };

    final double[] truth = new double[] { 0.5, 2, 3, 1, 8.5, 10 };

    scorer.prepare();
    otherHalf.prepare();

    for (int i = 0; i < pred.length / 2; i++) {
      scorer.addResult(pred[i], truth[i], 1);
    }
    for (int i = pred.length / 2; i < pred.length; i++) {
      otherHalf.addResult(pred[i], truth[i], 1);
    }

    scorer.addResults(otherHalf);

    assertEquals((0.25 + 1 + 25 + 0.25 + 1) / 82.334, scorer.getScore(), 1e-4);
    assertEquals((0.25 + 1 + 25 + 0.25 + 1) / 82.334, scorer.clone().getScore(), 1e-4);
  }
}
