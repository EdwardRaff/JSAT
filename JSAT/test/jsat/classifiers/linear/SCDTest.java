package jsat.classifiers.linear;

import static org.junit.Assert.assertEquals;

import java.util.Random;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import jsat.lossfunctions.LogisticLoss;
import jsat.lossfunctions.SquaredLoss;

/**
 *
 * @author Edward Raff
 */
public class SCDTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public SCDTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of train method, of class SCD.
   */
  @Test
  public void testTrain_RegressionDataSet() {
    System.out.println("train");
    final Random rand = new Random(123);

    final SCD scd = new SCD(new SquaredLoss(), 1e-6, 1000);// needs more iters
                                                           // for regression
    scd.train(FixedProblems.getLinearRegression(500, rand));

    for (final DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList()) {
      final double truth = dpp.getPair();
      final double pred = scd.regress(dpp.getDataPoint());

      final double relErr = (truth - pred) / truth;
      assertEquals(0.0, relErr, 0.1);// Give it a decent wiggle room b/c of
                                     // regularization
    }
  }

  /**
   * Test of trainC method, of class SCD.
   */
  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");

    final ClassificationDataSet train = FixedProblems.get2ClassLinear(400, new Random());

    final SCD scd = new SCD(new LogisticLoss(), 1e-6, 100);
    scd.trainC(train);

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(400, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), scd.classify(dpp.getDataPoint()).mostLikely());
    }

  }
}
