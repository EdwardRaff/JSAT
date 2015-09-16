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

/**
 *
 * @author Edward Raff
 */
public class PassiveAggressiveTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public PassiveAggressiveTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of train method, of class PassiveAggressive.
   */
  @Test
  public void testTrain_RegressionDataSet() {
    System.out.println("train");
    final Random rand = new Random(123);

    for (final PassiveAggressive.Mode mode : PassiveAggressive.Mode.values()) {
      final PassiveAggressive pa = new PassiveAggressive();
      pa.setMode(mode);
      pa.setEps(0.00001);
      pa.setEpochs(10);
      pa.setC(20);
      pa.train(FixedProblems.getLinearRegression(400, rand));

      for (final DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList()) {
        final double truth = dpp.getPair();
        final double pred = pa.regress(dpp.getDataPoint());

        final double relErr = (truth - pred) / truth;
        assertEquals(0.0, relErr, 0.1);// Give it a decent wiggle room b/c of
                                       // regularization
      }
    }
  }

  /**
   * Test of trainC method, of class PassiveAggressive.
   */
  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");
    final ClassificationDataSet train = FixedProblems.get2ClassLinear(400, new Random());

    for (final PassiveAggressive.Mode mode : PassiveAggressive.Mode.values()) {
      final PassiveAggressive pa = new PassiveAggressive();
      pa.setMode(mode);
      pa.trainC(train);

      final ClassificationDataSet test = FixedProblems.get2ClassLinear(400, new Random());

      for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
        assertEquals(dpp.getPair().longValue(), pa.classify(dpp.getDataPoint()).mostLikely());
      }
    }
  }

}
