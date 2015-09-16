package jsat.classifiers.linear;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
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
public class SMIDASTest {

  public SMIDASTest() {
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
   * Test of trainC method, of class SMIDAS.
   */
  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");

    ClassificationDataSet train = FixedProblems.get2ClassLinear(400, new Random());

    SMIDAS smidas = new SMIDAS(0.1);
    smidas.setLoss(StochasticSTLinearL1.Loss.LOG);
    smidas.trainC(train);

    ClassificationDataSet test = FixedProblems.get2ClassLinear(400, new Random());

    for (DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), smidas.classify(dpp.getDataPoint()).mostLikely());
    }

  }

  /**
   * Test of train method, of class SMIDAS.
   */
  @Test
  public void testTrain_RegressionDataSet() {
    System.out.println("train");
    Random rand = new Random(123);

    SMIDAS smidas = new SMIDAS(0.02);
    smidas.setMinScaled(-1);
    smidas.setLoss(StochasticSTLinearL1.Loss.SQUARED);
    smidas.train(FixedProblems.getLinearRegression(500, rand));

    for (DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList()) {
      double truth = dpp.getPair();
      double pred = smidas.regress(dpp.getDataPoint());

      double relErr = (truth - pred) / truth;
      assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
    }
  }
}
