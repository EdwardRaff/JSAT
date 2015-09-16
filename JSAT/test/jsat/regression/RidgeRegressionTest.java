package jsat.regression;

import static org.junit.Assert.assertEquals;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.FixedProblems;
import jsat.classifiers.DataPointPair;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class RidgeRegressionTest {

  static ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  public RidgeRegressionTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testTrain_RegressionDataSet() {
    System.out.println("train");
    final Random rand = new Random(2);

    for (final RidgeRegression.SolverMode mode : RidgeRegression.SolverMode.values()) {
      final RidgeRegression regressor = new RidgeRegression(1e-9, mode);

      regressor.train(FixedProblems.getLinearRegression(400, rand));

      for (final DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, new Random(3)).getAsDPPList()) {
        final double truth = dpp.getPair();
        final double pred = regressor.regress(dpp.getDataPoint());

        final double relErr = (truth - pred) / truth;

        assertEquals(0.0, relErr, 0.05);
      }
    }
  }

  @Test
  public void testTrain_RegressionDataSet_Executor() {
    System.out.println("train");
    final Random rand = new Random(2);

    for (final RidgeRegression.SolverMode mode : RidgeRegression.SolverMode.values()) {
      final RidgeRegression regressor = new RidgeRegression(1e-9, mode);

      regressor.train(FixedProblems.getLinearRegression(400, rand), ex);

      for (final DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, new Random(3)).getAsDPPList()) {
        final double truth = dpp.getPair();
        final double pred = regressor.regress(dpp.getDataPoint());

        final double relErr = (truth - pred) / truth;

        assertEquals(0.0, relErr, 0.05);
      }
    }
  }
}
