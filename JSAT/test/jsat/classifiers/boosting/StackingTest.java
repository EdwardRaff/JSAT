package jsat.classifiers.boosting;

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
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.OneVSAll;
import jsat.classifiers.linear.LinearBatch;
import jsat.classifiers.linear.LogisticRegressionDCD;
import jsat.lossfunctions.AbsoluteLoss;
import jsat.lossfunctions.HuberLoss;
import jsat.lossfunctions.SoftmaxLoss;
import jsat.lossfunctions.SquaredLoss;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class StackingTest {

  static ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  public StackingTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testClassifyBinary() {
    System.out.println("binary classifiation");

    Stacking stacking = new Stacking(new LogisticRegressionDCD(), new LinearBatch(new SoftmaxLoss(), 1e-15),
        new LinearBatch(new SoftmaxLoss(), 100), new LinearBatch(new SoftmaxLoss(), 1e10));

    final ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());

    stacking = stacking.clone();
    stacking.trainC(train);
    stacking = stacking.clone();

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), stacking.classify(dpp.getDataPoint()).mostLikely());
    }
  }

  @Test
  public void testClassifyBinaryMT() {
    System.out.println("binary classifiation MT");

    Stacking stacking = new Stacking(new LogisticRegressionDCD(), new LinearBatch(new SoftmaxLoss(), 1e-15),
        new LinearBatch(new SoftmaxLoss(), 100), new LinearBatch(new SoftmaxLoss(), 1e10));

    final ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());

    stacking = stacking.clone();
    stacking.trainC(train, ex);
    stacking = stacking.clone();

    final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), stacking.classify(dpp.getDataPoint()).mostLikely());
    }
  }

  @Test
  public void testClassifyMulti() {
    Stacking stacking = new Stacking(new OneVSAll(new LogisticRegressionDCD(), true),
        new LinearBatch(new SoftmaxLoss(), 1e-15), new LinearBatch(new SoftmaxLoss(), 100),
        new LinearBatch(new SoftmaxLoss(), 1e10));

    final ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());

    stacking = stacking.clone();
    stacking.trainC(train);
    stacking = stacking.clone();

    final ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), stacking.classify(dpp.getDataPoint()).mostLikely());
    }
  }

  @Test
  public void testClassifyMultiMT() {
    System.out.println("multi class classification MT");

    Stacking stacking = new Stacking(new OneVSAll(new LogisticRegressionDCD(), true),
        new LinearBatch(new SoftmaxLoss(), 1e-15), new LinearBatch(new SoftmaxLoss(), 100),
        new LinearBatch(new SoftmaxLoss(), 1e10));

    final ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());

    stacking = stacking.clone();
    stacking.trainC(train, ex);
    stacking = stacking.clone();

    final ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());

    for (final DataPointPair<Integer> dpp : test.getAsDPPList()) {
      assertEquals(dpp.getPair().longValue(), stacking.classify(dpp.getDataPoint()).mostLikely());
    }
  }

  @Test
  public void testRegression() {
    System.out.println("regression");

    Stacking stacking = new Stacking((Regressor) new LinearBatch(new AbsoluteLoss(), 1e-10),
        new LinearBatch(new SquaredLoss(), 1e-15), new LinearBatch(new AbsoluteLoss(), 100),
        new LinearBatch(new HuberLoss(), 1e1));
    final RegressionDataSet train = FixedProblems.getLinearRegression(500, new Random());

    stacking = stacking.clone();
    stacking.train(train);
    stacking = stacking.clone();

    final RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());

    for (final DataPointPair<Double> dpp : test.getAsDPPList()) {
      final double truth = dpp.getPair();
      final double pred = stacking.regress(dpp.getDataPoint());
      final double relErr = (truth - pred) / truth;
      assertEquals(0, relErr, 0.1);
    }
  }

  @Test
  public void testRegressionMT() {
    System.out.println("regression MT");

    Stacking stacking = new Stacking((Regressor) new LinearBatch(new AbsoluteLoss(), 1e-10),
        new LinearBatch(new SquaredLoss(), 1e-15), new LinearBatch(new AbsoluteLoss(), 100),
        new LinearBatch(new HuberLoss(), 1e1));
    final RegressionDataSet train = FixedProblems.getLinearRegression(500, new Random());

    stacking = stacking.clone();
    stacking.train(train, ex);
    stacking = stacking.clone();

    final RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());

    for (final DataPointPair<Double> dpp : test.getAsDPPList()) {
      final double truth = dpp.getPair();
      final double pred = stacking.regress(dpp.getDataPoint());
      final double relErr = (truth - pred) / truth;
      assertEquals(0, relErr, 0.01);
    }
  }

}
