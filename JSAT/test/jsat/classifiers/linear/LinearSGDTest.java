/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.linear;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.SquaredLoss;
import jsat.math.optimization.stochastic.AdaGrad;
import jsat.math.optimization.stochastic.GradientUpdater;
import jsat.math.optimization.stochastic.RMSProp;
import jsat.math.optimization.stochastic.SimpleSGD;
import jsat.regression.RegressionDataSet;
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
public class LinearSGDTest {

  public LinearSGDTest() {
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

  static boolean[] useBiasOptions = new boolean[]{true, false};
  static GradientUpdater[] updaters = new GradientUpdater[]{new SimpleSGD(), new AdaGrad(), new RMSProp()};

  @Test
  public void testClassifyBinary() {
    System.out.println("binary classifiation");

    for (boolean useBias : useBiasOptions) {
      for (GradientUpdater gu : updaters) {
        LinearSGD linearsgd = new LinearSGD(new HingeLoss(), 1e-4, 1e-5);
        linearsgd.setUseBias(useBias);
        linearsgd.setGradientUpdater(gu);

        ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());

        linearsgd.trainC(train);

        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

        for (DataPointPair<Integer> dpp : test.getAsDPPList()) {
          assertEquals(dpp.getPair().longValue(), linearsgd.classify(dpp.getDataPoint()).mostLikely());
        }
      }
    }
  }

  @Test
  public void testClassifyMulti() {
    System.out.println("multi class classification");
    for (boolean useBias : useBiasOptions) {
      for (GradientUpdater gu : updaters) {
        LinearSGD linearsgd = new LinearSGD(new HingeLoss(), 1e-4, 1e-5);
        linearsgd.setUseBias(useBias);
        linearsgd.setGradientUpdater(gu);

        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());

        linearsgd.trainC(train);

        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());

        for (DataPointPair<Integer> dpp : test.getAsDPPList()) {
          assertEquals(dpp.getPair().longValue(), linearsgd.classify(dpp.getDataPoint()).mostLikely());
        }
      }
    }
  }

  @Test
  public void testRegression() {
    System.out.println("regression");
    for (boolean useBias : useBiasOptions) {
      for (GradientUpdater gu : updaters) {
        LinearSGD linearsgd = new LinearSGD(new SquaredLoss(), 0.0, 0.0);
        linearsgd.setUseBias(useBias);
        linearsgd.setGradientUpdater(gu);

        //SGD needs more iterations/data to learn a really close fit
        RegressionDataSet train = FixedProblems.getLinearRegression(10000, new Random());

        linearsgd.setEpochs(50);
        if (!(gu instanceof SimpleSGD))//the others need a higher learning rate than the default
        {
          linearsgd.setEta(0.5);
          linearsgd.setEpochs(100);//more iters b/c RMSProp probably isn't the best for this overly simple problem
        }
        linearsgd.train(train);

        RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());

        for (DataPointPair<Double> dpp : test.getAsDPPList()) {
          double truth = dpp.getPair();
          double pred = linearsgd.regress(dpp.getDataPoint());
          double relErr = (truth - pred) / truth;
          assertEquals(0, relErr, 0.1);
        }
      }
    }
  }
}
