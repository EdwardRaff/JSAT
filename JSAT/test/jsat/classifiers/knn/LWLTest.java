/*
 * Copyright (C) 2015 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.classifiers.knn;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.bayesian.NaiveBayesUpdateable;
import jsat.classifiers.svm.DCDs;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.regression.Regressor;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class LWLTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public LWLTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testClone() {
    System.out.println("clone");

    LWL instance = new LWL(new NaiveBayesUpdateable(), 15, new EuclideanDistance());

    final ClassificationDataSet t1 = FixedProblems.getSimpleKClassLinear(100, 3);
    final ClassificationDataSet t2 = FixedProblems.getSimpleKClassLinear(100, 6);

    instance = instance.clone();

    instance.trainC(t1);

    final LWL result = instance.clone();
    for (int i = 0; i < t1.getSampleSize(); i++) {
      assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
    }
    result.trainC(t2);

    for (int i = 0; i < t1.getSampleSize(); i++) {
      assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());
    }

    for (int i = 0; i < t2.getSampleSize(); i++) {
      assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());
    }

  }

  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");

    final LWL instance = new LWL(new NaiveBayesUpdateable(), 15, new EuclideanDistance());

    final ClassificationDataSet train = FixedProblems.getCircles(1000, 1.0, 10.0, 100.0);
    final ClassificationDataSet test = FixedProblems.getCircles(100, 1.0, 10.0, 100.0);

    final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
    cme.evaluateTestSet(test);

    assertTrue(cme.getErrorRate() <= 0.001);

  }

  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");

    final LWL instance = new LWL(new NaiveBayesUpdateable(), 15, new EuclideanDistance());

    final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

    final ClassificationDataSet train = FixedProblems.getCircles(1000, 1.0, 10.0, 100.0);
    final ClassificationDataSet test = FixedProblems.getCircles(100, 1.0, 10.0, 100.0);

    final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
    cme.evaluateTestSet(test);

    assertTrue(cme.getErrorRate() <= 0.001);

    ex.shutdownNow();
  }

  @Test
  public void testTrainC_RegressionDataSet() {
    System.out.println("train");

    final LWL instance = new LWL((Regressor) new DCDs(), 30, new EuclideanDistance());

    final RegressionDataSet train = FixedProblems.getLinearRegression(1000, new XORWOW());
    final RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW());

    final RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train);
    rme.evaluateTestSet(test);

    assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.25);

  }

  @Test
  public void testTrainC_RegressionDataSet_ExecutorService() {
    System.out.println("train");

    final LWL instance = new LWL((Regressor) new DCDs(), 15, new EuclideanDistance());

    final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

    final RegressionDataSet train = FixedProblems.getLinearRegression(1000, new XORWOW());
    final RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW());

    final RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train, ex);
    rme.evaluateTestSet(test);

    assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.25);

    ex.shutdownNow();
  }

}
