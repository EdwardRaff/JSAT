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
package jsat.classifiers.trees;

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
import jsat.datatransform.DataTransformProcess;
import jsat.datatransform.NumericalToHistogram;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class DecisionTreeTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public DecisionTreeTest() {
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
    for (final boolean useCatFeatures : new boolean[] { true, false }) {
      DecisionTree instance = new DecisionTree();

      final ClassificationDataSet t1 = FixedProblems.getSimpleKClassLinear(1000, 3);
      final ClassificationDataSet t2 = FixedProblems.getSimpleKClassLinear(1000, 2);
      if (useCatFeatures) {
        t1.applyTransform(new NumericalToHistogram(t1));
        t2.applyTransform(new NumericalToHistogram(t2));
      }

      instance = instance.clone();

      instance.trainC(t1);

      final DecisionTree result = instance.clone();
      int errors = 0;
      for (int i = 0; i < t1.getSampleSize(); i++) {
        errors += Math.abs(t1.getDataPointCategory(i) - result.classify(t1.getDataPoint(i)).mostLikely());
      }
      assertTrue(errors < 50);
      result.trainC(t2);

      errors = 0;
      for (int i = 0; i < t1.getSampleSize(); i++) {
        errors += Math.abs(t1.getDataPointCategory(i) - instance.classify(t1.getDataPoint(i)).mostLikely());
      }
      assertTrue(errors < 50);

      errors = 0;
      for (int i = 0; i < t2.getSampleSize(); i++) {
        errors += Math.abs(t2.getDataPointCategory(i) - result.classify(t2.getDataPoint(i)).mostLikely());
      }
      assertTrue(errors < 50);

    }
  }

  @Test
  public void testTrainC_ClassificationDataSet() {
    System.out.println("trainC");

    for (final TreePruner.PruningMethod pruneMethod : TreePruner.PruningMethod.values()) {
      for (final DecisionStump.NumericHandlingC numericHandling : DecisionStump.NumericHandlingC.values()) {
        for (final ImpurityScore.ImpurityMeasure gainMethod : ImpurityScore.ImpurityMeasure.values()) {
          for (final boolean useCatFeatures : new boolean[] { true, false }) {
            final DecisionTree instance = new DecisionTree();
            instance.setGainMethod(gainMethod);
            instance.setTestProportion(0.3);
            instance.setNumericHandling(numericHandling);
            instance.setPruningMethod(pruneMethod);
            final ClassificationDataSet train = FixedProblems.getCircles(5000, 1.0, 10.0, 100.0);
            final ClassificationDataSet test = FixedProblems.getCircles(100, 1.0, 10.0, 100.0);
            final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
            if (useCatFeatures) {
              cme.setDataTransformProcess(
                  new DataTransformProcess(new NumericalToHistogram.NumericalToHistogramTransformFactory()));
            }
            cme.evaluateTestSet(test);
            assertTrue(cme.getErrorRate() <= 0.05);
          }
        }
      }
    }
  }

  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");

    for (final TreePruner.PruningMethod pruneMethod : TreePruner.PruningMethod.values()) {
      for (final DecisionStump.NumericHandlingC numericHandling : DecisionStump.NumericHandlingC.values()) {
        for (final ImpurityScore.ImpurityMeasure gainMethod : ImpurityScore.ImpurityMeasure.values()) {
          for (final boolean useCatFeatures : new boolean[] { true, false }) {
            final DecisionTree instance = new DecisionTree();
            instance.setGainMethod(gainMethod);
            instance.setTestProportion(0.3);
            instance.setNumericHandling(numericHandling);
            instance.setPruningMethod(pruneMethod);
            final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
            final ClassificationDataSet train = FixedProblems.getCircles(5000, 1.0, 10.0, 100.0);
            final ClassificationDataSet test = FixedProblems.getCircles(100, 1.0, 10.0, 100.0);
            final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
            if (useCatFeatures) {
              cme.setDataTransformProcess(
                  new DataTransformProcess(new NumericalToHistogram.NumericalToHistogramTransformFactory()));
            }
            cme.evaluateTestSet(test);
            assertTrue(cme.getErrorRate() <= 0.05);
            ex.shutdownNow();
          }
        }
      }
    }
  }

  @Test
  public void testTrainC_RegressionDataSet() {
    System.out.println("train");
    for (final TreePruner.PruningMethod pruneMethod : TreePruner.PruningMethod.values()) {
      for (final DecisionStump.NumericHandlingC numericHandling : DecisionStump.NumericHandlingC.values()) {
        for (final ImpurityScore.ImpurityMeasure gainMethod : ImpurityScore.ImpurityMeasure.values()) {
          for (final boolean useCatFeatures : new boolean[] { true, false }) {
            final DecisionTree instance = new DecisionTree();
            instance.setGainMethod(gainMethod);
            instance.setTestProportion(0.3);
            instance.setNumericHandling(numericHandling);
            instance.setPruningMethod(pruneMethod);
            final RegressionDataSet train = FixedProblems.getLinearRegression(3000, new XORWOW());
            final RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW());
            final RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train);
            if (useCatFeatures) {
              rme.setDataTransformProcess(
                  new DataTransformProcess(new NumericalToHistogram.NumericalToHistogramTransformFactory()));
            }
            rme.evaluateTestSet(test);
            assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 3);
          }
        }
      }
    }
  }

  @Test
  public void testTrainC_RegressionDataSet_ExecutorService() {
    System.out.println("train");

    for (final TreePruner.PruningMethod pruneMethod : TreePruner.PruningMethod.values()) {
      for (final DecisionStump.NumericHandlingC numericHandling : DecisionStump.NumericHandlingC.values()) {
        for (final ImpurityScore.ImpurityMeasure gainMethod : ImpurityScore.ImpurityMeasure.values()) {
          for (final boolean useCatFeatures : new boolean[] { true, false }) {
            final DecisionTree instance = new DecisionTree();
            instance.setGainMethod(gainMethod);
            instance.setTestProportion(0.3);
            instance.setNumericHandling(numericHandling);
            instance.setPruningMethod(pruneMethod);
            final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
            final RegressionDataSet train = FixedProblems.getLinearRegression(3000, new XORWOW());
            final RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW());
            final RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train, ex);
            if (useCatFeatures) {
              rme.setDataTransformProcess(
                  new DataTransformProcess(new NumericalToHistogram.NumericalToHistogramTransformFactory()));
            }
            rme.evaluateTestSet(test);
            assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 3);
            ex.shutdownNow();
          }
        }
      }
    }
  }

}
