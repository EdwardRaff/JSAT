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
package jsat.datatransform.kernel;

import static org.junit.Assert.assertEquals;
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
import jsat.classifiers.Classifier;
import jsat.classifiers.svm.DCDs;
import jsat.datatransform.DataModelPipeline;
import jsat.distributions.kernels.RBFKernel;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class KernelPCATest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  // Test uses Transform to solve a problem that is not linearly seprable in the
  // original space
  public KernelPCATest() {
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

    DataModelPipeline instance = new DataModelPipeline((Classifier) new DCDs(),
        new KernelPCA.KernelPCATransformFactory(new RBFKernel(0.5), 20, 100, Nystrom.SamplingMethod.KMEANS));

    final ClassificationDataSet t1 = FixedProblems.getInnerOuterCircle(500, new XORWOW());
    final ClassificationDataSet t2 = FixedProblems.getInnerOuterCircle(500, new XORWOW(), 2.0, 10.0);

    instance = instance.clone();

    instance.trainC(t1);

    final DataModelPipeline result = instance.clone();

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

    for (final Nystrom.SamplingMethod sampMethod : Nystrom.SamplingMethod.values()) {
      final DataModelPipeline instance = new DataModelPipeline((Classifier) new DCDs(),
          new KernelPCA.KernelPCATransformFactory(new RBFKernel(0.5), 20, 100, sampMethod));

      final ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, new XORWOW());
      final ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, new XORWOW());

      final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
      cme.evaluateTestSet(test);

      assertEquals(0, cme.getErrorRate(), 0.0);
    }

  }

  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");

    final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

    for (final Nystrom.SamplingMethod sampMethod : Nystrom.SamplingMethod.values()) {
      final DataModelPipeline instance = new DataModelPipeline((Classifier) new DCDs(),
          new KernelPCA.KernelPCATransformFactory(new RBFKernel(0.5), 20, 100, sampMethod));

      final ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, new XORWOW());
      final ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, new XORWOW());

      final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
      cme.evaluateTestSet(test);

      assertEquals(0, cme.getErrorRate(), 0.0);
    }
    ex.shutdownNow();

  }

}
