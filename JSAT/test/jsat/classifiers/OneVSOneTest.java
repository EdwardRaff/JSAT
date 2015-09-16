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
package jsat.classifiers;

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
import jsat.classifiers.svm.DCDs;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class OneVSOneTest {

  @BeforeClass
  public static void setUpClass() {
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public OneVSOneTest() {
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

    final ClassificationDataSet t1 = FixedProblems.getSimpleKClassLinear(1000, 7);
    final ClassificationDataSet t2 = FixedProblems.getSimpleKClassLinear(1000, 9);

    OneVSOne instance = new OneVSOne(new DCDs());

    instance = instance.clone();

    instance.trainC(t1);

    final OneVSOne result = instance.clone();
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
    for (final boolean conc : new boolean[] { true, false }) {
      final OneVSOne instance = new OneVSOne(new DCDs(), conc);

      final ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(1000, 7);
      final ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(100, 7);

      final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
      cme.evaluateTestSet(test);

      assertTrue(cme.getErrorRate() <= 0.001);
    }
  }

  @Test
  public void testTrainC_ClassificationDataSet_ExecutorService() {
    System.out.println("trainC");
    for (final boolean conc : new boolean[] { true, false }) {
      final OneVSOne instance = new OneVSOne(new DCDs(), conc);

      final ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(1000, 7);
      final ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(100, 7);

      final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

      final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
      cme.evaluateTestSet(test);

      assertTrue(cme.getErrorRate() <= 0.001);
    }
  }
}
