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
package jsat.linear.distancemetrics;

import static jsat.TestTools.assertEqualsRelDiff;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.distributions.multivariate.NormalM;
import jsat.linear.ConstantVector;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.SingularValueDecomposition;
import jsat.linear.Vec;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class MahalanobisDistanceTest {

  static private ExecutorService ex;
  static private Matrix trueCov;
  static private Vec zero;
  static private Vec ones;
  static private Vec half;
  static private Vec inc;

  static private List<Vec> vecs;

  static private double[][] expected;

  @BeforeClass
  public static void setUpClass() {
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  public MahalanobisDistanceTest() {
  }

  @Before
  public void setUp() {
    trueCov = new DenseMatrix(5, 5);

    final Random rand = new XORWOW();
    for (int i = 0; i < trueCov.rows(); i++) {
      for (int j = 0; j < trueCov.cols(); j++) {
        trueCov.set(i, j, rand.nextDouble());
      }
    }
    trueCov = trueCov.multiplyTranspose(trueCov);// guaranteed Positive Semi
                                                 // Definite now

    zero = new DenseVector(5);

    ones = new DenseVector(5);
    ones.mutableAdd(1.0);

    half = new DenseVector(5);
    half.mutableAdd(0.5);

    inc = new DenseVector(5);
    for (int i = 0; i < inc.length(); i++) {
      inc.set(i, i);
    }

    vecs = Arrays.asList(zero, ones, half, inc);
    final Matrix trueInv = new SingularValueDecomposition(trueCov.clone()).getPseudoInverse();

    expected = new double[4][4];
    for (int i = 0; i < expected.length; i++) {
      final Vec vi = vecs.get(i);
      for (int j = 0; j < expected.length; j++) {
        final Vec vj = vecs.get(j);
        final Vec dif = vi.subtract(vj);
        expected[i][j] = Math.sqrt(dif.dot(trueInv.multiply(dif)));
      }
    }
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testDist_Vec_Vec() {
    System.out.println("dist");

    final NormalM normal = new NormalM(new ConstantVector(0.0, 5), trueCov.clone());
    final MahalanobisDistance dist = new MahalanobisDistance();
    dist.train(normal.sample(1000, new XORWOW()));

    final List<Double> cache = dist.getAccelerationCache(vecs);
    final List<Double> cache2 = dist.getAccelerationCache(vecs, ex);
    if (cache != null) {
      assertEquals(cache.size(), cache2.size());
      for (int i = 0; i < cache.size(); i++) {
        assertEquals(cache.get(i), cache2.get(i), 0.0);
      }
      assertTrue(dist.supportsAcceleration());
    } else {
      assertNull(cache2);
      assertFalse(dist.supportsAcceleration());
    }

    try {
      dist.dist(half, new DenseVector(half.length() + 1));
      fail("Distance between vecs should have erred");
    } catch (final Exception ex) {

    }

    for (int i = 0; i < vecs.size(); i++) {
      for (int j = 0; j < vecs.size(); j++) {
        final MahalanobisDistance d = dist.clone();
        assertEqualsRelDiff(expected[i][j], d.dist(vecs.get(i), vecs.get(j)), 1e-1);
        assertEqualsRelDiff(expected[i][j], d.dist(i, j, vecs, cache), 1e-1);
        assertEqualsRelDiff(expected[i][j], d.dist(i, vecs.get(j), vecs, cache), 1e-1);
        assertEqualsRelDiff(expected[i][j], d.dist(i, vecs.get(j), dist.getQueryInfo(vecs.get(j)), vecs, cache), 1e-1);
      }
    }
  }

  @Test
  public void testDist_Vec_Vec_ExecutorService() {
    System.out.println("dist");

    final NormalM normal = new NormalM(new ConstantVector(0.0, 5), trueCov.clone());
    final MahalanobisDistance dist = new MahalanobisDistance();
    dist.train(normal.sample(1000, new XORWOW()), ex);

    final List<Double> cache = dist.getAccelerationCache(vecs);
    final List<Double> cache2 = dist.getAccelerationCache(vecs, ex);
    if (cache != null) {
      assertEquals(cache.size(), cache2.size());
      for (int i = 0; i < cache.size(); i++) {
        assertEquals(cache.get(i), cache2.get(i), 0.0);
      }
      assertTrue(dist.supportsAcceleration());
    } else {
      assertNull(cache2);
      assertFalse(dist.supportsAcceleration());
    }

    try {
      dist.dist(half, new DenseVector(half.length() + 1));
      fail("Distance between vecs should have erred");
    } catch (final Exception ex) {

    }

    for (int i = 0; i < vecs.size(); i++) {
      for (int j = 0; j < vecs.size(); j++) {
        final MahalanobisDistance d = dist.clone();
        assertEqualsRelDiff(expected[i][j], d.dist(vecs.get(i), vecs.get(j)), 1e-1);
        assertEqualsRelDiff(expected[i][j], d.dist(i, j, vecs, cache), 1e-1);
        assertEqualsRelDiff(expected[i][j], d.dist(i, vecs.get(j), vecs, cache), 1e-1);
        assertEqualsRelDiff(expected[i][j], d.dist(i, vecs.get(j), dist.getQueryInfo(vecs.get(j)), vecs, cache), 1e-1);
      }
    }
  }

  @Test
  public void testMetricBound() {
    System.out.println("metricBound");
    final EuclideanDistance instance = new EuclideanDistance();
    assertTrue(instance.metricBound() > 0);
    assertTrue(Double.isInfinite(instance.metricBound()));
  }

  @Test
  public void testMetricProperties() {
    System.out.println("isSymmetric");
    final EuclideanDistance instance = new EuclideanDistance();
    assertTrue(instance.isSymmetric());
    assertTrue(instance.isSubadditive());
    assertTrue(instance.isIndiscemible());
  }

}
