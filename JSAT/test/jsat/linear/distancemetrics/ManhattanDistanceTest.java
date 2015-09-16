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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class ManhattanDistanceTest {

  static private ExecutorService ex;
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

  public ManhattanDistanceTest() {
  }

  @Before
  public void setUp() {
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
    expected = new double[][] { { 0, 5, 2.5, 10, }, { 5, 0, 2.5, 7, }, { 2.5, 2.5, 0.0, 8.5, }, { 10, 7, 8.5, 0, }, };
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testDist_Vec_Vec() {
    System.out.println("dist");

    final ManhattanDistance dist = new ManhattanDistance();

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
        final ManhattanDistance d = dist.clone();
        assertEquals(expected[i][j], d.dist(vecs.get(i), vecs.get(j)), 1e-12);
        assertEquals(expected[i][j], d.dist(i, j, vecs, cache), 1e-12);
        assertEquals(expected[i][j], d.dist(i, vecs.get(j), vecs, cache), 1e-12);
        assertEquals(expected[i][j], d.dist(i, vecs.get(j), dist.getQueryInfo(vecs.get(j)), vecs, cache), 1e-12);
      }
    }
  }

  @Test
  public void testMetricBound() {
    System.out.println("metricBound");
    final ManhattanDistance instance = new ManhattanDistance();
    assertTrue(instance.metricBound() > 0);
    assertTrue(Double.isInfinite(instance.metricBound()));
  }

  @Test
  public void testMetricProperties() {
    System.out.println("isSymmetric");
    final ManhattanDistance instance = new ManhattanDistance();
    assertTrue(instance.isSymmetric());
    assertTrue(instance.isSubadditive());
    assertTrue(instance.isIndiscemible());
  }

}
