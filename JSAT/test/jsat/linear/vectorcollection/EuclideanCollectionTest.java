/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.linear.vectorcollection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class EuclideanCollectionTest {

  static List<VectorCollectionFactory<Vec>> collectionFactories;

  @BeforeClass
  public static void setUpClass() {
    collectionFactories = new ArrayList<VectorCollectionFactory<Vec>>();
    collectionFactories.add(new EuclideanCollection.EuclideanCollectionFactory<Vec>());
  }

  @AfterClass
  public static void tearDownClass() {
  }

  public EuclideanCollectionTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testSearch_Vec_double() {
    System.out.println("search");
    final Random rand = new XORWOW(123);

    final VectorArray<Vec> vecCol = new VectorArray<Vec>(new EuclideanDistance());
    for (int i = 0; i < 250; i++) {
      vecCol.add(Vec.random(3, rand));
    }

    final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

    for (final VectorCollectionFactory<Vec> factory : collectionFactories) {
      VectorCollection<Vec> collection0 = factory.getVectorCollection(vecCol, new EuclideanDistance());
      VectorCollection<Vec> collection1 = factory.getVectorCollection(vecCol, new EuclideanDistance(), ex);

      collection0 = collection0.clone();
      collection1 = collection1.clone();

      for (int iters = 0; iters < 10; iters++) {
        for (final double range : new double[] { 0.25, 0.5, 0.75, 2.0 }) {
          final int randIndex = rand.nextInt(vecCol.size());

          final List<? extends VecPaired<Vec, Double>> foundTrue = vecCol.search(vecCol.get(randIndex), range);
          final List<? extends VecPaired<Vec, Double>> foundTest0 = collection0.search(vecCol.get(randIndex), range);
          final List<? extends VecPaired<Vec, Double>> foundTest1 = collection1.search(vecCol.get(randIndex), range);

          final VectorArray<VecPaired<Vec, Double>> testSearch0 = new VectorArray<VecPaired<Vec, Double>>(
              new EuclideanDistance(), foundTest0);
          assertEquals(factory.getClass().getName() + " failed", foundTrue.size(), foundTest0.size());
          for (final Vec v : foundTrue) {
            final List<? extends VecPaired<VecPaired<Vec, Double>, Double>> nn = testSearch0.search(v, 1);
            assertTrue(factory.getClass().getName() + " failed", nn.get(0).equals(v, 1e-13));
          }

          final VectorArray<VecPaired<Vec, Double>> testSearch1 = new VectorArray<VecPaired<Vec, Double>>(
              new EuclideanDistance(), foundTest1);
          assertEquals(factory.getClass().getName() + " failed", foundTrue.size(), foundTest1.size());
          for (final Vec v : foundTrue) {
            final List<? extends VecPaired<VecPaired<Vec, Double>, Double>> nn = testSearch1.search(v, 1);
            assertTrue(factory.getClass().getName() + " failed", nn.get(0).equals(v, 1e-13));
          }

        }
      }
    }

    ex.shutdownNow();
  }

  @Test
  public void testSearch_Vec_int() {
    System.out.println("search");
    final Random rand = new XORWOW(123);

    final VectorArray<Vec> vecCol = new VectorArray<Vec>(new EuclideanDistance());
    for (int i = 0; i < 250; i++) {
      vecCol.add(Vec.random(3, rand));
    }

    for (final VectorCollectionFactory<Vec> factory : collectionFactories) {
      final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

      VectorCollection<Vec> collection0 = factory.getVectorCollection(vecCol, new EuclideanDistance());
      VectorCollection<Vec> collection1 = factory.getVectorCollection(vecCol, new EuclideanDistance(), ex);

      collection0 = collection0.clone();
      collection1 = collection1.clone();

      ex.shutdownNow();

      for (int iters = 0; iters < 10; iters++) {
        for (final int neighbours : new int[] { 1, 2, 4, 10, 20 }) {
          final int randIndex = rand.nextInt(vecCol.size());

          final List<? extends VecPaired<Vec, Double>> foundTrue = vecCol.search(vecCol.get(randIndex), neighbours);
          final List<? extends VecPaired<Vec, Double>> foundTest0 = collection0.search(vecCol.get(randIndex),
              neighbours);
          final List<? extends VecPaired<Vec, Double>> foundTest1 = collection1.search(vecCol.get(randIndex),
              neighbours);

          final VectorArray<VecPaired<Vec, Double>> testSearch0 = new VectorArray<VecPaired<Vec, Double>>(
              new EuclideanDistance(), foundTest0);
          assertEquals(factory.getClass().getName() + " failed", foundTrue.size(), foundTest0.size());
          for (final Vec v : foundTrue) {
            final List<? extends VecPaired<VecPaired<Vec, Double>, Double>> nn = testSearch0.search(v, 1);
            assertTrue(factory.getClass().getName() + " failed", nn.get(0).equals(v, 1e-13));
          }

          final VectorArray<VecPaired<Vec, Double>> testSearch1 = new VectorArray<VecPaired<Vec, Double>>(
              new EuclideanDistance(), foundTest1);
          assertEquals(factory.getClass().getName() + " failed " + neighbours, foundTrue.size(), foundTest1.size());
          for (final Vec v : foundTrue) {
            final List<? extends VecPaired<VecPaired<Vec, Double>, Double>> nn = testSearch1.search(v, 1);
            assertTrue(factory.getClass().getName() + " failed " + neighbours, nn.get(0).equals(v, 1e-13));
          }

        }
      }
    }

  }
}
