/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering.hierarchical;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.dissimilarity.SingleLinkDissimilarity;
import jsat.distributions.Uniform;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;

/**
 *
 * @author Edward Raff
 */
public class SimpleHACTest {
  /*
   * README: KMeans is a very heuristic algorithm, so its not easy to make a
   * test where we are very sure it will get the correct awnser. That is why
   * only 2 of the methods are tested [ Using KPP, becase random seed selection
   * still isnt consistent enough]
   *
   */

  static private SimpleHAC simpleHAC;
  static private SimpleDataSet easyData10;
  static private ExecutorService ex;

  @BeforeClass
  public static void setUpClass() throws Exception {
    simpleHAC = new SimpleHAC(new SingleLinkDissimilarity(new EuclideanDistance()));
    final GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new Random(12), 2, 5);
    easyData10 = gdg.generateData(30);// HAC is O(n^3), so we make the data set
                                      // a good deal smaller
    ex = Executors.newFixedThreadPool(10);
  }

  @AfterClass
  public static void tearDownClass() throws Exception {
    ex.shutdown();
  }

  public SimpleHACTest() {
  }

  @Before
  public void setUp() {

  }

  @Test
  public void testCluster_DataSet() {
    System.out.println("cluster(dataset)");
    final List<List<DataPoint>> clusters = simpleHAC.cluster(easyData10);
    assertEquals(10, clusters.size());
    final Set<Integer> seenBefore = new IntSet();
    for (final List<DataPoint> cluster : clusters) {
      final int thisClass = cluster.get(0).getCategoricalValue(0);
      assertFalse(seenBefore.contains(thisClass));
      for (final DataPoint dp : cluster) {
        assertEquals(thisClass, dp.getCategoricalValue(0));
      }
    }
  }

  @Test
  public void testCluster_DataSet_ExecutorService() {
    System.out.println("cluster(dataset, ExecutorService)");
    final List<List<DataPoint>> clusters = simpleHAC.cluster(easyData10, ex);
    assertEquals(10, clusters.size());
    final Set<Integer> seenBefore = new IntSet();
    for (final List<DataPoint> cluster : clusters) {
      final int thisClass = cluster.get(0).getCategoricalValue(0);
      assertFalse(seenBefore.contains(thisClass));
      for (final DataPoint dp : cluster) {
        assertEquals(thisClass, dp.getCategoricalValue(0));
      }
    }
  }

  /**
   * Test of cluster method, of class KMeans.
   */
  @Test
  public void testCluster_DataSet_int() {
    System.out.println("cluster(dataset, int)");
    final List<List<DataPoint>> clusters = simpleHAC.cluster(easyData10, 10);
    assertEquals(10, clusters.size());
    final Set<Integer> seenBefore = new IntSet();
    for (final List<DataPoint> cluster : clusters) {
      final int thisClass = cluster.get(0).getCategoricalValue(0);
      assertFalse(seenBefore.contains(thisClass));
      for (final DataPoint dp : cluster) {
        assertEquals(thisClass, dp.getCategoricalValue(0));
      }
    }
  }

  /**
   * Test of cluster method, of class KMeans.
   */
  @Test
  public void testCluster_DataSet_int_ExecutorService() {
    System.out.println("cluster(dataset, int, ExecutorService)");
    final List<List<DataPoint>> clusters = simpleHAC.cluster(easyData10, 10, ex);
    assertEquals(10, clusters.size());
    final Set<Integer> seenBefore = new IntSet();
    for (final List<DataPoint> cluster : clusters) {
      final int thisClass = cluster.get(0).getCategoricalValue(0);
      assertFalse(seenBefore.contains(thisClass));
      for (final DataPoint dp : cluster) {
        assertEquals(thisClass, dp.getCategoricalValue(0));
      }
    }
  }

  @Test
  public void testCluster_DataSet_int_int() {
    System.out.println("cluster(dataset, int, int)");
    final List<List<DataPoint>> clusters = simpleHAC.cluster(easyData10, 2, 20);
    assertEquals(10, clusters.size());
    final Set<Integer> seenBefore = new IntSet();
    for (final List<DataPoint> cluster : clusters) {
      final int thisClass = cluster.get(0).getCategoricalValue(0);
      assertFalse(seenBefore.contains(thisClass));
      for (final DataPoint dp : cluster) {
        assertEquals(thisClass, dp.getCategoricalValue(0));
      }
    }
  }

  @Test
  public void testCluster_DataSet_int_int_ExecutorService() {
    System.out.println("cluster(dataset, int, int, ExecutorService)");
    final List<List<DataPoint>> clusters = simpleHAC.cluster(easyData10, 2, 20, ex);
    assertEquals(10, clusters.size());
    final Set<Integer> seenBefore = new IntSet();
    for (final List<DataPoint> cluster : clusters) {
      final int thisClass = cluster.get(0).getCategoricalValue(0);
      assertFalse(seenBefore.contains(thisClass));
      for (final DataPoint dp : cluster) {
        assertEquals(thisClass, dp.getCategoricalValue(0));
      }
    }
  }
}
