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

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.evaluation.DaviesBouldinIndex;
import jsat.clustering.kmeans.ElkanKMeans;
import jsat.distributions.Uniform;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;

/**
 *
 * @author Edward Raff
 */
public class DivisiveGlobalClustererTest {

  static private DivisiveGlobalClusterer dgc;
  static private SimpleDataSet easyData;
  static private ExecutorService ex;

  @BeforeClass
  public static void setUpClass() throws Exception {
    final GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new Random(12), 2, 2);
    easyData = gdg.generateData(60);
    ex = Executors.newFixedThreadPool(10);
  }

  @AfterClass
  public static void tearDownClass() throws Exception {
  }

  public DivisiveGlobalClustererTest() {
  }

  @Before
  public void setUp() {
    final DistanceMetric dm = new EuclideanDistance();
    dgc = new DivisiveGlobalClusterer(new ElkanKMeans(dm), new DaviesBouldinIndex(dm));
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testCluster_DataSet() {
    System.out.println("cluster(dataset)");
    final List<List<DataPoint>> clusters = dgc.cluster(easyData);
    assertEquals(4, clusters.size());
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
    final List<List<DataPoint>> clusters = dgc.cluster(easyData, ex);
    assertEquals(4, clusters.size());
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
  public void testCluster_DataSet_int() {
    System.out.println("cluster(dataset, int)");
    final List<List<DataPoint>> clusters = dgc.cluster(easyData, 4);
    assertEquals(4, clusters.size());
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
  public void testCluster_DataSet_int_ExecutorService() {
    System.out.println("cluster(dataset, int, ExecutorService)");
    final List<List<DataPoint>> clusters = dgc.cluster(easyData, 4, ex);
    assertEquals(4, clusters.size());
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
    final List<List<DataPoint>> clusters = dgc.cluster(easyData, 2, 20);
    assertEquals(4, clusters.size());
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
    final List<List<DataPoint>> clusters = dgc.cluster(easyData, 2, 20, ex);
    assertEquals(4, clusters.size());
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
