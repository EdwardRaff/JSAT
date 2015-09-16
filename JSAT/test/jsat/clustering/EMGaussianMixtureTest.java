package jsat.clustering;

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
import jsat.distributions.Normal;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class EMGaussianMixtureTest {

  static private SimpleDataSet easyData;
  static private ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    // use normal distribution to match the Gausian assumption, uniform can
    // cause weidness
    // unfirm for kMeans used 0.15, 3 sntd devs of 0.05 = 0.15
    final GridDataGenerator gdg = new GridDataGenerator(new Normal(0, 0.05), new Random(12), 2, 2);
    easyData = gdg.generateData(50);
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  public EMGaussianMixtureTest() {

  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testCluster_3args_2() {
    System.out.println("cluster(dataset, int, threadpool)");
    final EMGaussianMixture em = new EMGaussianMixture(new EuclideanDistance(), new Random(),
        SeedSelectionMethods.SeedSelection.KPP);
    final List<List<DataPoint>> clusters = em.cluster(easyData, 4, ex);
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
    final EMGaussianMixture em = new EMGaussianMixture(new EuclideanDistance(), new Random(),
        SeedSelectionMethods.SeedSelection.KPP);
    final List<List<DataPoint>> clusters = em.cluster(easyData, 4);
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
