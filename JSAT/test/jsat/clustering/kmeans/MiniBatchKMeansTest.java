package jsat.clustering.kmeans;

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
import jsat.clustering.SeedSelectionMethods;
import jsat.distributions.Uniform;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class MiniBatchKMeansTest {
  // NOTE: FARTHER FIST seed + 2 x 2 grid of 4 classes results in a
  // deterministic result given a high density

  static private SimpleDataSet easyData10;
  static private ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    final GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new Random(12), 2, 2);
    easyData10 = gdg.generateData(110);
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  public MiniBatchKMeansTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  /**
   * Test of cluster method, of class MiniBatchKMeans.
   */
  @Test
  public void testCluster_3args_1() {
    System.out.println("cluster");
    final MiniBatchKMeans kMeans = new MiniBatchKMeans(new EuclideanDistance(), 50, 50,
        SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
    final List<List<DataPoint>> clusters = kMeans.cluster(easyData10, 10, ex);
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
   * Test of cluster method, of class MiniBatchKMeans.
   */
  @Test
  public void testCluster_DataSet_intArr() {
    System.out.println("cluster");
    final MiniBatchKMeans kMeans = new MiniBatchKMeans(new EuclideanDistance(), 50, 50,
        SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
    final List<List<DataPoint>> clusters = kMeans.cluster(easyData10, 10);
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
