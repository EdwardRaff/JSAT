package jsat.clustering.kmeans;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import java.util.List;
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
import jsat.utils.random.XORWOW;

/**
 *
 * @author Edward Raff
 */
public class XMeansTest {

  static private SimpleDataSet easyData10;
  static private ExecutorService ex;

  @BeforeClass
  public static void setUpClass() {
    final GridDataGenerator gdg = new GridDataGenerator(new Uniform(0.0, 0.10), new XORWOW(), 2, 2);
    easyData10 = gdg.generateData(50);
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() {
    ex.shutdown();
  }

  public XMeansTest() {
  }

  @Before
  public void setUp() {
  }

  @After
  public void tearDown() {
  }

  @Test
  public void testCluster_3args_1_findK() {
    System.out.println("cluster findK");
    final XMeans kMeans = new XMeans(
        new HamerlyKMeans(new EuclideanDistance(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST));
    final List<List<DataPoint>> clusters = kMeans.cluster(easyData10, 2, 40);
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
  public void testCluster_4args_1_findK() {
    System.out.println("cluster findK");
    final XMeans kMeans = new XMeans(
        new HamerlyKMeans(new EuclideanDistance(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST));
    final List<List<DataPoint>> clusters = kMeans.cluster(easyData10, 2, 40, ex);
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
