package jsat.clustering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import java.util.EnumSet;
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
import jsat.distributions.Normal;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class OPTICSTest {

  static private OPTICS optics;

  static private EnumSet<OPTICS.ExtractionMethod> toTest = EnumSet.of(OPTICS.ExtractionMethod.THRESHHOLD,
      OPTICS.ExtractionMethod.THRESHHOLD);
  static private SimpleDataSet easyData10;
  static private ExecutorService ex;

  @BeforeClass
  public static void setUpClass() throws Exception {
    optics = new OPTICS();
    final GridDataGenerator gdg = new GridDataGenerator(new Normal(0, 0.05), new Random(12), 2, 5);
    easyData10 = gdg.generateData(100);
    ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
  }

  @AfterClass
  public static void tearDownClass() throws Exception {
    ex.shutdown();
  }

  public OPTICSTest() {
  }

  @Before
  public void setUp() {

  }

  @Test
  public void testCluster_DataSet() {
    System.out.println("cluster(dataset)");
    for (final OPTICS.ExtractionMethod method : toTest) {

      optics.setExtractionMethod(method);
      final List<List<DataPoint>> clusters = optics.cluster(easyData10);
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

  @Test
  public void testCluster_DataSet_ExecutorService() {
    for (final OPTICS.ExtractionMethod method : toTest) {
      optics.setExtractionMethod(method);
      System.out.println("cluster(dataset, ExecutorService)");
      final List<List<DataPoint>> clusters = optics.cluster(easyData10, ex);
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
}
