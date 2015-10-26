package jsat.clustering.hierarchical;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.kmeans.ElkanKMeans;
import jsat.clustering.evaluation.DaviesBouldinIndex;
import jsat.distributions.Uniform;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import org.junit.*;

/**
 *
 * @author Edward Raff
 */
public class DivisiveLocalClustererTest 
{

    static private DivisiveLocalClusterer dlc;
    static private SimpleDataSet easyData;
    static private ExecutorService ex;

    public DivisiveLocalClustererTest() {
    }

    @BeforeClass
    public static void setUpClass() throws Exception 
    {
        final DistanceMetric dm = new EuclideanDistance();
        dlc = new DivisiveLocalClusterer(new ElkanKMeans(dm), new DaviesBouldinIndex(dm));
        final GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new Random(12), 2, 2);
        easyData = gdg.generateData(100);
        ex = Executors.newFixedThreadPool(10);
    }

    @AfterClass
    public static void tearDownClass() throws Exception {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    /**
     * Test of cluster method, of class KMeans.
     */
    @Test
    public void testCluster_DataSet_int()
    {
        System.out.println("cluster(dataset, int)");
        final List<List<DataPoint>> clusters = dlc.cluster(easyData, 10);
        assertEquals(10, clusters.size());
        final Set<Integer> seenBefore = new IntSet();
        for (final List<DataPoint> cluster : clusters)
        {
            final int thisClass = cluster.get(0).getCategoricalValue(0);
            assertFalse(seenBefore.contains(thisClass));
            for (final DataPoint dp : cluster) {
              assertEquals(thisClass, dp.getCategoricalValue(0));
            }
        }
    }

    @Test
    public void testCluster_DataSet_int_ExecutorService()
    {
        System.out.println("cluster(dataset, int, ExecutorService)");
        final List<List<DataPoint>> clusters = dlc.cluster(easyData, 10, ex);
        assertEquals(10, clusters.size());
        final Set<Integer> seenBefore = new IntSet();
        for (final List<DataPoint> cluster : clusters)
        {
            final int thisClass = cluster.get(0).getCategoricalValue(0);
            assertFalse(seenBefore.contains(thisClass));
            for (final DataPoint dp : cluster) {
              assertEquals(thisClass, dp.getCategoricalValue(0));
            }
        }
    }
}
