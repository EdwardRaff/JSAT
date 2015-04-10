package jsat.clustering;

import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.Normal;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;

import org.junit.*;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class FLAMETest
{
    
    public FLAMETest()
    {
    }

    static private FLAME algo;
    static private SimpleDataSet easyData10;
    static private ExecutorService ex;

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        algo = new FLAME(new EuclideanDistance(), 30, 800);
        GridDataGenerator gdg = new GridDataGenerator(new Normal(0, 0.05), new Random(12), 2, 5);
        easyData10 = gdg.generateData(100);
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
        ex.shutdown();
    }
    
    @Before
    public void setUp()
    {
        
    }

    @Test
    public void testCluster_DataSet()
    {
        System.out.println("cluster(dataset)");
        Clusterer toUse = algo.clone();
        List<List<DataPoint>> clusters = toUse.cluster(easyData10);
        assertEquals(10, clusters.size());
        Set<Integer> seenBefore = new IntSet();
        for(List<DataPoint> cluster :  clusters)
        {
            int thisClass = cluster.get(0).getCategoricalValue(0);
            assertFalse(seenBefore.contains(thisClass));
            for(DataPoint dp : cluster)
                assertEquals(thisClass, dp.getCategoricalValue(0));
        }
    }
    
    @Test
    public void testCluster_DataSet_ExecutorService()
    {
        System.out.println("cluster(dataset, ExecutorService)");
        Clusterer toUse = algo.clone();
        List<List<DataPoint>> clusters = toUse.cluster(easyData10, ex);
        assertEquals(10, clusters.size());
        Set<Integer> seenBefore = new IntSet();
        for(List<DataPoint> cluster :  clusters)
        {
            int thisClass = cluster.get(0).getCategoricalValue(0);
            assertFalse(seenBefore.contains(thisClass));
            for(DataPoint dp : cluster)
                assertEquals(thisClass, dp.getCategoricalValue(0));
        }
    }
}
