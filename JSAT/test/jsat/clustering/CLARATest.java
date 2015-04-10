package jsat.clustering;

import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.Uniform;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

import org.junit.*;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class CLARATest
{
    
    public CLARATest()
    {
    }

    static private CLARA algo;
    static private SimpleDataSet easyData10;
    static private SimpleDataSet easyData2;
    static private ExecutorService ex;

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        algo = new CLARA();
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.05, 0.05), new XORWOW(12), 2, 5);
        easyData10 = gdg.generateData(40);
        gdg = new GridDataGenerator(new Uniform(-0.05, 0.05), new XORWOW(12), 2, 1);
        easyData2 = gdg.generateData(40);
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
    public void testCluster_DataSet_int()
    {
        System.out.println("cluster(dataset, int)");
        CLARA toUse = algo.clone();
        toUse.setSampleSize(easyData10.getSampleSize()/2);
        List<List<DataPoint>> clusters = toUse.cluster(easyData10, 10);
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
    public void testCluster_DataSet_int_ExecutorService()
    {
        System.out.println("cluster(dataset, int, ExecutorService)");
        CLARA toUse = algo.clone();
        toUse.setSampleCount(10);
        toUse.setSampleSize(easyData10.getSampleSize()/2);
        List<List<DataPoint>> clusters = toUse.cluster(easyData10, 10, ex);
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
        System.out.println("cluster(dataset, int, ExecutorService)");
        CLARA toUse = algo.clone();
        List<List<DataPoint>> clusters = toUse.cluster(easyData2, ex);
        assertEquals(2, clusters.size());
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
