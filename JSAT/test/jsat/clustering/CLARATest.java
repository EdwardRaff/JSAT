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

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        algo = new CLARA();
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.005, 0.005), new XORWOW(12), 2, 3);
        easyData10 = gdg.generateData(40);
        gdg = new GridDataGenerator(new Uniform(-0.005, 0.005), new XORWOW(12), 2, 1);
        easyData2 = gdg.generateData(40);
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
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
        toUse.setSampleSize(easyData10.size()/2);
        List<List<DataPoint>> clusters = toUse.cluster(easyData10, 6);
        assertEquals(6, clusters.size());
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
        toUse.setSampleCount(6);
        toUse.setSampleSize(easyData10.size()/2);
        List<List<DataPoint>> clusters = toUse.cluster(easyData10, 6, true);
        assertEquals(6, clusters.size());
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
        List<List<DataPoint>> clusters = toUse.cluster(easyData2, true);
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
