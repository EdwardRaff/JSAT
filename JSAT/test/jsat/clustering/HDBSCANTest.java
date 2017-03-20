package jsat.clustering;

import java.util.Set;

import jsat.classifiers.DataPoint;

import java.util.Random;
import java.util.concurrent.Executors;

import jsat.distributions.Uniform;
import jsat.utils.GridDataGenerator;
import jsat.SimpleDataSet;

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.distributions.Normal;

import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.VectorArray.VectorArrayFactory;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class HDBSCANTest
{
    static private HDBSCAN hdbscan;
    static private SimpleDataSet easyData10;
    static private ExecutorService ex;
    public HDBSCANTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        hdbscan = new HDBSCAN();
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), RandomUtil.getRandom(), 2, 5);
        easyData10 = gdg.generateData(40);
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


    /**
     * Test of cluster method, of class DBSCAN.
     */
    @Test
    public void testCluster_DataSet()
    {
        System.out.println("cluster(dataset)");
        List<List<DataPoint>> clusters = hdbscan.cluster(easyData10);
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

    /**
     * Test of cluster method, of class DBSCAN.
     */
    @Test
    public void testCluster_DataSet_ExecutorService()
    {
        System.out.println("cluster(dataset, executorService)");
        List<List<DataPoint>> clusters = hdbscan.cluster(easyData10, ex);
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
