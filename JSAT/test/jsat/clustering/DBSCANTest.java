/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
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

import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.linear.vectorcollection.VectorArray.VectorArrayFactory;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class DBSCANTest
{
    static private DBSCAN dbscan;
    static private SimpleDataSet easyData10;
    static private ExecutorService ex;
    public DBSCANTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        dbscan = new DBSCAN(new EuclideanDistance(), new VectorArrayFactory<VecPaired<Vec, Integer>>());
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new Random(12), 2, 5);
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
    public void testCluster_DataSet_int()
    {
        System.out.println("cluster(dataset, int)");
        List<List<DataPoint>> clusters = dbscan.cluster(easyData10, 5);
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
    public void testCluster_DataSet()
    {
        System.out.println("cluster(dataset)");
        List<List<DataPoint>> clusters = dbscan.cluster(easyData10);
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
        List<List<DataPoint>> clusters = dbscan.cluster(easyData10, ex);
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
    public void testCluster_3args_1()
    {
        System.out.println("cluster(dataset, double, int)");
        //We know the range is [-.15, .15]
        List<List<DataPoint>> clusters = dbscan.cluster(easyData10, 0.15, 5);
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
    public void testCluster_3args_2()
    {
        System.out.println("cluster(dataset, int, executorService)");
        List<List<DataPoint>> clusters = dbscan.cluster(easyData10, 3, ex);
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
    public void testCluster_4args()
    {
        System.out.println("cluster(dataset, double, int, executorService)");
        //We know the range is [-.15, .15]
        List<List<DataPoint>> clusters = dbscan.cluster(easyData10, 0.15, 5, ex);
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
