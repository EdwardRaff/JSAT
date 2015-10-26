/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering.kmeans;

import java.util.Set;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.KClustererBase;
import jsat.distributions.Uniform;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
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
public class ElkanKMeansTest
{
    static private SimpleDataSet easyData10;
    static private ExecutorService ex;
    /**
     * Used as the starting seeds for k-means clustering to get consistent desired behavior
     */
    static private List<Vec> seeds;
    
    public ElkanKMeansTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new XORWOW(1238962356), 2, 5);
        easyData10 = gdg.generateData(110);
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
        //generate seeds that should lead to exact solution 
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-1e-10, 1e-10), new XORWOW(5638973498234L), 2, 5);
        SimpleDataSet seedData = gdg.generateData(1);
        seeds = seedData.getDataVectors();
        for(Vec v : seeds) {
          v.mutableAdd(0.1);//shift off center so we aren't starting at the expected solution
        }
    }



    /**
     * Test of cluster method, of class ElkanKMeans.
     */
    @Test
    public void testCluster_DataSet_int()
    {
        System.out.println("cluster(dataset, int)");
        ElkanKMeans kMeans = new ElkanKMeans(new EuclideanDistance());
        int[] assignment = new int[easyData10.getSampleSize()];
        kMeans.cluster(easyData10, null, 10, seeds, assignment, true, null, true);
        List<List<DataPoint>> clusters = KClustererBase.createClusterListFromAssignmentArray(assignment, easyData10);
        assertEquals(10, clusters.size());
        Set<Integer> seenBefore = new IntSet();
        for(List<DataPoint> cluster :  clusters)
        {
            int thisClass = cluster.get(0).getCategoricalValue(0);
            assertFalse(seenBefore.contains(thisClass));
            for(DataPoint dp : cluster) {
              assertEquals(thisClass, dp.getCategoricalValue(0));
            }
        }
    }


    /**
     * Test of cluster method, of class ElkanKMeans.
     */
    @Test
    public void testCluster_3args_2()
    {
        System.out.println("cluster(dataset, int, threadpool)");
        ElkanKMeans kMeans = new ElkanKMeans(new EuclideanDistance());
        int[] assignment = new int[easyData10.getSampleSize()];
        kMeans.cluster(easyData10, null, 10, seeds, assignment, true, ex, true);
        List<List<DataPoint>> clusters = KClustererBase.createClusterListFromAssignmentArray(assignment, easyData10);
        assertEquals(10, clusters.size());
        Set<Integer> seenBefore = new IntSet();
        for(List<DataPoint> cluster :  clusters)
        {
            int thisClass = cluster.get(0).getCategoricalValue(0);
            assertFalse(seenBefore.contains(thisClass));
            for(DataPoint dp : cluster) {
              assertEquals(thisClass, dp.getCategoricalValue(0));
            }
        }
    }
}
