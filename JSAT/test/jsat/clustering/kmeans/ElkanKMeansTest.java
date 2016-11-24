/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering.kmeans;

import java.util.ArrayList;
import java.util.Set;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.KClustererBase;
import jsat.distributions.Uniform;
import jsat.linear.ConstantVector;
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
        for(Vec v : seeds)
            v.mutableAdd(0.1);//shift off center so we aren't starting at the expected solution
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
        kMeans.cluster(easyData10, null, 10, seeds, assignment, true, null, true, null);
        List<List<DataPoint>> clusters = KClustererBase.createClusterListFromAssignmentArray(assignment, easyData10);
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
     * Test of cluster method, of class ElkanKMeans.
     */
    @Test
    public void testCluster_3args_2()
    {
        System.out.println("cluster(dataset, int, threadpool)");
        ElkanKMeans kMeans = new ElkanKMeans(new EuclideanDistance());
        int[] assignment = new int[easyData10.getSampleSize()];
        kMeans.cluster(easyData10, null, 10, seeds, assignment, true, ex, true, null);
        List<List<DataPoint>> clusters = KClustererBase.createClusterListFromAssignmentArray(assignment, easyData10);
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
    public void testCluster_Weighted()
    {
        System.out.println("cluster(dataset, int, threadpool)");
        ElkanKMeans kMeans = new ElkanKMeans();
        kMeans.setStoreMeans(true);
        ElkanKMeans kMeans2 = new ElkanKMeans();
        kMeans2.setStoreMeans(true);
        
        SimpleDataSet data2 = easyData10.getTwiceShallowClone();
        for(int i = 0; i < data2.getSampleSize(); i++)
            data2.getDataPoint(i).setWeight(15.0);
        
        int[] assignment = new int[easyData10.getSampleSize()];
        List<Vec> orig_seeds = new ArrayList<Vec>();
        List<Vec> seeds2 = new ArrayList<Vec>();
        for(Vec v : seeds)
        {
            orig_seeds.add(v.clone());
            seeds2.add(v.clone());
        }
        kMeans.cluster(easyData10, null, 10, seeds, assignment, true, ex, true, null);
        kMeans2.cluster(data2, null, 10, seeds2, assignment, true, ex, true, null);
        
        //multiplied weights by a constant, should get same solutions
        
        for(int i = 0; i < 10; i++)
        {
            double diff = seeds.get(i).subtract(seeds2.get(i)).sum();
            assertEquals(0.0, diff, 1e-10);
        }
        
        
        //restore means and try again with randomish weights, should end up with something close
        
        for(int i = 0; i < orig_seeds.size(); i++)
        {
            orig_seeds.get(i).copyTo(seeds.get(i));
            orig_seeds.get(i).copyTo(seeds2.get(i));
        }
        
        Random rand = new XORWOW(897654);
        for(int i = 0; i < data2.getSampleSize(); i++)
            data2.getDataPoint(i).setWeight(0.5+5*rand.nextDouble());
        kMeans.cluster(easyData10, null, 10, seeds, assignment, true, ex, true, null);
        kMeans2.cluster(data2, null, 10, seeds2, assignment, true, ex, true, null);
        
        //multiplied weights by a constant, should get similar solutions, but slightly different
        
        for(int i = 0; i < 10; i++)
        {
            double diff = seeds.get(i).subtract(seeds2.get(i)).sum();
            assertEquals(0.0, diff, 0.1);
            assertTrue(Math.abs(diff) > 1e-10 );
        }
        
    }
}
