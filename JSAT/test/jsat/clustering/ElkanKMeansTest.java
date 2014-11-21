/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering;

import java.util.Set;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.distributions.Uniform;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
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
public class ElkanKMeansTest
{
    /*
     * README: 
     * ElkanKMeans is a very heuristic algorithm, so its not easy to make a test where we are very 
     * sure it will get the correct awnser. That is why only 2 of the methods are tested 
     * [ Using KPP, becase random seed selection still isnt consistent enough] 
     * 
     */
    static private ElkanKMeans kMeans;
    static private SimpleDataSet easyData10;
    static private ExecutorService ex;
    public ElkanKMeansTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        kMeans = new ElkanKMeans(new EuclideanDistance(), new Random(11), SeedSelection.KPP);
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new Random(12), 2, 5);
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
        
    }



    /**
     * Test of cluster method, of class ElkanKMeans.
     */
    @Test
    public void testCluster_DataSet_int()
    {
        System.out.println("cluster(dataset, int)");
        List<List<DataPoint>> clusters = kMeans.cluster(easyData10, 10);
        assertEquals(10, clusters.size());
        Set<Integer> seenBefore = new HashSet<Integer>();
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
        List<List<DataPoint>> clusters = kMeans.cluster(easyData10, 10, ex);
        assertEquals(10, clusters.size());
        Set<Integer> seenBefore = new HashSet<Integer>();
        for(List<DataPoint> cluster :  clusters)
        {
            int thisClass = cluster.get(0).getCategoricalValue(0);
            assertFalse(seenBefore.contains(thisClass));
            for(DataPoint dp : cluster)
                assertEquals(thisClass, dp.getCategoricalValue(0));
        }
    }
}
