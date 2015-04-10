/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering;

import java.util.Set;

import jsat.classifiers.DataPoint;

import java.util.concurrent.Executors;

import jsat.distributions.Uniform;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.SimpleDataSet;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;

import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.linear.distancemetrics.EuclideanDistance;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class PAMTest
{
    //Like KMeans the cluster number detection isnt stable enough yet that we can test that it getst he right result. 
    static private PAM pam;
    static private SimpleDataSet easyData10;
    static private ExecutorService ex;
    
    public PAMTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        pam = new PAM(new EuclideanDistance(), new Random(), SeedSelection.KPP);
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.05, 0.05), new Random(), 2, 5);
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
     * Test of cluster method, of class PAM.
     */
    @Test
    public void testCluster_3args_1()
    {
        System.out.println("cluster(dataSet, int, ExecutorService)");
        List<List<DataPoint>> clusters = pam.cluster(easyData10, 10, ex);
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
     * Test of cluster method, of class PAM.
     */
    @Test
    public void testCluster_DataSet_int()
    {
        System.out.println("cluster(dataset, int)");
        List<List<DataPoint>> clusters = pam.cluster(easyData10, 10);
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
