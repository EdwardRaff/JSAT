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

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class EMGaussianMixtureTest
{
    static private SimpleDataSet easyData;
    static private ExecutorService ex;
    
    public EMGaussianMixtureTest()
    {
        
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        //use normal distribution to match the Gausian assumption, uniform can cause weidness
        //unfirm for kMeans used 0.15, 3 sntd devs of 0.05 = 0.15
        GridDataGenerator gdg = new GridDataGenerator(new Normal(0, 0.05), new Random(12), 2, 2);
        easyData = gdg.generateData(50);
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }
    
    @AfterClass
    public static void tearDownClass()
    {
        ex.shutdown();
    }
    
    @Before
    public void setUp()
    {
    }
    
    @After
    public void tearDown()
    {
    }
    
    
    @Test
    public void testCluster_DataSet_int()
    {
        System.out.println("cluster(dataset, int)");
        EMGaussianMixture em = new EMGaussianMixture(SeedSelectionMethods.SeedSelection.KPP);
        List<List<DataPoint>> clusters = em.cluster(easyData, 4);
        assertEquals(4, clusters.size());
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
    public void testCluster_3args_2()
    {
        System.out.println("cluster(dataset, int, threadpool)");
        EMGaussianMixture em = new EMGaussianMixture(SeedSelectionMethods.SeedSelection.KPP);
        List<List<DataPoint>> clusters = em.cluster(easyData, 4, ex);
        assertEquals(4, clusters.size());
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
