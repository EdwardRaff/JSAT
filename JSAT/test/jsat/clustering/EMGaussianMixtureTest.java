package jsat.clustering;

import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.NormalClampedSample;

import jsat.SimpleDataSet;
import static jsat.TestTools.checkClusteringByCat;
import jsat.classifiers.DataPoint;
import jsat.clustering.kmeans.HamerlyKMeans;
import jsat.distributions.Normal;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;

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
    
    public EMGaussianMixtureTest()
    {
        
    }
    
    @BeforeClass
    public static void setUpClass()
    {
    }
    
    @AfterClass
    public static void tearDownClass()
    {
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
    public void testCluster_3args_2()
    {
        System.out.println("cluster(dataset, int, threadpool)");
        
        boolean good = false;
        int count = 0;
        do
        {
            GridDataGenerator gdg = new GridDataGenerator(new NormalClampedSample(0, 0.05), RandomUtil.getRandom(), 2, 2);
            easyData = gdg.generateData(50);

            good = true;
            for (boolean parallel : new boolean[]{true, false})
            {
                EMGaussianMixture em = new EMGaussianMixture(SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);

                List<List<DataPoint>> clusters = em.cluster(easyData, 4, parallel);
                assertEquals(4, clusters.size());
                good = good & checkClusteringByCat(clusters);
            }
        }
        while (!good && count++ < 3);
        assertTrue(good);
    }

}
