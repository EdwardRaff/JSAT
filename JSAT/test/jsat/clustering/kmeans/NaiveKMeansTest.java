
package jsat.clustering.kmeans;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;
import jsat.clustering.KClustererBase;
import jsat.clustering.SeedSelectionMethods;
import jsat.distributions.Uniform;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;

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
public class NaiveKMeansTest
{
    static private SimpleDataSet easyData10;
    static private ExecutorService ex;
    /**
     * Used as the starting seeds for k-means clustering to get consistent desired behavior
     */
    static private List<Vec> seeds;
    
    public NaiveKMeansTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass() throws Exception
    {
        final GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new XORWOW(), 2, 5);
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
        final GridDataGenerator gdg = new GridDataGenerator(new Uniform(-1e-10, 1e-10), new XORWOW(), 2, 5);
        final SimpleDataSet seedData = gdg.generateData(1);
        seeds = seedData.getDataVectors();
        for(final Vec v : seeds) {
          v.mutableAdd(0.1);//shift off center so we aren't starting at the expected solution
        }
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of cluster method, of class NaiveKMeans.
     */
    @Test
    public void testCluster_DataSet_intArr()
    {
        System.out.println("cluster");
        final NaiveKMeans kMeans = new NaiveKMeans(new EuclideanDistance(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        final int[] assignment = new int[easyData10.getSampleSize()];
        kMeans.cluster(easyData10, null, 10, seeds, assignment, true, null, true);
        final List<List<DataPoint>> clusters = KClustererBase.createClusterListFromAssignmentArray(assignment, easyData10);
        assertEquals(10, clusters.size());
        final Set<Integer> seenBefore = new IntSet();
        for(final List<DataPoint> cluster :  clusters)
        {
            final int thisClass = cluster.get(0).getCategoricalValue(0);
            assertFalse(seenBefore.contains(thisClass));
            for(final DataPoint dp : cluster) {
              assertEquals(thisClass, dp.getCategoricalValue(0));
            }
        }
    }

    /**
     * Test of cluster method, of class NaiveKMeans.
     */
    @Test
    public void testCluster_3args_1()
    {
        System.out.println("cluster");
        final NaiveKMeans kMeans = new NaiveKMeans(new EuclideanDistance(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST);
        final int[] assignment = new int[easyData10.getSampleSize()];
        kMeans.cluster(easyData10, null, 10, seeds, assignment, true, ex, true);
        final List<List<DataPoint>> clusters = KClustererBase.createClusterListFromAssignmentArray(assignment, easyData10);
        assertEquals(10, clusters.size());
        final Set<Integer> seenBefore = new IntSet();
        for(final List<DataPoint> cluster :  clusters)
        {
            final int thisClass = cluster.get(0).getCategoricalValue(0);
            assertFalse(seenBefore.contains(thisClass));
            for(final DataPoint dp : cluster) {
              assertEquals(thisClass, dp.getCategoricalValue(0));
            }
        }
    }
}
