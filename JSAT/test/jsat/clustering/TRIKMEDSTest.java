/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering;

import java.util.Set;

import jsat.classifiers.DataPoint;


import jsat.distributions.Uniform;
import jsat.utils.GridDataGenerator;
import jsat.SimpleDataSet;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import static jsat.TestTools.checkClusteringByCat;

import jsat.clustering.SeedSelectionMethods.SeedSelection;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceCounter;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.random.RandomUtil;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class TRIKMEDSTest
{
    //Like KMeans the cluster number detection isnt stable enough yet that we can test that it getst he right result. 
    static private TRIKMEDS pam;
    static private SimpleDataSet easyData10;
    
    public TRIKMEDSTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        pam = new TRIKMEDS(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(1000);
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.05, 0.05), RandomUtil.getRandom(), 2, 5);
        easyData10 = gdg.generateData(100);
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
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
        boolean good = false;
        int count = 0;
        do
        {
            List<List<DataPoint>> clusters = pam.cluster(easyData10, 10, true);
            assertEquals(10, clusters.size());
            good = checkClusteringByCat(clusters);
        }
        while(!good && count++ < 3);
        assertTrue(good);
    }

    /**
     * Test of cluster method, of class PAM.
     */
    @Test
    public void testCluster_DataSet_int()
    {
        System.out.println("cluster(dataset, int)");
        boolean good = false;
        int count = 0;
        do
        {
            List<List<DataPoint>> clusters = pam.cluster(easyData10, 10);
            assertEquals(10, clusters.size());
            good = checkClusteringByCat(clusters);
        }
        while(!good && count++ < 3);
        assertTrue(good);
    }
    
    
    @Test
    public void testCluster_AvoidingCalcs()
    {
        System.out.println("cluster(dataset, int)");
        //Use a deterministic seed initialization. Lets see that the new method does LESS distance computations
        DistanceCounter dm = new DistanceCounter(new EuclideanDistance());
        TRIKMEDS newMethod = new TRIKMEDS(dm, RandomUtil.getRandom(), SeedSelection.MEAN_QUANTILES);
        PAM oldMethod = new PAM(dm, RandomUtil.getRandom(), SeedSelection.MEAN_QUANTILES);
        
        
        newMethod.setStoreMedoids(true);
        oldMethod.setStoreMedoids(true);
        newMethod.cluster(easyData10, 10);
        long newDistanceCalcs = dm.getCallCount();
        dm.resetCounter();
        oldMethod.cluster(easyData10, 10);
        long oldDistanceCalcs = dm.getCallCount();
        dm.resetCounter();
        
        assertTrue(newDistanceCalcs < oldDistanceCalcs);
        //We did less calculations. Did we get the same centroids?
        
        Set<Integer> newMedioids = IntStream.of(newMethod.getMedoids()).boxed().collect(Collectors.toSet());
        Set<Integer> oldMedioids = IntStream.of(newMethod.getMedoids()).boxed().collect(Collectors.toSet());
        for(int i : newMedioids)
            assertTrue(oldMedioids.contains(i));
    }
    
    
    @Test
    public void test_medoid()
    {
        System.out.println("cluster(dataset, int)");
        //Use a deterministic seed initialization. Lets see that the new method does LESS distance computations
        DistanceCounter dm = new DistanceCounter(new EuclideanDistance());
        
        List<Vec> X = easyData10.getDataVectors();
        for(boolean parallel : new boolean[]{true, false})
        {
            assertEquals(PAM.medoid(parallel, X, dm), TRIKMEDS.medoid(parallel, X, dm));
        }
    }
}
