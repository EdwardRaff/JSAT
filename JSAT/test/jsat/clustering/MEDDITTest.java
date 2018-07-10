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
import jsat.distributions.Normal;
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
public class MEDDITTest
{
    //Like KMeans the cluster number detection isnt stable enough yet that we can test that it getst he right result. 
    static private MEDDIT pam;
    static private SimpleDataSet easyData10;
    
    public MEDDITTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
        pam = new MEDDIT(new EuclideanDistance(), RandomUtil.getRandom(), SeedSelection.FARTHEST_FIRST);
        pam.setMaxIterations(500);
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.05, 0.05), RandomUtil.getRandom(), 2, 5);
        easyData10 = gdg.generateData(200);
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
    
    //This test works but takes a while... so just commenting it out but leaving incase I need for debuging later
//    @Test
    public void testCluster_AvoidingCalcs()
    {
        System.out.println("cluster(dataset, int)");
        //Use a deterministic seed initialization. Lets see that the new method does LESS distance computations
        DistanceCounter dm = new DistanceCounter(new EuclideanDistance());
        MEDDIT newMethod = new MEDDIT(dm, RandomUtil.getRandom(), SeedSelection.MEAN_QUANTILES);
        PAM oldMethod = new PAM(dm, RandomUtil.getRandom(), SeedSelection.MEAN_QUANTILES);
        
        
        //MEDDIT works best when dimenion is higher, and poorly when dimension is low. So lets put it in the happy area
        GridDataGenerator gdg = new GridDataGenerator(new Normal(0, 0.1), RandomUtil.getRandom(), 2, 2, 2, 2);
        SimpleDataSet data = gdg.generateData(500);
        
        long N = data.size();
        newMethod.setStoreMedoids(true);
        oldMethod.setStoreMedoids(true);
        
        //To make this test run faster, lets just do a few iterations. We should both reach the same result
        newMethod.setMaxIterations(5);
        oldMethod.setMaxIterations(5);
        
        newMethod.cluster(data, 10);
        long newDistanceCalcs = dm.getCallCount();
        dm.resetCounter();
        oldMethod.cluster(data, 10);
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
        
        //MEDDIT works best when dimenion is higher, and poorly when dimension is low. So lets put it in the happy area
        GridDataGenerator gdg = new GridDataGenerator(new Normal(0, 0.1), RandomUtil.getRandom(), 2, 2, 2, 2);

        
        
        List<Vec> X = gdg.generateData(500).getDataVectors();
        double tol = 0.01;
        
        int tureMed = PAM.medoid(true, X, dm);
        long pamD = dm.getCallCount();
        
        dm.resetCounter();
        for(boolean parallel : new boolean[]{false, true})
        {
            dm.resetCounter();
            int approxMed = MEDDIT.medoid(parallel, X, tol , dm);
            
            
            assertEquals(tureMed, approxMed);
            
            assertTrue(pamD > dm.getCallCount());
        }
    }
}
