/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package jsat.clustering.kmeans;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.DataSet;

import jsat.FixedProblems;
import jsat.SimpleDataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.Uniform;
import jsat.distributions.kernels.LinearKernel;
import jsat.distributions.kernels.RBFKernel;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.utils.DoubleList;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

import org.junit.*;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class LloydKernelKMeansTest
{
    static private ExecutorService ex;
    public LloydKernelKMeansTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
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

    /**
     * Test of cluster method, of class LloydKernelKMeans.
     */
    @Test
    public void testCluster_4args()
    {
        System.out.println("cluster");
        LloydKernelKMeans kmeans = new LloydKernelKMeans(new RBFKernel(0.1));
        ClassificationDataSet toCluster = FixedProblems.getCircles(1000, RandomUtil.getRandom(), 1e-3, 1.0);
        int[] result = kmeans.cluster(toCluster, 2, ex, (int[])null);
        //make sure each cluster has points from only 1 class. If true then everyone is good
        Map<Integer, Set<Integer>> tmp = new HashMap<Integer, Set<Integer>>();
        for(int c = 0; c< toCluster.getClassSize(); c++)
            tmp.put(c, new IntSet());
        for(int i = 0; i < result.length; i++)
            tmp.get(toCluster.getDataPointCategory(i)).add(result[i]);
        for(Set<Integer> set : tmp.values())
            assertEquals(1, set.size());
    }

    /**
     * Test of cluster method, of class LloydKernelKMeans.
     */
    @Test
    public void testCluster_3args()
    {
        System.out.println("cluster");
        LloydKernelKMeans kmeans = new LloydKernelKMeans(new RBFKernel(0.1));
        ClassificationDataSet toCluster = FixedProblems.getCircles(1000, RandomUtil.getRandom(), 1e-3, 1.0);
        int[] result = kmeans.cluster(toCluster, 2, (int[])null);
        //make sure each cluster has points from only 1 class. If true then everyone is good
        Map<Integer, Set<Integer>> tmp = new HashMap<Integer, Set<Integer>>();
        for(int c = 0; c< toCluster.getClassSize(); c++)
            tmp.put(c, new IntSet());
        for(int i = 0; i < result.length; i++)
            tmp.get(toCluster.getDataPointCategory(i)).add(result[i]);
        for(Set<Integer> set : tmp.values())
            assertEquals(1, set.size());
    }
    
    
    @Test
    public void testCluster_Weighted()
    {
        System.out.println("cluster(dataset, int, threadpool)");
        LloydKernelKMeans kmeans = new LloydKernelKMeans(new LinearKernel());
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new XORWOW(1238962356), 2);
        ClassificationDataSet toCluster = gdg.generateData(200).asClassificationDataSet(0);
        //make the LAST data point so far out it will screw everything up, UNLCESS you understand that it has a tiny weight
        toCluster.getDataPoint(toCluster.getSampleSize()-1).getNumericalValues().set(0, 1.9e100);
        Random rand = new XORWOW(897654);
        for(int i = 0; i < toCluster.getSampleSize(); i++)
            toCluster.getDataPoint(i).setWeight(0.5+5*rand.nextDouble());
        toCluster.getDataPoint(toCluster.getSampleSize()-1).setWeight(1e-200);
        
        int[] result = kmeans.cluster(toCluster, 2, (int[])null);
        //make sure each cluster has points from only 1 class. If true then everyone is good
        Map<Integer, Set<Integer>> tmp = new HashMap<Integer, Set<Integer>>();
        IntSet allSeen = new IntSet();
        for(int c = 0; c< toCluster.getClassSize(); c++)
            tmp.put(c, new IntSet());
        for(int i = 0; i < result.length-1; i++)
        {
            tmp.get(toCluster.getDataPointCategory(i)).add(result[i]);
            allSeen.add(result[i]);
        }
        for(Set<Integer> set : tmp.values())
            assertEquals(1, set.size());
        assertEquals(2, allSeen.size());//make sure we saw both clusters!
        
        result = kmeans.cluster(toCluster, 2, ex, (int[])null);
        //make sure each cluster has points from only 1 class. If true then everyone is good
        tmp = new HashMap<Integer, Set<Integer>>();
        allSeen = new IntSet();
        for(int c = 0; c< toCluster.getClassSize(); c++)
            tmp.put(c, new IntSet());
        for(int i = 0; i < result.length-1; i++)
        {
            tmp.get(toCluster.getDataPointCategory(i)).add(result[i]);
            allSeen.add(result[i]);
        }
        for(Set<Integer> set : tmp.values())
            assertEquals(1, set.size());
        assertEquals(2, allSeen.size());//make sure we saw both clusters!
    }
}
