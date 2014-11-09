/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package jsat.clustering;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.DataSet;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.kernels.RBFKernel;
import jsat.utils.SystemInfo;
import jsat.utils.random.XOR96;
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
public class ElkanKernelKMeansTest
{
    static private ExecutorService ex;
    public ElkanKernelKMeansTest()
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
     * Test of cluster method, of class ElkanKernelKMeans.
     */
    @Test
    public void testCluster_4args()
    {
        System.out.println("cluster");
        ElkanKernelKMeans kmeans = new ElkanKernelKMeans(new RBFKernel(0.1));
        ClassificationDataSet toCluster = FixedProblems.getCircles(1000, new XOR96(), 1e-3, 1.0);
        int[] result = kmeans.cluster(toCluster, 2, ex, (int[])null);
        //make sure each cluster has points from only 1 class. If true then everyone is good
        Map<Integer, Set<Integer>> tmp = new HashMap<Integer, Set<Integer>>();
        for(int c = 0; c< toCluster.getClassSize(); c++)
            tmp.put(c, new HashSet<Integer>());
        for(int i = 0; i < result.length; i++)
            tmp.get(toCluster.getDataPointCategory(i)).add(result[i]);
        for(Set<Integer> set : tmp.values())
            assertEquals(1, set.size());
    }

    /**
     * Test of cluster method, of class ElkanKernelKMeans.
     */
    @Test
    public void testCluster_3args()
    {
        System.out.println("cluster");
        ElkanKernelKMeans kmeans = new ElkanKernelKMeans(new RBFKernel(0.1));
        ClassificationDataSet toCluster = FixedProblems.getCircles(1000, new XOR96(), 1e-3, 1.0);
        int[] result = kmeans.cluster(toCluster, 2, (int[])null);
        //make sure each cluster has points from only 1 class. If true then everyone is good
        Map<Integer, Set<Integer>> tmp = new HashMap<Integer, Set<Integer>>();
        for(int c = 0; c< toCluster.getClassSize(); c++)
            tmp.put(c, new HashSet<Integer>());
        for(int i = 0; i < result.length; i++)
            tmp.get(toCluster.getDataPointCategory(i)).add(result[i]);
        for(Set<Integer> set : tmp.values())
            assertEquals(1, set.size());
    }
    
}
