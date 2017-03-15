/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.clustering;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jsat.*;
import jsat.classifiers.DataPoint;
import jsat.clustering.kmeans.HamerlyKMeans;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
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
public class GapStatisticTest
{
    static private SimpleDataSet easyData10;
    static private ExecutorService ex;
    static private int K = 2*2;
    
    public GapStatisticTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        GridDataGenerator gdg = new GridDataGenerator(new NormalClampedSample(0.0, 0.05), RandomUtil.getRandom(), 2, 2);
        easyData10 = gdg.generateData(200);
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
    public void testCluster_4args_1_findK()
    {
        System.out.println("cluster findK");
        for(boolean PCSample: new boolean[]{true, false})
        {
            GapStatistic gap = new GapStatistic(new HamerlyKMeans(new EuclideanDistance(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST));
            gap.setPCSampling(PCSample);
            List<List<DataPoint>> clusters = gap.cluster(easyData10, 1, 20, ex);
            
            assertEquals(K, clusters.size());
            Set<Integer> seenBefore = new IntSet();
            for(List<DataPoint> cluster :  clusters)
            {
                int thisClass = cluster.get(0).getCategoricalValue(0);
                assertFalse(seenBefore.contains(thisClass));
                seenBefore.add(thisClass);
                for(DataPoint dp : cluster)
                    assertEquals(thisClass, dp.getCategoricalValue(0));
            }
        }
    }
    
    @Test
    public void testCluster_3args_1_findK()
    {
        System.out.println("cluster findK");
        for(boolean PCSample: new boolean[]{true, false})
        {
            GapStatistic gap = new GapStatistic(new HamerlyKMeans(new EuclideanDistance(), SeedSelectionMethods.SeedSelection.FARTHEST_FIRST));
            gap.setPCSampling(PCSample);
            List<List<DataPoint>> clusters = gap.cluster(easyData10, 1, 20);
            
            assertEquals(K, clusters.size());
            Set<Integer> seenBefore = new IntSet();
            for(List<DataPoint> cluster :  clusters)
            {
                int thisClass = cluster.get(0).getCategoricalValue(0);
                assertFalse(seenBefore.contains(thisClass));
                seenBefore.add(thisClass);
                for(DataPoint dp : cluster)
                    assertEquals(thisClass, dp.getCategoricalValue(0));
            }
        }
    }
    
}
