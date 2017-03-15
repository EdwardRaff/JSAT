/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.svm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
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
public class PegasosTest
{
    
    public PegasosTest()
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

    /**
     * Test of trainC method, of class Pegasos.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        ExecutorService threadPool = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        Pegasos instance = new Pegasos();
        instance.trainC(train, threadPool);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
        threadPool.shutdown();
    }

    /**
     * Test of trainC method, of class Pegasos.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        Pegasos instance = new Pegasos();
        instance.trainC(train);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }

}
