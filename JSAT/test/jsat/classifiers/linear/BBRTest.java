/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.linear;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPointPair;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
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
public class BBRTest
{
    private static ExecutorService ex;
    
    public BBRTest()
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

    /**
     * Test of trainC method, of class BBR.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        for (BBR.Prior prior : BBR.Prior.values())
        {
            BBR lr = new BBR(0.01, 1000, prior);
            lr.trainC(train, ex);

            ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

            for (DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), lr.classify(dpp.getDataPoint()).mostLikely());
        }
    }

    /**
     * Test of trainC method, of class BBR.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

        for (BBR.Prior prior : BBR.Prior.values())
        {
            BBR lr = new BBR(0.01, 1000, prior);
            lr.trainC(train);

            ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

            for (DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), lr.classify(dpp.getDataPoint()).mostLikely());
        }
    }
    
    @Test
    public void test_reported_bug_params()
    {
        Classifier classifier = new jsat.classifiers.linear.BBR(7);
        if (classifier instanceof Parameterized)
        {
            List<Parameter> list = ((Parameterized) classifier).getParameters();
            for (final Parameter p : list)
            {
                assertNotNull(p.getName());
                assertNotNull(p.getValueString());
            }
        }
    }
}
