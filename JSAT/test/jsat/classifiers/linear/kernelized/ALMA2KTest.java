package jsat.classifiers.linear.kernelized;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.distributions.kernels.RBFKernel;
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
public class ALMA2KTest
{
    
    public ALMA2KTest()
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
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");

        ALMA2K instance = new ALMA2K(new RBFKernel(0.5), 0.8);

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, RandomUtil.getRandom());
        ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, RandomUtil.getRandom());

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
        cme.evaluateTestSet(test);

        assertEquals(0, cme.getErrorRate(), 0.0);

        ex.shutdownNow();

    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");

        
        ALMA2K instance = new ALMA2K(new RBFKernel(0.5), 0.8);
        
        ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, RandomUtil.getRandom());
        ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, RandomUtil.getRandom());

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
        cme.evaluateTestSet(test);

        assertEquals(0, cme.getErrorRate(), 0.0);

    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        ALMA2K instance = new ALMA2K(new RBFKernel(0.5), 0.8);
        
        ClassificationDataSet t1 = FixedProblems.getInnerOuterCircle(500, RandomUtil.getRandom());
        ClassificationDataSet t2 = FixedProblems.getInnerOuterCircle(500, RandomUtil.getRandom(), 2.0, 10.0);

        instance = instance.clone();

        instance.trainC(t1);

        instance.setAveraged(true);
        ALMA2K result = instance.clone();
        assertTrue(result.isAveraged());
        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
        result.trainC(t2);

        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());

        for (int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());

    }
}
