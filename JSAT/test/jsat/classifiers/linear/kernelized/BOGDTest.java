package jsat.classifiers.linear.kernelized;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.*;
import jsat.distributions.kernels.RBFKernel;
import jsat.lossfunctions.HingeLoss;
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
public class BOGDTest
{
    
    public BOGDTest()
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

        

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        
        for(boolean sampling : new boolean[]{true, false})
        {
            BOGD instance = new BOGD(new RBFKernel(0.5), 50, 0.5, 1e-3, 10, new HingeLoss());
            instance.setUniformSampling(sampling);
       
            ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, RandomUtil.getRandom(), 1, 4);
            ClassificationDataSet test = FixedProblems.getCircles(100, 0.0, RandomUtil.getRandom(), 1, 4);

            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.0);
        }

        ex.shutdownNow();

    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        
        for(boolean sampling : new boolean[]{true, false})
        {
            BOGD instance = new BOGD(new RBFKernel(0.5), 50, 0.5, 1e-3, 10, new HingeLoss());
            instance.setUniformSampling(sampling);
        
            ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, RandomUtil.getRandom(), 1, 4);
            ClassificationDataSet test = FixedProblems.getCircles(100, 0.0, RandomUtil.getRandom(), 1, 4);

            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.0);
        }
    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        BOGD instance = new BOGD(new RBFKernel(0.5), 50, 0.5, 1e-3, 10, new HingeLoss());
        
        ClassificationDataSet t1 = FixedProblems.getCircles(500, 0.0, RandomUtil.getRandom(), 1, 4);
        ClassificationDataSet t2 = FixedProblems.getCircles(500, 0.0, RandomUtil.getRandom(), 0.5, 3.0);

        instance.setUniformSampling(true);
        instance = instance.clone();

        instance.trainC(t1);

        instance.setUniformSampling(false);
        BOGD result = instance.clone();
        assertFalse(result.isUniformSampling());
        
        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
        result.trainC(t2);

        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());

        for (int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());

    }
}
