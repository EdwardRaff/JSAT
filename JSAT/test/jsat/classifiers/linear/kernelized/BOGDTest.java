package jsat.classifiers.linear.kernelized;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.*;
import jsat.distributions.kernels.RBFKernel;
import jsat.lossfunctions.HingeLoss;
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

        

        final ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        
        for(final boolean sampling : new boolean[]{true, false})
        {
            final BOGD instance = new BOGD(new RBFKernel(0.5), 30, 0.5, 1e-3, 10, new HingeLoss());
            instance.setUniformSampling(sampling);
       
            final ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, new XORWOW());
            final ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, new XORWOW());

            final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.0);
        }

        ex.shutdownNow();

    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        
        for(final boolean sampling : new boolean[]{true, false})
        {
            final BOGD instance = new BOGD(new RBFKernel(0.5), 30, 0.5, 1e-3, 10, new HingeLoss());
            instance.setUniformSampling(sampling);
        
            final ClassificationDataSet train = FixedProblems.getInnerOuterCircle(200, new XORWOW());
            final ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, new XORWOW());

            final ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.0);
        }
    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        BOGD instance = new BOGD(new RBFKernel(0.5), 30, 0.5, 1e-3, 10, new HingeLoss());
        
        final ClassificationDataSet t1 = FixedProblems.getInnerOuterCircle(500, new XORWOW());
        final ClassificationDataSet t2 = FixedProblems.getInnerOuterCircle(500, new XORWOW(), 2.0, 10.0);

        instance.setUniformSampling(true);
        instance = instance.clone();

        instance.trainC(t1);

        instance.setUniformSampling(false);
        final BOGD result = instance.clone();
        assertFalse(result.isUniformSampling());
        
        for (int i = 0; i < t1.getSampleSize(); i++) {
          assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
        }
        result.trainC(t2);

        for (int i = 0; i < t1.getSampleSize(); i++) {
          assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());
        }

        for (int i = 0; i < t2.getSampleSize(); i++) {
          assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());
        }

    }
}
