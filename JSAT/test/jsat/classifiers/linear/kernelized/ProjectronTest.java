package jsat.classifiers.linear.kernelized;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.*;
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
public class ProjectronTest
{
    static private ExecutorService ex;
    
    public ProjectronTest()
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

    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        for(boolean useMargin : new boolean[]{true, false})
        {
            Projectron instance = new Projectron(new RBFKernel(0.5));
            instance.setUseMarginUpdates(useMargin);
            
            ClassificationDataSet train = FixedProblems.getInnerOuterCircle(1000, RandomUtil.getRandom());
            //add some miss labled data to get the error code to cick in and get exercised
            for(int i = 0; i < 500; i+=20)
            {
                DataPoint dp = train.getDataPoint(i);
                int y = train.getDataPointCategory(i);
                int badY = (y == 0) ? 1 : 0;
                train.addDataPoint(dp, badY);
            }

            ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, RandomUtil.getRandom());

            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.3);//given some leway due to label noise
        }
        ex.shutdownNow();

    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        
        for(boolean useMargin : new boolean[]{true, false})
        {
            Projectron instance = new Projectron(new RBFKernel(0.5));
            instance.setUseMarginUpdates(useMargin);
        
            ClassificationDataSet train = FixedProblems.getInnerOuterCircle(1000, RandomUtil.getRandom());
            //add some miss labled data to get the error code to cick in and get exercised
            for(int i = 0; i < 500; i+=20)
            {
                DataPoint dp = train.getDataPoint(i);
                int y = train.getDataPointCategory(i);
                int badY = (y == 0) ? 1 : 0;
                train.addDataPoint(dp, badY);
            }

            ClassificationDataSet test = FixedProblems.getInnerOuterCircle(100, RandomUtil.getRandom());
            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
            cme.evaluateTestSet(test);

            assertEquals(0, cme.getErrorRate(), 0.3);//given some leway due to label noise
        }
    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        Projectron instance = new Projectron(new RBFKernel(0.5));
        
        ClassificationDataSet t1 = FixedProblems.getInnerOuterCircle(500, RandomUtil.getRandom());
        ClassificationDataSet t2 = FixedProblems.getInnerOuterCircle(500, RandomUtil.getRandom(), 2.0, 10.0);

        instance = instance.clone();

        instance.trainC(t1);

        Projectron result = instance.clone();
        
        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
        result.trainC(t2);

        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());

        for (int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());

    }
}
