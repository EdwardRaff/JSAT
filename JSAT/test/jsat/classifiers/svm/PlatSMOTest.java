package jsat.classifiers.svm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.kernels.RBFKernel;
import jsat.regression.RegressionDataSet;
import jsat.utils.SystemInfo;
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
public class PlatSMOTest
{
    static private ExecutorService ex;
    
    public PlatSMOTest()
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
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

        for (boolean modification1 : new boolean[] {true, false})
            for(SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
            {
                PlatSMO classifier = new PlatSMO(new RBFKernel(0.5));
                classifier.setCacheMode(cacheMode);
                classifier.setC(10);
                classifier.setModificationOne(modification1);
                classifier.trainC(trainSet, ex);
                
                for (int i = 0; i < testSet.getSampleSize(); i++)
                    assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
            }
    }
    
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));


        for (boolean modification1 : new boolean[] {true, false})
            for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
            {
                PlatSMO classifier = new PlatSMO(new RBFKernel(0.5));
                classifier.setCacheMode(cacheMode);
                classifier.setC(10);
                classifier.setModificationOne(modification1);
                classifier.trainC(trainSet);
                
                for (int i = 0; i < testSet.getSampleSize(); i++)
                    assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
            }
    }

    /**
     * Test of train method, of class PlatSMO.
     */
    @Test
    public void testTrain_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");
        RegressionDataSet trainSet = FixedProblems.getSimpleRegression1(150, new Random(2));
        RegressionDataSet testSet = FixedProblems.getSimpleRegression1(50, new Random(3));
        
        for (boolean modification1 : new boolean[] {true, false})
            for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
            {
                PlatSMO smo = new PlatSMO(new RBFKernel(0.5));
                smo.setCacheMode(cacheMode);
                smo.setC(1);
                smo.setEpsilon(0.1);
                smo.setModificationOne(modification1);
                smo.train(trainSet, ex);
                
                double errors = 0;
                for (int i = 0; i < testSet.getSampleSize(); i++)
                    errors += Math.pow(testSet.getTargetValue(i) - smo.regress(testSet.getDataPoint(i)), 2);
                assertTrue(errors/testSet.getSampleSize() < 1);
            }
    }

    /**
     * Test of train method, of class PlatSMO.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        RegressionDataSet trainSet = FixedProblems.getSimpleRegression1(150, new Random(2));
        RegressionDataSet testSet = FixedProblems.getSimpleRegression1(50, new Random(3));


        for (boolean modification1 : new boolean[] {true, false})
            for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
            {
                PlatSMO smo = new PlatSMO(new RBFKernel(0.5));
                smo.setCacheMode(cacheMode);
                smo.setC(1);
                smo.setEpsilon(0.1);
                smo.setModificationOne(modification1);
                smo.train(trainSet);
                
                double errors = 0;
                for (int i = 0; i < testSet.getSampleSize(); i++)
                    errors += Math.pow(testSet.getTargetValue(i) - smo.regress(testSet.getDataPoint(i)), 2);
                assertTrue(errors/testSet.getSampleSize() < 1);
            }
    }
}
