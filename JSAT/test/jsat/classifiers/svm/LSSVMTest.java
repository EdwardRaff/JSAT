package jsat.classifiers.svm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.*;
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
public class LSSVMTest
{
    static private ExecutorService ex;
    public LSSVMTest()
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
     * Test of trainC method, of class LSSVM.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

        for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
        {
            LSSVM classifier = new LSSVM(new RBFKernel(0.5), cacheMode);
            classifier.setCacheMode(cacheMode);
            classifier.setC(1);
            classifier.trainC(trainSet, ex);

            for (int i = 0; i < testSet.getSampleSize(); i++)
                assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
        }
    }

    /**
     * Test of trainC method, of class LSSVM.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

        for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
        {
            LSSVM classifier = new LSSVM(new RBFKernel(0.5), cacheMode);
            classifier.setCacheMode(cacheMode);
            classifier.setC(1);
            classifier.trainC(trainSet);

            for (int i = 0; i < testSet.getSampleSize(); i++)
                assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
        }
    }

    /**
     * Test of train method, of class LSSVM.
     */
    @Test
    public void testTrain_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");
        RegressionDataSet trainSet = FixedProblems.getSimpleRegression1(150, new Random(2));
        RegressionDataSet testSet = FixedProblems.getSimpleRegression1(50, new Random(3));


        for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
        {
            LSSVM lssvm = new LSSVM(new RBFKernel(0.5), cacheMode);
            lssvm.setCacheMode(cacheMode);
            lssvm.setC(1);
            lssvm.train(trainSet, ex);

            double errors = 0;
            for (int i = 0; i < testSet.getSampleSize(); i++)
                errors += Math.pow(testSet.getTargetValue(i) - lssvm.regress(testSet.getDataPoint(i)), 2);
            assertTrue(errors / testSet.getSampleSize() < 1);
        }
    }

    /**
     * Test of train method, of class LSSVM.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        RegressionDataSet trainSet = FixedProblems.getSimpleRegression1(150, new Random(2));
        RegressionDataSet testSet = FixedProblems.getSimpleRegression1(50, new Random(3));


        for (SupportVectorLearner.CacheMode cacheMode : SupportVectorLearner.CacheMode.values())
        {
            LSSVM lssvm = new LSSVM(new RBFKernel(0.5), cacheMode);
            lssvm.setCacheMode(cacheMode);
            lssvm.setC(1);
            lssvm.train(trainSet);

            double errors = 0;
            for (int i = 0; i < testSet.getSampleSize(); i++)
                errors += Math.pow(testSet.getTargetValue(i) - lssvm.regress(testSet.getDataPoint(i)), 2);
            assertTrue(errors / testSet.getSampleSize() < 1);
        }
    }

}
