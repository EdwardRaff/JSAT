package jsat.classifiers.linear.kernelized;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.distributions.kernels.KernelPoint;
import jsat.distributions.kernels.RBFKernel;
import jsat.lossfunctions.*;
import jsat.regression.RegressionDataSet;
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
public class KernelSGDTest
{
    static private ExecutorService ex;
    
    public KernelSGDTest()
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

        KernelSGD classifier = new KernelSGD(new HingeLoss(), new RBFKernel(0.5), 1e-5, KernelPoint.BudgetStrategy.STOP, 100);

        classifier.trainC(trainSet, ex);

        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(150, new Random(2));
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(50, new Random(3));

        KernelSGD classifier = new KernelSGD(new HingeLoss(), new RBFKernel(0.5), 1e-5, KernelPoint.BudgetStrategy.STOP, 100);

        classifier.trainC(trainSet);

        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());

    }
    
    @Test
    public void testTrainC_ClassificationDataSet_Multi_ExecutorService()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getCircles(150, new Random(2), 1.0, 2.0, 4.0);
        ClassificationDataSet testSet = FixedProblems.getCircles(50, new Random(3), 1.0, 2.0, 4.0);

        KernelSGD classifier = new KernelSGD(new HingeLoss(), new RBFKernel(0.5), 1e-5, KernelPoint.BudgetStrategy.STOP, 100);

        classifier.trainC(trainSet, ex);

        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());

    }
    
    @Test
    public void testTrainC_ClassificationDataSet_Multi()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getCircles(150, new Random(2), 1.0, 2.0, 4.0);
        ClassificationDataSet testSet = FixedProblems.getCircles(50, new Random(3), 1.0, 2.0, 4.0);

        KernelSGD classifier = new KernelSGD(new HingeLoss(), new RBFKernel(0.5), 1e-5, KernelPoint.BudgetStrategy.STOP, 100);

        classifier.trainC(trainSet);

        for (int i = 0; i < testSet.getSampleSize(); i++)
            assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());

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

        KernelSGD classifier = new KernelSGD(new EpsilonInsensitiveLoss(0.1), new RBFKernel(0.5), 1e-5, KernelPoint.BudgetStrategy.MERGE_RBF, 50);
        classifier.setEpochs(10);
        classifier.train(trainSet, ex);

        double errors = 0;
        for (int i = 0; i < testSet.getSampleSize(); i++)
            errors += Math.pow(testSet.getTargetValue(i) - classifier.regress(testSet.getDataPoint(i)), 2);
        assertTrue(errors / testSet.getSampleSize() < 1);
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

        KernelSGD classifier = new KernelSGD(new EpsilonInsensitiveLoss(0.1), new RBFKernel(0.5), 1e-5, KernelPoint.BudgetStrategy.MERGE_RBF, 50);
        classifier.setEpochs(10);
        classifier.train(trainSet);

        double errors = 0;
        for (int i = 0; i < testSet.getSampleSize(); i++)
            errors += Math.pow(testSet.getTargetValue(i) - classifier.regress(testSet.getDataPoint(i)), 2);
        assertTrue(errors / testSet.getSampleSize() < 1);
    }
    
    @Test
    public void testClone()
    {
        System.out.println("clone");

        KernelSGD instance = new KernelSGD(new LogisticLoss(), new RBFKernel(0.5), 1e-4, KernelPoint.BudgetStrategy.MERGE_RBF, 100);
        
        ClassificationDataSet t1 = FixedProblems.getInnerOuterCircle(500, RandomUtil.getRandom());
        ClassificationDataSet t2 = FixedProblems.getInnerOuterCircle(500, RandomUtil.getRandom(), 2.0, 10.0);

        instance = instance.clone();

        instance.trainC(t1);


        KernelSGD result = instance.clone();
        
        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
        result.trainC(t2);

        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());

        for (int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());

    }

}
