package jsat.classifiers.svm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.datatransform.DataTransform;
import jsat.distributions.kernels.LinearKernel;
import jsat.distributions.kernels.RBFKernel;
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
public class PlattSMOTest
{
    static private ExecutorService ex;
    
    public PlattSMOTest()
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
                PlattSMO classifier = new PlattSMO(new RBFKernel(0.5));
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
                PlattSMO classifier = new PlattSMO(new RBFKernel(0.5));
                classifier.setCacheMode(cacheMode);
                classifier.setC(10);
                classifier.setModificationOne(modification1);
                classifier.trainC(trainSet);
                
                for (int i = 0; i < testSet.getSampleSize(); i++)
                    assertEquals(testSet.getDataPointCategory(i), classifier.classify(testSet.getDataPoint(i)).mostLikely());
            }
    }

    /**
     * Test of train method, of class PlattSMO.
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
                PlattSMO smo = new PlattSMO(new RBFKernel(0.5));
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
     * Test of train method, of class PlattSMO.
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
                PlattSMO smo = new PlattSMO(new RBFKernel(0.5));
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
    
    @Test()
    public void testTrainWarmCFastSMO()
    {
        //problem needs to be non-linear to make SMO work harder
        ClassificationDataSet train = FixedProblems.getHalfCircles(250, RandomUtil.getRandom(), 0.1, 0.2);
        
        
        PlattSMO warmModel = new PlattSMO(new LinearKernel(1));
        warmModel.setC(1);
        warmModel.trainC(train);
        
        
        PlattSMO warm = new PlattSMO(new LinearKernel(1));
        warm.setC(1e4);//too large to train efficently like noraml
        
        long start, end;
        
        start = System.currentTimeMillis();
        warm.trainC(train, warmModel);
        end = System.currentTimeMillis();
        long warmTime = (end-start);
        
        
        PlattSMO notWarm = new PlattSMO(new LinearKernel(1));
        notWarm.setC(1e4);//too large to train efficently like noraml
        
        start = System.currentTimeMillis();
        notWarm.trainC(train);
        end = System.currentTimeMillis();
        long normTime = (end-start);
        
        assertTrue(warmTime < normTime*0.75);
        
    }
    
    @Test()
    public void testTrainWarmCFastOther()
    {
        ClassificationDataSet train = FixedProblems.getHalfCircles(250, RandomUtil.getRandom(), 0.1, 0.2);
        
        
        DCDs warmModel = new DCDs();
        warmModel.setUseL1(true);
        warmModel.setUseBias(true);
        warmModel.trainC(train);
        
        
        PlattSMO warm = new PlattSMO(new LinearKernel(1));
        warm.setC(1e4);//too large to train efficently like noraml
        
        long start, end;
        
        start = System.currentTimeMillis();
        warm.trainC(train, warmModel);
        end = System.currentTimeMillis();
        long warmTime = (end-start);
        
        
        PlattSMO notWarm = new PlattSMO(new LinearKernel(1));
        notWarm.setC(1e4);//too large to train efficently like noraml
        
        start = System.currentTimeMillis();
        notWarm.trainC(train);
        end = System.currentTimeMillis();
        long normTime = (end-start);
        
        assertTrue(warmTime < normTime*0.75);
        
    }
    
    @Test()
    public void testTrainWarmRFastOther()
    {
        RegressionDataSet train = FixedProblems.getLinearRegression(1000, RandomUtil.getRandom());
        double eps = train.getTargetValues().mean()/20;
        
        DCDs warmModel = new DCDs();
        warmModel.setEps(eps);
        warmModel.setUseL1(true);
        warmModel.setUseBias(true);
        warmModel.train(train);
        
        
        long start, end;
        
        
        PlattSMO notWarm = new PlattSMO(new LinearKernel(1));
        notWarm.setEpsilon(eps);
        notWarm.setC(1e2);
        
        start = System.currentTimeMillis();
        notWarm.train(train);
        end = System.currentTimeMillis();
        long normTime = (end-start);
        
        
        PlattSMO warm = new PlattSMO(new LinearKernel(1));
        warm.setEpsilon(eps);
        warm.setC(1e2);
        
        start = System.currentTimeMillis();
        warm.train(train, warmModel);
        end = System.currentTimeMillis();
        long warmTime = (end-start);
        
        assertTrue(warmTime < normTime*0.75);
        
    }
}
