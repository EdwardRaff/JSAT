package jsat.classifiers.svm;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
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
public class DCDsTest
{
    static private ExecutorService threadPool;
    
    public DCDsTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        threadPool = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }
    
    @AfterClass
    public static void tearDownClass()
    {
        threadPool.shutdown();
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
     * Test of trainC method, of class DCDs.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        DCDs instance = new DCDs();
        instance.trainC(train, threadPool);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }

    /**
     * Test of trainC method, of class DCDs.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

        DCDs instance = new DCDs();
        instance.trainC(train);

        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

        for (DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), instance.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testTrain_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");
        Random rand = RandomUtil.getRandom();

        DCDs dcds = new DCDs();
        dcds.train(FixedProblems.getLinearRegression(400, rand), threadPool);

        for (DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = dcds.regress(dpp.getDataPoint());

            double relErr = (truth - pred) / truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }
    
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = RandomUtil.getRandom();

        DCDs dcds = new DCDs();
        dcds.train(FixedProblems.getLinearRegression(400, rand));

        for (DataPointPair<Double> dpp : FixedProblems.getLinearRegression(100, rand).getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = dcds.regress(dpp.getDataPoint());

            double relErr = (truth - pred) / truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }
    
    @Test()
    public void testTrainWarmC()
    {
        ClassificationDataSet train = FixedProblems.getHalfCircles(10000, RandomUtil.getRandom(), 0.1, 0.5);
        
        DCDs warmModel = new DCDs();
        warmModel.trainC(train);
        warmModel.setC(1);
        
        
        
        long start, end;
        
        
        
        DCDs notWarm = new DCDs();
        notWarm.setC(1e1);
        
        start = System.currentTimeMillis();
        notWarm.trainC(train);
        end = System.currentTimeMillis();
        long normTime = (end-start);
        
        DCDs warm = new DCDs();
        warm.setC(1e1);
        
        start = System.currentTimeMillis();
        warm.trainC(train, warmModel);
        end = System.currentTimeMillis();
        long warmTime = (end-start);
        
        assertTrue(warmTime < normTime*0.80);   
    }
    
    @Test()
    public void testTrainWarR()
    {
        RegressionDataSet train = FixedProblems.getSimpleRegression1(4000, RandomUtil.getRandom());
        double eps = train.getTargetValues().mean()/0.9;
        
        DCDs warmModel = new DCDs();
        warmModel.setEps(eps);
        warmModel.train(train);
        
        
        DCDs warm = new DCDs();
        warm.setEps(eps);
        warm.setC(1e1);//too large to train efficently like noraml
        
        long start, end;
        
        start = System.currentTimeMillis();
        warm.train(train, warmModel);
        end = System.currentTimeMillis();
        long warmTime = (end-start);
        
        DCDs notWarm = new DCDs();
        notWarm.setEps(eps);
        notWarm.setC(1e1);//too large to train efficently like noraml
        
        start = System.currentTimeMillis();
        notWarm.train(train);
        end = System.currentTimeMillis();
        long normTime = (end-start);
        
        assertTrue(warmTime < normTime*0.80);   
    }
}
