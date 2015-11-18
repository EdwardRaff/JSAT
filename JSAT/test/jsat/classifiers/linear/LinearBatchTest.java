package jsat.classifiers.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import jsat.lossfunctions.*;
import jsat.regression.RegressionDataSet;
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
public class LinearBatchTest
{
    static ExecutorService ex;
    
    public LinearBatchTest()
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
    public void testClassifyBinary()
    {
        System.out.println("binary classifiation");
        
        for(boolean useBias : new boolean[]{false, true})
        {
            LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);

            ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.trainC(train);

            ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

            for(DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
        }
    }
    
    @Test()
    public void testTrainWarmCFast()
    {
        ClassificationDataSet train = FixedProblems.get2ClassLinear(10000, new XORWOW());
        
        LinearSGD warmModel = new LinearSGD(new SoftmaxLoss(), 1e-4, 0);
        warmModel.setEpochs(20);
        warmModel.trainC(train);
        
        
        long start, end;
        
        
        LinearBatch notWarm = new LinearBatch(new SoftmaxLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        notWarm.trainC(train);
        end = System.currentTimeMillis();
        long normTime = (end-start);
        
        
        LinearBatch warm = new LinearBatch(new SoftmaxLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        warm.trainC(train, warmModel);
        end = System.currentTimeMillis();
        long warmTime = (end-start);
        
        assertTrue("Warm start wasn't faster? "+warmTime + " vs " + normTime,warmTime < normTime*0.95);
    }
    
    @Test
    public void testClassifyBinaryMT()
    {
        System.out.println("binary classifiation MT");
        
        for(boolean useBias : new boolean[]{false, true})
        {
            LinearBatch linearBatch = new LinearBatch(new LogisticLoss(), 1e-4);

            ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.trainC(train, ex);

            ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());
        
            for(DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
        }
    }
    
    @Test
    public void testClassifyMulti()
    {
        System.out.println("multi class classification");
        for(boolean useBias : new boolean[]{false, true})
        {
            LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);

            ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.trainC(train);

            ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());

            for(DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
        }
    }
    
    @Test()
    public void testTrainWarmCMultieFast()
    {
        ClassificationDataSet train = FixedProblems.getHalfCircles(1000, new XORWOW(), 0.1, 1.0, 2.0, 5.0);
        
        LinearSGD warmModel = new LinearSGD(new SoftmaxLoss(), 1e-4, 0);
        warmModel.setEpochs(20);
        warmModel.trainC(train);
        
        
        long start, end;
        
        
        LinearBatch notWarm = new LinearBatch(new SoftmaxLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        notWarm.trainC(train);
        end = System.currentTimeMillis();
        long normTime = (end-start);
        
        
        LinearBatch warm = new LinearBatch(new SoftmaxLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        warm.trainC(train, warmModel);
        end = System.currentTimeMillis();
        long warmTime = (end-start);
        
        assertTrue("Warm start wasn't faster? "+warmTime + " vs " + normTime,warmTime < normTime*0.95);
    }
    
    @Test
    public void testClassifyMultiMT()
    {
        System.out.println("multi class classification MT");
        
        for(boolean useBias : new boolean[]{false, true})
        {
            LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);

            ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.trainC(train, ex);

            ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());

            for(DataPointPair<Integer> dpp : test.getAsDPPList())
                assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
        }
    }
    
    @Test
    public void testRegression()
    {
        System.out.println("regression");
        
        
        for(boolean useBias : new boolean[]{false, true})
        {
            LinearBatch linearBatch = new LinearBatch(new SquaredLoss(), 1e-4);
            RegressionDataSet train = FixedProblems.getLinearRegression(500, new Random());
            
            linearBatch.setUseBiasTerm(useBias);
            
            linearBatch.train(train);

            RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());

            for(DataPointPair<Double> dpp : test.getAsDPPList())
            {
                double truth = dpp.getPair();
                double pred = linearBatch.regress(dpp.getDataPoint());
                double relErr = (truth-pred)/truth;
                assertEquals(0, relErr, 0.1);
            }
        }
    }
    
    @Test
    public void testRegressionMT()
    {
        System.out.println("regression MT");
        for(boolean useBias : new boolean[]{false, true})
        {
            LinearBatch linearBatch = new LinearBatch(new SquaredLoss(), 1e-4);
            RegressionDataSet train = FixedProblems.getLinearRegression(500, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.train(train, ex);

            RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());

            for(DataPointPair<Double> dpp : test.getAsDPPList())
            {
                double truth = dpp.getPair();
                double pred = linearBatch.regress(dpp.getDataPoint());
                double relErr = (truth-pred)/truth;
                assertEquals(0, relErr, 0.01);
            }
        }
    }

    @Test()
    public void testTrainWarmRFast()
    {
        RegressionDataSet train = FixedProblems.getLinearRegression(100000, new XORWOW());
        
        LinearBatch warmModel = new LinearBatch(new SquaredLoss(), 1e-4);
        warmModel.train(train);
        
        
        long start, end;
        
        
        LinearBatch notWarm = new LinearBatch(new SquaredLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        notWarm.train(train);
        end = System.currentTimeMillis();
        long normTime = (end-start);
        
        
        LinearBatch warm = new LinearBatch(new SquaredLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        warm.train(train, warmModel);
        end = System.currentTimeMillis();
        long warmTime = (end-start);
        
        assertTrue("Warm start wasn't faster? "+warmTime + " vs " + normTime,warmTime < normTime*0.95);
    }
}
