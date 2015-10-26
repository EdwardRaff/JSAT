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
        
        for(final boolean useBias : new boolean[]{false, true})
        {
            final LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);

            final ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.trainC(train);

            final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

            for(final DataPointPair<Integer> dpp : test.getAsDPPList()) {
              assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
            }
        }
    }
    
    @Test()
    public void testTrainWarmCFast()
    {
        final ClassificationDataSet train = FixedProblems.get2ClassLinear(10000, new XORWOW());
        
        final LinearSGD warmModel = new LinearSGD(new SoftmaxLoss(), 1e-4, 0);
        warmModel.setEpochs(20);
        warmModel.trainC(train);
        
        
        long start, end;
        
        
        final LinearBatch notWarm = new LinearBatch(new SoftmaxLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        notWarm.trainC(train);
        end = System.currentTimeMillis();
        final long normTime = (end-start);
        
        
        final LinearBatch warm = new LinearBatch(new SoftmaxLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        warm.trainC(train, warmModel);
        end = System.currentTimeMillis();
        final long warmTime = (end-start);
        
        assertTrue(warmTime < normTime*0.75);
    }
    
    @Test
    public void testClassifyBinaryMT()
    {
        System.out.println("binary classifiation MT");
        
        for(final boolean useBias : new boolean[]{false, true})
        {
            final LinearBatch linearBatch = new LinearBatch(new LogisticLoss(), 1e-4);

            final ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.trainC(train, ex);

            final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());
        
            for(final DataPointPair<Integer> dpp : test.getAsDPPList()) {
              assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
            }
        }
    }
    
    @Test
    public void testClassifyMulti()
    {
        System.out.println("multi class classification");
        for(final boolean useBias : new boolean[]{false, true})
        {
            final LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);

            final ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.trainC(train);

            final ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());

            for(final DataPointPair<Integer> dpp : test.getAsDPPList()) {
              assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
            }
        }
    }
    
    @Test()
    public void testTrainWarmCMultieFast()
    {
        final ClassificationDataSet train = FixedProblems.getHalfCircles(1000, new XORWOW(), 0.1, 1.0, 2.0, 5.0);
        
        final LinearSGD warmModel = new LinearSGD(new SoftmaxLoss(), 1e-4, 0);
        warmModel.setEpochs(20);
        warmModel.trainC(train);
        
        
        long start, end;
        
        
        final LinearBatch notWarm = new LinearBatch(new SoftmaxLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        notWarm.trainC(train);
        end = System.currentTimeMillis();
        final long normTime = (end-start);
        
        
        final LinearBatch warm = new LinearBatch(new SoftmaxLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        warm.trainC(train, warmModel);
        end = System.currentTimeMillis();
        final long warmTime = (end-start);
        
        assertTrue(warmTime < normTime*0.75);
    }
    
    @Test
    public void testClassifyMultiMT()
    {
        System.out.println("multi class classification MT");
        
        for(final boolean useBias : new boolean[]{false, true})
        {
            final LinearBatch linearBatch = new LinearBatch(new HingeLoss(), 1e-4);

            final ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.trainC(train, ex);

            final ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());

            for(final DataPointPair<Integer> dpp : test.getAsDPPList()) {
              assertEquals(dpp.getPair().longValue(), linearBatch.classify(dpp.getDataPoint()).mostLikely());
            }
        }
    }
    
    @Test
    public void testRegression()
    {
        System.out.println("regression");
        
        
        for(final boolean useBias : new boolean[]{false, true})
        {
            final LinearBatch linearBatch = new LinearBatch(new SquaredLoss(), 1e-4);
            final RegressionDataSet train = FixedProblems.getLinearRegression(500, new Random());
            
            linearBatch.setUseBiasTerm(useBias);
            
            linearBatch.train(train);

            final RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());

            for(final DataPointPair<Double> dpp : test.getAsDPPList())
            {
                final double truth = dpp.getPair();
                final double pred = linearBatch.regress(dpp.getDataPoint());
                final double relErr = (truth-pred)/truth;
                assertEquals(0, relErr, 0.1);
            }
        }
    }
    
    @Test
    public void testRegressionMT()
    {
        System.out.println("regression MT");
        for(final boolean useBias : new boolean[]{false, true})
        {
            final LinearBatch linearBatch = new LinearBatch(new SquaredLoss(), 1e-4);
            final RegressionDataSet train = FixedProblems.getLinearRegression(500, new Random());

            linearBatch.setUseBiasTerm(useBias);
            linearBatch.train(train, ex);

            final RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());

            for(final DataPointPair<Double> dpp : test.getAsDPPList())
            {
                final double truth = dpp.getPair();
                final double pred = linearBatch.regress(dpp.getDataPoint());
                final double relErr = (truth-pred)/truth;
                assertEquals(0, relErr, 0.01);
            }
        }
    }

    @Test()
    public void testTrainWarmRFast()
    {
        final RegressionDataSet train = FixedProblems.getLinearRegression(100000, new XORWOW());
        
        final LinearBatch warmModel = new LinearBatch(new SquaredLoss(), 1e-4);
        warmModel.train(train);
        
        
        long start, end;
        
        
        final LinearBatch notWarm = new LinearBatch(new SquaredLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        notWarm.train(train);
        end = System.currentTimeMillis();
        final long normTime = (end-start);
        
        
        final LinearBatch warm = new LinearBatch(new SquaredLoss(), 1e-4);
        
        start = System.currentTimeMillis();
        warm.train(train, warmModel);
        end = System.currentTimeMillis();
        final long warmTime = (end-start);
        
        assertTrue(warmTime < normTime*0.75);
    }
}
