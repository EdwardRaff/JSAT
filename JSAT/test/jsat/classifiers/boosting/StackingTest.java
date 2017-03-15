
package jsat.classifiers.boosting;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.OneVSAll;
import jsat.classifiers.linear.LinearBatch;
import jsat.classifiers.linear.LogisticRegressionDCD;
import jsat.lossfunctions.AbsoluteLoss;
import jsat.lossfunctions.HuberLoss;
import jsat.lossfunctions.SoftmaxLoss;
import jsat.lossfunctions.SquaredLoss;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
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
public class StackingTest
{
    static ExecutorService ex;
    
    public StackingTest()
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
        
        Stacking stacking = new Stacking(new LogisticRegressionDCD(), new LinearBatch(new SoftmaxLoss(), 1e-15), new LinearBatch(new SoftmaxLoss(), 100), new LinearBatch(new SoftmaxLoss(), 1e10));
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(500, RandomUtil.getRandom());
        
        stacking = stacking.clone();
        stacking.trainC(train);
        stacking = stacking.clone();
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), stacking.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testClassifyBinaryMT()
    {
        System.out.println("binary classifiation MT");
        
        Stacking stacking = new Stacking(new LogisticRegressionDCD(), new LinearBatch(new SoftmaxLoss(), 1e-15), new LinearBatch(new SoftmaxLoss(), 100), new LinearBatch(new SoftmaxLoss(), 1e10));
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(500, RandomUtil.getRandom());
        
        stacking = stacking.clone();
        stacking.trainC(train, ex);
        stacking = stacking.clone();
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), stacking.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testClassifyMulti()
    {
        Stacking stacking = new Stacking(new OneVSAll(new LogisticRegressionDCD(), true), new LinearBatch(new SoftmaxLoss(), 1e-15), new LinearBatch(new SoftmaxLoss(), 100), new LinearBatch(new SoftmaxLoss(), 1e10));
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, RandomUtil.getRandom());
        
        stacking = stacking.clone();
        stacking.trainC(train);
        stacking = stacking.clone();
        
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), stacking.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testClassifyMultiMT()
    {
        System.out.println("multi class classification MT");
        
        Stacking stacking = new Stacking(new OneVSAll(new LogisticRegressionDCD(), true), new LinearBatch(new SoftmaxLoss(), 1e-15), new LinearBatch(new SoftmaxLoss(), 100), new LinearBatch(new SoftmaxLoss(), 1e10));
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, RandomUtil.getRandom());
        
        stacking = stacking.clone();
        stacking.trainC(train, ex);
        stacking = stacking.clone();
        
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), stacking.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testRegression()
    {
        System.out.println("regression");
        
        Stacking stacking = new Stacking((Regressor)new LinearBatch(new AbsoluteLoss(), 1e-10), new LinearBatch(new SquaredLoss(), 1e-15), new LinearBatch(new AbsoluteLoss(), 100), new LinearBatch(new HuberLoss(), 1e1));
        RegressionDataSet train = FixedProblems.getLinearRegression(500, RandomUtil.getRandom());
        
        stacking = stacking.clone();
        stacking.train(train);
        stacking = stacking.clone();
        
        RegressionDataSet test = FixedProblems.getLinearRegression(200, RandomUtil.getRandom());
        
        for(DataPointPair<Double> dpp : test.getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = stacking.regress(dpp.getDataPoint());
            double relErr = (truth-pred)/truth;
            assertEquals(0, relErr, 0.1);
        }
    }
    
    @Test
    public void testRegressionMT()
    {
        System.out.println("regression MT");
        
        Stacking stacking = new Stacking((Regressor)new LinearBatch(new AbsoluteLoss(), 1e-10), new LinearBatch(new SquaredLoss(), 1e-15), new LinearBatch(new AbsoluteLoss(), 100), new LinearBatch(new HuberLoss(), 1e1));
        RegressionDataSet train = FixedProblems.getLinearRegression(500, RandomUtil.getRandom());
        
        stacking = stacking.clone();
        stacking.train(train, ex);
        stacking = stacking.clone();
        
        RegressionDataSet test = FixedProblems.getLinearRegression(200, RandomUtil.getRandom());
        
        for(DataPointPair<Double> dpp : test.getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = stacking.regress(dpp.getDataPoint());
            double relErr = (truth-pred)/truth;
            assertEquals(0, relErr, 0.1);
        }
    }
    
}
