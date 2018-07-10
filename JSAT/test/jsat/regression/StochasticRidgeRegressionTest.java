package jsat.regression;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.datatransform.DenseSparceTransform;
import jsat.datatransform.LinearTransform;
import jsat.linear.DenseVector;
import jsat.math.decayrates.LinearDecay;
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
public class StochasticRidgeRegressionTest
{
    public StochasticRidgeRegressionTest()
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
    public void testTrainC_RegressionDataSet()
    {
        System.out.println("train");

        for(int batchSize : new int[]{1, 10, 20})
        {
            StochasticRidgeRegression instance = new StochasticRidgeRegression(1e-9, 40, batchSize, 0.01);
            instance.setEpochs(100);

            RegressionDataSet train = FixedProblems.getLinearRegression(500, RandomUtil.getRandom());
            
            LinearTransform lt = new LinearTransform(train);
            train.applyTransform(lt);
            for(int i = 0; i < 20; i++)
                train.addDataPoint(DenseVector.random(train.getNumNumericalVars()), train.getTargetValues().mean());
            if(batchSize == 10)
                train.applyTransform(new DenseSparceTransform(1));
            RegressionDataSet test = FixedProblems.getLinearRegression(100, RandomUtil.getRandom());
            test.applyTransform(lt);

            RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train);
            rme.evaluateTestSet(test);

            assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.35);
        }

    }

    @Test
    public void testTrainC_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");

        for(int batchSize : new int[]{1, 10, 20})
        {
            StochasticRidgeRegression instance = new StochasticRidgeRegression(1e-9, 40, batchSize, 0.01);
            instance.setEpochs(100);

            RegressionDataSet train = FixedProblems.getLinearRegression(500, RandomUtil.getRandom());
            LinearTransform lt = new LinearTransform(train);
            train.applyTransform(lt);
            
            for(int i = 0; i < 20; i++)
                train.addDataPoint(DenseVector.random(train.getNumNumericalVars()), train.getTargetValues().mean());
            if(batchSize == 10)
                train.applyTransform(new DenseSparceTransform(1));
            RegressionDataSet test = FixedProblems.getLinearRegression(100, RandomUtil.getRandom());
            test.applyTransform(lt);

            RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train, true);
            rme.evaluateTestSet(test);

            assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.35);
        }
    }
    
    @Test
    public void testClone()
    {
        System.out.println("clone");

        for(int batchSize : new int[]{1, 10, 20})
        {
            StochasticRidgeRegression instance = new StochasticRidgeRegression(1e-9, 40, batchSize, 0.01);
            instance.setEpochs(100);

            RegressionDataSet t1 = FixedProblems.getLinearRegression(5000, RandomUtil.getRandom());
            for(int i = 0; i < 20; i++)
                t1.addDataPoint(DenseVector.random(t1.getNumNumericalVars()), t1.getTargetValues().mean());
            RegressionDataSet t2 = FixedProblems.getLinearRegression(1000, RandomUtil.getRandom());
            t2.applyTransform(new LinearTransform(t2, -1, 1));
            
            if(batchSize == 10)
            {
                instance.setLearningDecay(new LinearDecay(0.5, 500));//just to exercise another part of the code 
                t1.applyTransform(new DenseSparceTransform(1));
            }

            instance = instance.clone();

            instance.train(t1);

            StochasticRidgeRegression result = instance.clone();
            for (int i = 0; i < t1.size(); i++)
                assertEquals(t1.getTargetValue(i), result.regress(t1.getDataPoint(i)), t1.getTargetValues().mean());
            result.train(t2);

            for (int i = 0; i < t1.size(); i++)
                assertEquals(t1.getTargetValue(i), instance.regress(t1.getDataPoint(i)), t1.getTargetValues().mean());

            for (int i = 0; i < t2.size(); i++)
                assertEquals(t2.getTargetValue(i), result.regress(t2.getDataPoint(i)), t2.getTargetValues().mean()*0.5);
        }

    }
}
