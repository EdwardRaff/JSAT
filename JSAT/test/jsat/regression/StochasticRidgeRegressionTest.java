package jsat.regression;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.datatransform.DenseSparceTransform;
import jsat.datatransform.LinearTransform;
import jsat.linear.DenseVector;
import jsat.math.decayrates.LinearDecay;
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
public class StochasticRidgeRegressionTest
{
    static ExecutorService ex;
    
    public StochasticRidgeRegressionTest()
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
    public void testTrainC_RegressionDataSet()
    {
        System.out.println("train");

        for(int batchSize : new int[]{1, 10, 20})
        {
            StochasticRidgeRegression instance = new StochasticRidgeRegression(1e-9, 40, batchSize, 0.1);

            RegressionDataSet train = FixedProblems.getLinearRegression(500, new XORWOW());
            for(int i = 0; i < 20; i++)
                train.addDataPoint(DenseVector.random(train.getNumNumericalVars()), train.getTargetValues().mean());
            if(batchSize == 10)
                train.applyTransform(new DenseSparceTransform(1));
            RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW());

            RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train);
            rme.evaluateTestSet(test);

            assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.25);
        }

    }

    @Test
    public void testTrainC_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");

        for(int batchSize : new int[]{1, 10, 20})
        {
            StochasticRidgeRegression instance = new StochasticRidgeRegression(1e-9, 40, batchSize, 0.1);

            ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

            RegressionDataSet train = FixedProblems.getLinearRegression(500, new XORWOW(1234));
            for(int i = 0; i < 20; i++)
                train.addDataPoint(DenseVector.random(train.getNumNumericalVars()), train.getTargetValues().mean());
            if(batchSize == 10)
                train.applyTransform(new DenseSparceTransform(1));
            RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW(1234));

            RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train, ex);
            rme.evaluateTestSet(test);

            assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.25);

            ex.shutdownNow();
        }
    }
    
    @Test
    public void testClone()
    {
        System.out.println("clone");

        for(int batchSize : new int[]{1, 10, 20})
        {
            StochasticRidgeRegression instance = new StochasticRidgeRegression(1e-9, 40, batchSize, 0.1);

            RegressionDataSet t1 = FixedProblems.getLinearRegression(500, new XORWOW(1234));
            for(int i = 0; i < 20; i++)
                t1.addDataPoint(DenseVector.random(t1.getNumNumericalVars()), t1.getTargetValues().mean());
            RegressionDataSet t2 = FixedProblems.getLinearRegression(100, new XORWOW(1234));
            t2.applyTransform(new LinearTransform(t2, -1, 1));
            
            if(batchSize == 10)
            {
                instance.setLearningDecay(new LinearDecay(0.5, 500));//just to exercise another part of the code 
                t1.applyTransform(new DenseSparceTransform(1));
            }

            instance = instance.clone();

            instance.train(t1);

            StochasticRidgeRegression result = instance.clone();
            for (int i = 0; i < t1.getSampleSize(); i++)
                assertEquals(t1.getTargetValue(i), result.regress(t1.getDataPoint(i)), t1.getTargetValues().mean());
            result.train(t2);

            for (int i = 0; i < t1.getSampleSize(); i++)
                assertEquals(t1.getTargetValue(i), instance.regress(t1.getDataPoint(i)), t1.getTargetValues().mean());

            for (int i = 0; i < t2.getSampleSize(); i++)
                assertEquals(t2.getTargetValue(i), result.regress(t2.getDataPoint(i)), t2.getTargetValues().mean()*0.5);
        }

    }
}
