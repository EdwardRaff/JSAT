/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.FixedProblems;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.LossFunc;
import jsat.lossfunctions.SquaredLoss;
import jsat.math.decayrates.DecayRate;
import jsat.regression.RegressionDataSet;
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
public class LinearSGDTest
{
    
    public LinearSGDTest()
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
    public void testClassifyBinary()
    {
        System.out.println("binary classifiation");
        
        LinearSGD linearsgd = new LinearSGD(new HingeLoss(), 1e-4, 1e-5);
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());
        
        linearsgd.trainC(train);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), linearsgd.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testClassifyMulti()
    {
        System.out.println("multi class classification");
        
        LinearSGD linearsgd = new LinearSGD(new HingeLoss(), 1e-4, 1e-5);
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());
        
        linearsgd.trainC(train);
        
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), linearsgd.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testRegression()
    {
        System.out.println("regression");
        
        LinearSGD linearsgd = new LinearSGD(new SquaredLoss(), 1e-4, 1e-5);
        
        //SGD needs more iterations/data to learn a really close fit
        
        RegressionDataSet train = FixedProblems.getLinearRegression(10000, new Random());
        
        linearsgd.setEpochs(50);
        linearsgd.train(train);
        
        RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());
        
        for(DataPointPair<Double> dpp : test.getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = linearsgd.regress(dpp.getDataPoint());
            double relErr = (truth-pred)/truth;
            assertEquals(0, relErr, 0.1);
        }
    }
}
