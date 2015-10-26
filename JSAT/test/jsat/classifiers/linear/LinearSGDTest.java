/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.linear;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.SquaredLoss;
import jsat.math.optimization.stochastic.AdaGrad;
import jsat.math.optimization.stochastic.GradientUpdater;
import jsat.math.optimization.stochastic.RMSProp;
import jsat.math.optimization.stochastic.SimpleSGD;
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

    static boolean[] useBiasOptions = new boolean[]{true, false};
    static GradientUpdater[] updaters = new GradientUpdater[]{new SimpleSGD(), new AdaGrad(), new RMSProp()};
    
    @Test
    public void testClassifyBinary()
    {
        System.out.println("binary classifiation");
        
        for(final boolean useBias : useBiasOptions)
        {
            for(final GradientUpdater gu : updaters)
            {
                final LinearSGD linearsgd = new LinearSGD(new HingeLoss(), 1e-4, 1e-5);
                linearsgd.setUseBias(useBias);
                linearsgd.setGradientUpdater(gu);

                final ClassificationDataSet train = FixedProblems.get2ClassLinear(500, new Random());

                linearsgd.trainC(train);

                final ClassificationDataSet test = FixedProblems.get2ClassLinear(200, new Random());

                for(final DataPointPair<Integer> dpp : test.getAsDPPList()) {
                  assertEquals(dpp.getPair().longValue(), linearsgd.classify(dpp.getDataPoint()).mostLikely());
                }
            }
        }
    }
    
    @Test
    public void testClassifyMulti()
    {
        System.out.println("multi class classification");
        for(final boolean useBias : useBiasOptions)
        {
            for(final GradientUpdater gu : updaters)
            {
                final LinearSGD linearsgd = new LinearSGD(new HingeLoss(), 1e-4, 1e-5);
                linearsgd.setUseBias(useBias);
                linearsgd.setGradientUpdater(gu);

                final ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(500, 6, new Random());

                linearsgd.trainC(train);

                final ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(200, 6, new Random());

                for(final DataPointPair<Integer> dpp : test.getAsDPPList()) {
                  assertEquals(dpp.getPair().longValue(), linearsgd.classify(dpp.getDataPoint()).mostLikely());
                }
            }
        }
    }
    
    @Test
    public void testRegression()
    {
        System.out.println("regression");
        for(final boolean useBias : useBiasOptions)
        {
            for(final GradientUpdater gu : updaters)
            {
                final LinearSGD linearsgd = new LinearSGD(new SquaredLoss(), 0.0, 0.0);
                linearsgd.setUseBias(useBias);
                linearsgd.setGradientUpdater(gu);
                
                //SGD needs more iterations/data to learn a really close fit

                final RegressionDataSet train = FixedProblems.getLinearRegression(10000, new Random());

                linearsgd.setEpochs(50);
                if(!(gu instanceof SimpleSGD))//the others need a higher learning rate than the default
                {
                    linearsgd.setEta(0.5);
                    linearsgd.setEpochs(100);//more iters b/c RMSProp probably isn't the best for this overly simple problem
                }
                linearsgd.train(train);

                final RegressionDataSet test = FixedProblems.getLinearRegression(200, new Random());

                for(final DataPointPair<Double> dpp : test.getAsDPPList())
                {
                    final double truth = dpp.getPair();
                    final double pred = linearsgd.regress(dpp.getDataPoint());
                    final double relErr = (truth-pred)/truth;
                    assertEquals(0, relErr, 0.1);
                }
            }
        }
    }
}
