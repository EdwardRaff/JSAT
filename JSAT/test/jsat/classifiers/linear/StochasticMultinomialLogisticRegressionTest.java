/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.linear;

import java.util.Random;
import jsat.FixedProblems;
import jsat.classifiers.*;
import jsat.exceptions.UntrainedModelException;
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
public class StochasticMultinomialLogisticRegressionTest
{
    
    public StochasticMultinomialLogisticRegressionTest()
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


    /**
     * Test of trainC method, of class StochasticMultinomialLogisticRegression.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        
        final ClassificationDataSet train = FixedProblems.get2ClassLinear(400, new Random());
        
        
        for(final StochasticMultinomialLogisticRegression.Prior prior : StochasticMultinomialLogisticRegression.Prior.values())
        {
        
            final StochasticMultinomialLogisticRegression smlgr = new StochasticMultinomialLogisticRegression();
            smlgr.setPrior(prior);
            smlgr.trainC(train);

            final ClassificationDataSet test = FixedProblems.get2ClassLinear(400, new Random());

            for(final DataPointPair<Integer> dpp : test.getAsDPPList()) {
              assertEquals(dpp.getPair().longValue(), smlgr.classify(dpp.getDataPoint()).mostLikely());
            }
        }
    }

    /**
     * Test of clone method, of class StochasticMultinomialLogisticRegression.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        final StochasticMultinomialLogisticRegression smlgr = new StochasticMultinomialLogisticRegression();
        
        final Classifier cloned = smlgr.clone();
        
        final ClassificationDataSet train = FixedProblems.get2ClassLinear(400, new Random());
        cloned.trainC(train);
        
        
        
        try
        {
            smlgr.classify(train.getDataPoint(0));
            fail("Exception should have occured");
        }
        catch(final UntrainedModelException ex)
        {
            
        }
        
        
        train.classSampleCount(train.getDataPointCategory(0));
        
    }

}
