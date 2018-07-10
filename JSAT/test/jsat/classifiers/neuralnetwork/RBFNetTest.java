/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.neuralnetwork;

import java.util.EnumSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.regression.RegressionDataSet;
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
public class RBFNetTest
{
    /*
     * RBF is a bit heuristic and works best with more data - so the training set size is enlarged
     */
    
    public RBFNetTest()
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
     * Test of train method, of class RBFNet.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(2000, RandomUtil.getRandom());
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(200, RandomUtil.getRandom());
       
        for(RBFNet.Phase1Learner p1l : RBFNet.Phase1Learner.values())
            for(RBFNet.Phase2Learner p2l : RBFNet.Phase2Learner.values())
            {
                RBFNet net = new RBFNet(25).clone();
                net.setAlpha(1);//CLOSEST_OPPOSITE_CENTROID needs a smaller value, shoudld be fine for others on this data set 
                net.setPhase1Learner(p1l);
                net.setPhase2Learner(p2l);
                net.train(trainSet, true);

                net = net.clone();
                for (int i = 0; i < testSet.size(); i++)
                    assertEquals(testSet.getDataPointCategory(i), net.classify(testSet.getDataPoint(i)).mostLikely());
            }
    }

    /**
     * Test of train method, of class RBFNet.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet trainSet = FixedProblems.getInnerOuterCircle(2000, RandomUtil.getRandom());
        ClassificationDataSet testSet = FixedProblems.getInnerOuterCircle(200, RandomUtil.getRandom());
        
        
        for(RBFNet.Phase1Learner p1l : RBFNet.Phase1Learner.values())
            for(RBFNet.Phase2Learner p2l : RBFNet.Phase2Learner.values())
            {
                RBFNet net = new RBFNet(25);
                net.setAlpha(1);//CLOSEST_OPPOSITE_CENTROID needs a smaller value, shoudld be fine for others on this data set
                net.setPhase1Learner(p1l);
                net.setPhase2Learner(p2l);
                net = net.clone();
                net.train(trainSet);
                net = net.clone();
                for (int i = 0; i < testSet.size(); i++)
                    assertEquals(testSet.getDataPointCategory(i), net.classify(testSet.getDataPoint(i)).mostLikely());
            }
        
    }

    /**
     * Test of train method, of class RBFNet.
     */
    @Test
    public void testTrain_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");
        
        RegressionDataSet trainSet =  FixedProblems.getSimpleRegression1(2000, RandomUtil.getRandom());
        RegressionDataSet testSet =  FixedProblems.getSimpleRegression1(200, RandomUtil.getRandom());
        
        for(RBFNet.Phase1Learner p1l : RBFNet.Phase1Learner.values())
            for(RBFNet.Phase2Learner p2l :  EnumSet.complementOf(EnumSet.of(RBFNet.Phase2Learner.CLOSEST_OPPOSITE_CENTROID)))
            {
                RBFNet net = new RBFNet(25);
                net.setAlpha(1);//CLOSEST_OPPOSITE_CENTROID needs a smaller value, shoudld be fine for others on this data set 
                net.setPhase1Learner(p1l);
                net.setPhase2Learner(p2l);
                net = net.clone();
                net.train(trainSet, true);
                net = net.clone();
                
                double errors = 0;
                for (int i = 0; i < testSet.size(); i++)
                    errors += Math.pow(testSet.getTargetValue(i) - net.regress(testSet.getDataPoint(i)), 2);
                assertTrue(errors/testSet.size() < 1);
            }
    }

    /**
     * Test of train method, of class RBFNet.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        RegressionDataSet trainSet =  FixedProblems.getSimpleRegression1(2000, RandomUtil.getRandom());
        RegressionDataSet testSet =  FixedProblems.getSimpleRegression1(200, RandomUtil.getRandom());
        
        for(RBFNet.Phase1Learner p1l : RBFNet.Phase1Learner.values())
            for(RBFNet.Phase2Learner p2l :  EnumSet.complementOf(EnumSet.of(RBFNet.Phase2Learner.CLOSEST_OPPOSITE_CENTROID)))
            {
                RBFNet net = new RBFNet(25);
                net.setAlpha(1);//CLOSEST_OPPOSITE_CENTROID needs a smaller value, shoudld be fine for others on this data set 
                net.setPhase1Learner(p1l);
                net.setPhase2Learner(p2l);
                net = net.clone();
                net.train(trainSet);
                net = net.clone();
                
                double errors = 0;
                for (int i = 0; i < testSet.size(); i++)
                    errors += Math.pow(testSet.getTargetValue(i) - net.regress(testSet.getDataPoint(i)), 2);
                assertTrue(errors/testSet.size() < 1);
            }
        
    }

}
