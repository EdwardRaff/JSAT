package jsat.classifiers.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.FixedProblems;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
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
public class SMIDASTest
{
    
    public SMIDASTest()
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
     * Test of trainC method, of class SMIDAS.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        
        ClassificationDataSet train = FixedProblems.get2ClassLinear(400, new Random());
        
        SMIDAS smidas = new SMIDAS(0.1);
        smidas.setLoss(StochasticSTLinearL1.Loss.LOG);
        smidas.trainC(train);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear(400, new Random());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), smidas.classify(dpp.getDataPoint()).mostLikely());
        
    }

    /**
     * Test of train method, of class SMIDAS.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = new Random(123);
        Vec m0 = new DenseVector(new double[]{1,2,3,4,5,6,7});
        
        Vec trueW = new DenseVector(new double[]{-5,4,-3,2,-1,0,0});
        
        NormalM c0 = new NormalM(m0, Matrix.eye(m0.length()));
        
        RegressionDataSet train = new RegressionDataSet(m0.length(), new CategoricalData[0]);
        
        for(Vec s : c0.sample(500, rand))
            train.addDataPoint(s, new int[0], s.dot(trueW));
        
        SMIDAS smidas = new SMIDAS(0.1);
        smidas.setMinScaled(-1);
        smidas.setLoss(StochasticSTLinearL1.Loss.SQUARED);
        smidas.train(train);
        
        RegressionDataSet test = new RegressionDataSet(m0.length(), new CategoricalData[0]);
        for(Vec s : c0.sample(100, rand))
            test.addDataPoint(s, new int[0], s.dot(trueW));
        
        for(DataPointPair<Double> dpp : test.getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = smidas.regress(dpp.getDataPoint());
            
            double relErr = (truth-pred)/truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }
}
