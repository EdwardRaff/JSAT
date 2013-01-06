
package jsat.classifiers.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionBase;
import jsat.regression.RegressionDataSet;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward
 */
public class LinearL1SCDTest
{
    
    public LinearL1SCDTest()
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
     * Test of train method, of class LinearL1SCD.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("train");
        Random rand = new Random(123);
        Vec m0 = new DenseVector(new double[]{12,14,25,31,10,9,1});
        
        Vec trueW = new DenseVector(new double[]{2,-4,3,-1,1,0,0});
        
        NormalM c0 = new NormalM(m0, Matrix.eye(m0.length()));
        
        RegressionDataSet train = new RegressionDataSet(m0.length(), new CategoricalData[0]);
        
        for(Vec s : c0.sample(600, rand))
            train.addDataPoint(s, new int[0], s.dot(trueW));
        
        LinearL1SCD scd = new LinearL1SCD();
        scd.setMinScaled(-1);
        scd.setLoss(StochasticSTLinearL1.Loss.SQUARED);
        scd.train(train);
        
        RegressionDataSet test = new RegressionDataSet(m0.length(), new CategoricalData[0]);
        for(Vec s : c0.sample(100, rand))
            test.addDataPoint(s, new int[0], s.dot(trueW));
        
        for(DataPointPair<Double> dpp : test.getAsDPPList())
        {
            double truth = dpp.getPair();
            double pred = scd.regress(dpp.getDataPoint());
            
            double relErr = (truth-pred)/truth;
            assertEquals(0.0, relErr, 0.1);//Give it a decent wiggle room b/c of regularization
        }
    }


    /**
     * Test of trainC method, of class LinearL1SCD.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        Vec m0 = new DenseVector(new double[]{12,14,25,31,10,9,1});
        Vec m1 = new DenseVector(new double[]{-9,-7,-13,-6,-11,-9,-1});
        
        NormalM c0 = new NormalM(m0, Matrix.eye(m0.length()));
        NormalM c1 = new NormalM(m1, Matrix.eye(m0.length()));
        
        ClassificationDataSet train = new ClassificationDataSet(m0.length(), new CategoricalData[0], new CategoricalData(2));
        
        for(Vec s : c0.sample(400, new Random()))
            train.addDataPoint(s, new int[0], 0);
        for(Vec s : c1.sample(400, new Random()))
            train.addDataPoint(s, new int[0], 1);
        
        LinearL1SCD scd = new LinearL1SCD();
        scd.setLoss(StochasticSTLinearL1.Loss.LOG);
        scd.trainC(train);
        
        ClassificationDataSet test = new ClassificationDataSet(m0.length(), new CategoricalData[0], new CategoricalData(2));
        for(Vec s : c0.sample(100, new Random()))
            test.addDataPoint(s, new int[0], 0);
        for(Vec s : c1.sample(100, new Random()))
            test.addDataPoint(s, new int[0], 1);
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), scd.classify(dpp.getDataPoint()).mostLikely());
    }
}
