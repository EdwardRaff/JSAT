package jsat.classifiers.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.*;
import jsat.datatransform.LinearTransform;
import jsat.datatransform.PNormNormalization;
import jsat.linear.*;
import jsat.lossfunctions.*;
import jsat.math.OnLineStatistics;
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
public class SDCATest
{
    /*
     * This test case is based off of the grouping example in the Elatic Net 
     * Paper Zou, H.,&amp;Hastie, T. (2005). Regularization and variable selection
     * via the elastic net. Journal of the Royal Statistical Society, Series B, 
     * 67(2), 301–320. doi:10.1111/j.1467-9868.2005.00503.x
     */
    
   
    public SDCATest()
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
     * Test of train method, of class LogisticRegressionDCD.
     */
    @Test
    public void testTrain_RegressionDataSet()
    {
        System.out.println("trainR");
        for(double alpha : new double[]{0.0, 0.5, 1.0})
            for(LossR loss : new LossR[]{new SquaredLoss(), new AbsoluteLoss(), new HuberLoss(), new EpsilonInsensitiveLoss(1.0)})
            {
                RegressionDataSet train = FixedProblems.getLinearRegression(400, RandomUtil.getRandom());

                SDCA sdca = new SDCA();
                sdca.setLoss(loss);
                sdca.setTolerance(1e-10);
                sdca.setLambda(1.0/train.size());
                sdca.setAlpha(alpha);
                sdca.train(train);

                RegressionDataSet test = FixedProblems.getLinearRegression(100, RandomUtil.getRandom());

                OnLineStatistics avgRelError = new OnLineStatistics();
                for(DataPointPair<Double> dpp : test.getAsDPPList())
                {
                    double truth = dpp.getPair();
                    double pred = sdca.regress(dpp.getDataPoint());

                    double relErr = (truth-pred)/truth;
                    avgRelError.add(relErr);
                }
                if(loss instanceof AbsoluteLoss || loss instanceof EpsilonInsensitiveLoss)//sensative to small errors make it a little off at time
                    assertEquals("Loss: " + loss.toString() + " alpha: " + alpha, 0.0, avgRelError.getMean(), 0.2);
                else
                    assertEquals("Loss: " + loss.toString() + " alpha: " + alpha, 0.0, avgRelError.getMean(), 0.01);
            }
    }
    
    /**
     * Test of train method, of class LogisticRegressionDCD.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        for(double alpha : new double[]{0.0, 0.5, 1.0})
            for(LossC loss : new LossC[]{new LogisticLoss(), new HingeLoss()})
            {
                ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

                SDCA sdca = new SDCA();
                sdca.setLoss(loss);
                sdca.setLambda(1.0/train.size());
                sdca.setAlpha(alpha);
                sdca.train(train, true);

                ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

                for(DataPointPair<Integer> dpp : test.getAsDPPList())
                    assertEquals(dpp.getPair().longValue(), sdca.classify(dpp.getDataPoint()).mostLikely());
            }
    }

    /**
     * Test of train method, of class LogisticRegressionDCD.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        for(double alpha : new double[]{0.0, 0.5, 1.0})
            for(LossC loss : new LossC[]{new LogisticLoss(), new HingeLoss()})
            {
                ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

                SDCA sdca = new SDCA();
                sdca.setLoss(loss);
                sdca.setLambda(1.0/train.size());
                sdca.setAlpha(alpha);
                sdca.train(train);

                ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

                for(DataPointPair<Integer> dpp : test.getAsDPPList())
                    assertEquals(dpp.getPair().longValue(), sdca.classify(dpp.getDataPoint()).mostLikely());
            }
    }
    
    @Test
    public void testScale()
    {
        System.out.println("testScale");
        ClassificationDataSet train = FixedProblems.get2ClassLinear(1000, RandomUtil.getRandom());
        
        Vec base = null;
        for(double max : new double[]{1.0, 2.0, 4.0, 5.0, 6.0, 10.0, 20.0, 50.0})
        {
            SDCA sdca = new SDCA();
            
            sdca.setUseBias(false);//bias term makes scaling non-trivial, so remove from this test
            sdca.setLoss(new LogisticLoss());
            sdca.setLambda(1.0 / train.size());
            sdca.setAlpha(0.0);
            
            ClassificationDataSet t = train.shallowClone();
            t.applyTransform(new LinearTransform(t, 0, max));
            
            sdca.train(t);
            if(base == null)
                base = sdca.getRawWeight(0).clone();
            else
                assertTrue("Failed on scale " + max, base.equals(sdca.getRawWeight(0).multiply(max), 0.1));
//            System.out.println(sdca.getRawWeight(0).multiply(max));
//            System.out.println(sdca.getBias(0));
        }
    }

    /**
     * Test of setLambda method, of class NewGLMNET.
     */
    @Test
    public void testSetC()
    {
        System.out.println("train");
//        for (int round = 0; round < 100; round++)
        {
            for (int attempts = 5; attempts >= 0; attempts--)
            {
                Random rand = RandomUtil.getRandom();
                ClassificationDataSet data = new ClassificationDataSet(6, new CategoricalData[0], new CategoricalData(2));
                /**
                 * B/c of the what SDCA works, it has trouble picking just 1 of
                 * perfectly correlated features. So we will make a 2nd version
                 * of the dataset which has 1 pure strong feature, 2 weak
                 * features with noise, and 3 weak features.
                 */
                ClassificationDataSet dataN = new ClassificationDataSet(6, new CategoricalData[0], new CategoricalData(2));

                for (int i = 0; i < 500; i++)
                {
                    double Z1 = rand.nextDouble() * 20 - 10;
                    double Z2 = rand.nextDouble() * 20 - 10;
                    Vec v = DenseVector.toDenseVec(Z1, -Z1, Z1, Z2, -Z2, Z2);
                    data.addDataPoint(v, (int) (Math.signum(Z1 + 0.1 * Z2) + 1) / 2);
                    
                    double eps_1 = rand.nextGaussian()*10;
                    double eps_2 = rand.nextGaussian()*10;
                    v = DenseVector.toDenseVec(Z1, -Z1/10 + eps_1, Z1/10+ eps_2, Z2, -Z2, Z2);
                    dataN.addDataPoint(v, (int) (Math.signum(Z1 + 0.1 * Z2) + 1) / 2);
                }
                data.applyTransform(new PNormNormalization());
                dataN.applyTransform(new PNormNormalization());

                for (LossC loss : new LossC[]{new LogisticLoss(), new HingeLoss()})
                {
                    Vec w = new ConstantVector(1.0, 6);
                    SDCA sdca = new SDCA();
                    sdca.setLoss(loss);

                    double maxLam = LinearTools.maxLambdaLogisticL1(data);

                    sdca.setMaxIters(100);
                    sdca.setUseBias(false);
                    
                    sdca.setAlpha(1.0);
                    
                    sdca.setLambda(maxLam);
                    double search_const = 0.025;
                    while(w.nnz() != 1)// I should be able to find a value of lambda that results in only 1 feature
                    {//SDCA requires a bit more searching b/c it behaved differently than normal coordinate descent solvers when selecting features
                        do
                        {
                            sdca.setLambda(sdca.getLambda() * (1+search_const));
                            sdca.train(dataN);
                            w = sdca.getRawWeight(0);
                        }
                        while (w.nnz() > 1);

                        //did we go too far?
                        while (w.nnz() == 0)
                        {
                            sdca.setLambda(sdca.getLambda()/ (1+search_const/3));
                            sdca.train(dataN);
                            w = sdca.getRawWeight(0);
                        }
                        search_const *= 0.95;
                    }
                    
                    assertEquals(1, w.nnz());
                    int nonZeroIndex = w.getNonZeroIterator().next().getIndex();
                    assertTrue(nonZeroIndex == 0);//should be one of the more important weights
                    assertEquals(1, (int)Math.signum(w.get(nonZeroIndex)));
                    
                    //elastic case
                    sdca.setLambda(maxLam / 10);
                    sdca.setAlpha(0.5);//now we should get the top 3 on
                    do
                    {
                        sdca.setLambda(sdca.getLambda() * 1.05);
                        sdca.train(data, sdca);
                        w = sdca.getRawWeight(0);
                    }
                    while (w.nnz() > 3);//we should be able to find this pretty easily
                    assertEquals(3, w.nnz());
                    assertEquals(1, (int) Math.signum(w.get(0)));
                    assertEquals(-1, (int) Math.signum(w.get(1)));
                    assertEquals(1, (int) Math.signum(w.get(2)));
                    //also want to make sure that they are all about equal in size
                    assertTrue(Math.abs((w.get(0) + w.get(1) * 2 + w.get(2)) / 3) < 0.4);

                    //Lets increase reg but switch to L2, we should see all features turn on!
                    sdca.setLambda(sdca.getLambda() * 3);
                    sdca.setAlpha(0.0);//now everyone should turn on

                    sdca.train(data);
                    w = sdca.getRawWeight(0);
                    if ((int) Math.signum(w.get(3)) != 1 && attempts > 0)//model probablly still right, but got a bad epsilon solution... try again please!
                    {
                        continue;
                    }
                    assertEquals(6, w.nnz());
                    assertEquals(1, (int) Math.signum(w.get(0)));
                    assertEquals(-1, (int) Math.signum(w.get(1)));
                    assertEquals(1, (int) Math.signum(w.get(2)));
                    assertEquals(1, (int) Math.signum(w.get(3)));
                    assertEquals(-1, (int) Math.signum(w.get(4)));
                    assertEquals(1, (int) Math.signum(w.get(5)));
                }
                break;//made it throgh the test no problemo!

            }
        }
    }
    
    @Test
    public void testWarmOther()
    {
        System.out.println("testWarm");
        Random rand  = RandomUtil.getRandom();
        ClassificationDataSet train = new ClassificationDataSet(600, new CategoricalData[0], new CategoricalData(2));
        
        for(int i = 0; i < 200; i++)
        {
            double Z1 = rand.nextDouble()*20-10;
            double Z2 = rand.nextDouble()*20-10;
            
            Vec v = new DenseVector(train.getNumNumericalVars());
            for(int j = 0; j < v.length(); j++)
            {
                if (j > 500)
                {
                    if (j % 2 == 0)
                        v.set(j, Z2 * ((j + 1) / 600.0) + rand.nextGaussian() / (j + 1));
                    else
                        v.set(j, Z1 * ((j + 1) / 600.0) + rand.nextGaussian() / (j + 1));
                }
                else
                    v.set(j, rand.nextGaussian()*20);
            }
            
            train.addDataPoint(v, (int) (Math.signum(Z1+0.1*Z2)+1)/2);
        }
        
        train.applyTransform(new LinearTransform(train));
        
        SDCA truth = new SDCA();
        truth.setMaxIters(1000);
        truth.setAlpha(0.5);
        truth.setLoss(new LogisticLoss());
        truth.setTolerance(1e-10);
        truth.setLambda(1.0/train.size());
        truth.train(train);
        
        SDCA warm = new SDCA();
        warm.setMaxIters(100);
        warm.setLoss(new LogisticLoss());
        warm.setAlpha(0.5);
        warm.setTolerance(1e-7);
        warm.setLambda(1.0/train.size());
        warm.train(train, truth);
        
        assertEquals(0, warm.getRawWeight(0).subtract(truth.getRawWeight(0)).pNorm(2), 1e-4);
        assertTrue(warm.epochs_taken + " ?< " + truth.epochs_taken, warm.epochs_taken < truth.epochs_taken);
    }
}
