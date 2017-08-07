package jsat.classifiers.linear;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.SimpleWeightVectorModel;
import jsat.classifiers.*;
import jsat.linear.*;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.LogisticLoss;
import jsat.lossfunctions.LossC;
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
    
    static ExecutorService ex;
    
    public SDCATest()
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
    
    /**
     * Test of trainC method, of class LogisticRegressionDCD.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        for(double alpha : new double[]{0.0, 0.5})
            for(LossC loss : new LossC[]{new LogisticLoss(), new HingeLoss()})
            {
                ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

                SDCA sdca = new SDCA();
                sdca.setLoss(loss);
                sdca.setLambda(1.0/train.getSampleSize());
                sdca.setAlpha(alpha);
                sdca.trainC(train, ex);

                ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

                for(DataPointPair<Integer> dpp : test.getAsDPPList())
                    assertEquals(dpp.getPair().longValue(), sdca.classify(dpp.getDataPoint()).mostLikely());
            }
    }

    /**
     * Test of trainC method, of class LogisticRegressionDCD.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        for(double alpha : new double[]{0.0, 0.5})
            for(LossC loss : new LossC[]{new LogisticLoss(), new HingeLoss()})
            {
                ClassificationDataSet train = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

                SDCA sdca = new SDCA();
                sdca.setLoss(loss);
                sdca.setLambda(1.0/train.getSampleSize());
                sdca.setAlpha(alpha);
                sdca.trainC(train);

                ClassificationDataSet test = FixedProblems.get2ClassLinear(200, RandomUtil.getRandom());

                for(DataPointPair<Integer> dpp : test.getAsDPPList())
                    assertEquals(dpp.getPair().longValue(), sdca.classify(dpp.getDataPoint()).mostLikely());
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

                for (int i = 0; i < 500; i++)
                {
                    double Z1 = rand.nextDouble() * 20 - 10;
                    double Z2 = rand.nextDouble() * 20 - 10;
                    Vec v = DenseVector.toDenseVec(Z1, -Z1, Z1, Z2, -Z2, Z2);
                    data.addDataPoint(v, (int) (Math.signum(Z1 + 0.1 * Z2) + 1) / 2);
                }

                for (LossC loss : new LossC[]{new LogisticLoss(), new HingeLoss()})
                {
                    Vec w = new ConstantVector(1.0, 6);
                    SDCA sdca = new SDCA();
                    sdca.setLoss(loss);

                    double maxLam = LinearTools.maxLambdaLogisticL1(data);

                    sdca.setMaxIters(1000);
                    sdca.setUseBias(false);
                    sdca.setAlpha(1.0);
                    //SDCA dosn't do fully L1 well, so skip to elastic and L2 tests

                    sdca.setLambda(maxLam / 1000);
                    sdca.setAlpha(0.5);//now we should get the top 3 on
                    do
                    {
                        sdca.setLambda(sdca.getLambda() * 1.05);
                        sdca.trainC(data);
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

                    sdca.trainC(data);
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
    
    private static class DumbWeightHolder implements Classifier, SimpleWeightVectorModel
    {
        public Vec w;
        public double b;

        @Override
        public CategoricalResults classify(DataPoint data)
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public void trainC(ClassificationDataSet dataSet)
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public boolean supportsWeightedData()
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public Classifier clone()
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public Vec getRawWeight(int index)
        {
            return w;
        }

        @Override
        public double getBias(int index)
        {
            return b;
        }

        @Override
        public int numWeightsVecs()
        {
            return 1;
        }
        
    }
    
    
}
