/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.trees;

import java.util.Arrays;


import java.util.Random;
import java.util.concurrent.Executors;

import jsat.utils.GridDataGenerator;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;

import java.util.concurrent.ExecutorService;

import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPointPair;
import jsat.datatransform.InsertMissingValuesTransform;
import jsat.datatransform.NumericalToHistogram;
import jsat.distributions.Uniform;
import jsat.regression.RegressionDataSet;

import org.junit.*;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class DecisionStumpTest
{
    static private ClassificationDataSet easyNumAtTrain;
    static private ClassificationDataSet easyNumAtTest;
    static private ClassificationDataSet easyCatAtTrain;
    static private ClassificationDataSet easyCatAtTest;
    static private RegressionDataSet easyNumAtTrain_R;
    static private RegressionDataSet easyNumAtTest_R;
    static private RegressionDataSet easyCatAtTrain_R;
    static private RegressionDataSet easyCatAtTest_R;
    static private ExecutorService ex;
    static private DecisionStump stump;
    
    public DecisionStumpTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
    }
    
    @Before
    public void setUp()
    {
        stump = new DecisionStump();
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new Random(12), 2);
        easyNumAtTrain = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        easyNumAtTest = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        
        easyCatAtTrain = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        easyCatAtTest = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        NumericalToHistogram nth = new NumericalToHistogram(easyCatAtTrain, 2);
        easyCatAtTrain.applyTransform(nth);
        easyCatAtTest.applyTransform(nth);
        
        
        easyNumAtTrain_R = new RegressionDataSet(easyNumAtTrain.getAsFloatDPPList());
        easyNumAtTest_R = new RegressionDataSet(easyNumAtTest.getAsFloatDPPList());
        easyCatAtTrain_R = new RegressionDataSet(easyCatAtTrain.getAsFloatDPPList());
        easyCatAtTest_R = new RegressionDataSet(easyCatAtTest.getAsFloatDPPList());
        
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }

    @After
    public void tearDown() throws Exception
    {
    }

    /**
     * Test of trainC method, of class DecisionStump.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC(ClassificationDataSet, ExecutorService)");
        stump.trainC(easyNumAtTrain, ex);
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), stump.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
    }
    
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService_missing()
    {
        System.out.println("trainC(ClassificationDataSet, ExecutorService)");
        ClassificationDataSet toTrain = easyNumAtTrain.shallowClone();
        toTrain.applyTransform(new InsertMissingValuesTransform(0.25));
        stump.trainC(toTrain, ex);
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), stump.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
        
        //test applying missing values, just make sure no error since we can/t pred if only feat is missing
        easyNumAtTest.applyTransform(new InsertMissingValuesTransform(0.5));
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            stump.classify(easyNumAtTest.getDataPoint(i));
    }
    
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService_missing_cat()
    {
        System.out.println("trainC(ClassificationDataSet, ExecutorService)");
        ClassificationDataSet toTrain = easyCatAtTrain.shallowClone();
        toTrain.applyTransform(new InsertMissingValuesTransform(0.25));
        stump.trainC(toTrain, ex);
        for(int i = 0; i < easyCatAtTest.getSampleSize(); i++)
            assertEquals(easyCatAtTest.getDataPointCategory(i), stump.classify(easyCatAtTest.getDataPoint(i)).mostLikely());
        
        
        //test applying missing values, just make sure no error since we can/t pred if only feat is missing
        easyCatAtTest.applyTransform(new InsertMissingValuesTransform(0.5));
        for(int i = 0; i < easyCatAtTest.getSampleSize(); i++)
            stump.classify(easyCatAtTest.getDataPoint(i));
    }
    
    /**
     * Test of trainC method, of class DecisionStump.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC(ClassificationDataSet)");
        stump.trainC(easyNumAtTrain);
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), stump.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
    }
    
    @Test
    public void testTrainC_RegressionDataSet_ExecutorService_missing()
    {
        System.out.println("trainR(RegressionDataSet, ExecutorService)");
        RegressionDataSet toTrain = easyNumAtTrain_R.shallowClone();
        toTrain.applyTransform(new InsertMissingValuesTransform(0.25));
        stump.train(toTrain, ex);
        for(int i = 0; i < easyNumAtTest_R.getSampleSize(); i++)
            assertEquals(easyNumAtTest_R.getTargetValue(i), stump.regress(easyNumAtTest_R.getDataPoint(i)), 0.2);
        
        //test applying missing values, just make sure no error since we can/t pred if only feat is missing
        easyNumAtTest_R.applyTransform(new InsertMissingValuesTransform(0.5));
        for(int i = 0; i < easyNumAtTest_R.getSampleSize(); i++)
            stump.regress(easyNumAtTest_R.getDataPoint(i));
    }
    
    @Test
    public void testTrainC_RegressionDataSet_ExecutorService_missing_cat()
    {
        System.out.println("trainR(RegressionDataSet, ExecutorService)");
        RegressionDataSet toTrain = easyCatAtTrain_R.shallowClone();
        toTrain.applyTransform(new InsertMissingValuesTransform(0.25));
        stump.train(toTrain, ex);
        for(int i = 0; i < easyCatAtTest_R.getSampleSize(); i++)
            assertEquals(easyCatAtTest_R.getTargetValue(i), stump.regress(easyCatAtTest_R.getDataPoint(i)), 0.2);
        
        //test applying missing values, just make sure no error since we can/t pred if only feat is missing
        easyCatAtTest_R.applyTransform(new InsertMissingValuesTransform(0.5));
        for(int i = 0; i < easyCatAtTest_R.getSampleSize(); i++)
            stump.regress(easyCatAtTest_R.getDataPoint(i));
    }

    /**
     * Test of trainC method, of class DecisionStump.
     */
    @Test
    public void testTrainC_List_Set()
    {
        System.out.println("trainC(List<DataPointPair>, Set<integer>)");
        stump.setPredicting(easyNumAtTrain.getPredicting());
        stump.trainC(easyNumAtTrain.getAsDPPList(), new IntSet(Arrays.asList(0)));
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), stump.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
    }

    /**
     * Test of supportsWeightedData method, of class DecisionStump.
     */
    @Test
    public void testSupportsWeightedData()
    {
        System.out.println("supportsWeightedData");
        assertTrue(stump.supportsWeightedData());
    }

    /**
     * Test of clone method, of class DecisionStump.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        Classifier clone = stump.clone();
        clone.trainC(easyNumAtTrain);
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), clone.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
        try
        {
            stump.classify(easyNumAtTest.getDataPoint(0));
            fail("Stump should not have been trained");
        }
        catch(Exception ex )
        {
            
        }
        clone = null;
        stump.trainC(easyNumAtTrain);
        clone = stump.clone();
        for(int i = 0; i < easyNumAtTest.getSampleSize(); i++)
            assertEquals(easyNumAtTest.getDataPointCategory(i), clone.classify(easyNumAtTest.getDataPoint(i)).mostLikely());
    }

   
    @Test
    public void testInfoGainSplit()
    {
        System.out.println("testInfoGainSplit");
        
        DecisionStump instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.INFORMATION_GAIN);
        
        instance.trainC(easyCatAtTrain);
        for(DataPointPair<Integer> dpp : easyCatAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
        
        instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.INFORMATION_GAIN);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testInfoGainRatioSplit()
    {
        System.out.println("testInfoGainRatioSplit");
        
        DecisionStump instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.INFORMATION_GAIN_RATIO);
        
        instance.trainC(easyCatAtTrain);
        for(DataPointPair<Integer> dpp : easyCatAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
        
        instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.INFORMATION_GAIN_RATIO);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testGiniSplit()
    {
        System.out.println("testGiniSplit");
        
        DecisionStump instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.GINI);
        
        instance.trainC(easyCatAtTrain);
        for(DataPointPair<Integer> dpp : easyCatAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
        
        instance = new DecisionStump();
        instance.setGainMethod(ImpurityScore.ImpurityMeasure.GINI);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }

    @Test
    public void testNumericCBinary()
    {
        System.out.println("testNumericCBinary");
        
        DecisionStump instance = new DecisionStump();
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }
   
}
