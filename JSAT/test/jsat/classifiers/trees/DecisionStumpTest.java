/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.classifiers.trees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import jsat.distributions.Normal;
import java.util.Random;
import java.util.concurrent.Executors;
import jsat.utils.GridDataGenerator;
import jsat.utils.SystemInfo;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.trees.DecisionStump.GainMethod;
import jsat.datatransform.NumericalToHistogram;
import jsat.distributions.Distribution;
import jsat.distributions.Uniform;
import jsat.regression.RegressionDataSet;
import jsat.utils.PairedReturn;
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
        GridDataGenerator gdg = new GridDataGenerator(new Uniform(-0.15, 0.15), new Random(12), 1);
        easyNumAtTrain = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        easyNumAtTest = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        
        easyCatAtTrain = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        easyCatAtTest = new ClassificationDataSet(gdg.generateData(40).getBackingList(), 0);
        NumericalToHistogram nth = new NumericalToHistogram(easyCatAtTrain, 2);
        easyCatAtTrain.applyTransform(nth);
        easyCatAtTest.applyTransform(nth);
        
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }

    @After
    public void tearDown() throws Exception
    {
    }


    /**
     * Test of entropy method, of class DecisionStump.
     */
    @Test
    public void testEntropy()
    {
        System.out.println("entropy");
        List<DataPointPair<Integer>> dataPoints = new ArrayList<DataPointPair<Integer>>();
        for(int i = 0; i < 10; i++)
            dataPoints.add(new DataPointPair<Integer>(new DataPoint(null, null, null), 0));
        assertEquals(0.0, DecisionStump.entropy(dataPoints), 1e-14);
        
        for(int i = 0; i < 10; i++)
            dataPoints.add(new DataPointPair<Integer>(new DataPoint(null, null, null), 1));
        
        assertEquals(1.0, DecisionStump.entropy(dataPoints), 1e-14);
        
        for(int i = 0; i < 10; i++)
            dataPoints.add(new DataPointPair<Integer>(new DataPoint(null, null, null), 1));
        assertEquals(0.918295834, DecisionStump.entropy(dataPoints), 1e-6);
        
        for(int i = 0; i < 70; i++)
            dataPoints.add(new DataPointPair<Integer>(new DataPoint(null, null, null), 2));
        //1/10, 2/10, 7/10 split
        assertEquals(1.15677965, DecisionStump.entropy(dataPoints), 1e-6);
        
        for(int i = 0; i < 50; i++)
            dataPoints.add(new DataPointPair<Integer>(new DataPoint(null, null, null), 1));
        for(int i = 0; i < 60; i++)
            dataPoints.add(new DataPointPair<Integer>(new DataPoint(null, null, null), 0));
        //1/3 to each
        assertEquals(1.5849625, DecisionStump.entropy(dataPoints), 1e-6);
    }

    /**
     * Test of threshholdSplit method, of class DecisionStump.
     */
    @Test
    public void testThreshholdSplit()
    {
        System.out.println("threshholdSplit");
        Distribution dist1 = new Normal(0, 1);
        Distribution dist2 = new Normal(3, 2);
        PairedReturn<Integer, Double> ret = DecisionStump.threshholdSplit(dist1, dist2);
        assertEquals(0, (int) ret.getFirstItem());
        assertEquals(1.418344988105127, ret.getSecondItem(), 1e-6);
        
        ret = DecisionStump.threshholdSplit(dist2, dist1);
        assertEquals(1, (int) ret.getFirstItem());
        assertEquals(1.418344988105127, ret.getSecondItem(), 1e-6);
        
        dist2 = new Normal(3, 1);
        ret = DecisionStump.threshholdSplit(dist2, dist1);
        assertEquals(1, (int) ret.getFirstItem());
        assertEquals(1.5, ret.getSecondItem(), 1e-6);
        
        try
        {
            dist1 = new Normal(0, 2);
            dist2 = new Normal(3, 10);
            ret = DecisionStump.threshholdSplit(dist2, dist1);
            assertEquals(1, (int) ret.getFirstItem());
            assertEquals(4.896411863121647, ret.getSecondItem(), 1e-6);//This is the spliting point, but we dont expect it to get it 
        }
        catch(ArithmeticException ex)
        {
            
        }
        
        dist1 = new Normal(0, 1);
        dist2 = new Normal(100, 1);
        ret = DecisionStump.threshholdSplit(dist1, dist2);
        assertEquals(0, (int) ret.getFirstItem());
        assertEquals(50, ret.getSecondItem(), 1e-6);
        
        dist1 = new Normal(0, 1);
        dist2 = new Normal(100, 1);
        ret = DecisionStump.threshholdSplit(dist2, dist1);
        assertEquals(1, (int) ret.getFirstItem());
        assertEquals(50, ret.getSecondItem(), 1e-6);
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

    /**
     * Test of trainC method, of class DecisionStump.
     */
    @Test
    public void testTrainC_List_Set()
    {
        System.out.println("trainC(List<DataPointPair>, Set<integer>)");
        stump.setPredicting(easyNumAtTrain.getPredicting());
        stump.trainC(easyNumAtTrain.getAsDPPList(), new HashSet<Integer>(Arrays.asList(0)));
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
        instance.setGainMethod(GainMethod.INFORMATION_GAIN);
        
        instance.trainC(easyCatAtTrain);
        for(DataPointPair<Integer> dpp : easyCatAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
        
        instance = new DecisionStump();
        instance.setGainMethod(GainMethod.INFORMATION_GAIN);
        
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
        instance.setGainMethod(GainMethod.INFORMATION_GAIN_RATIO);
        
        instance.trainC(easyCatAtTrain);
        for(DataPointPair<Integer> dpp : easyCatAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
        
        instance = new DecisionStump();
        instance.setGainMethod(GainMethod.INFORMATION_GAIN_RATIO);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testGiniSplit()
    {
        System.out.println("testInfoGainRatioSplit");
        
        DecisionStump instance = new DecisionStump();
        instance.setGainMethod(GainMethod.GINI);
        
        instance.trainC(easyCatAtTrain);
        for(DataPointPair<Integer> dpp : easyCatAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
        
        instance = new DecisionStump();
        instance.setGainMethod(GainMethod.GINI);
        
        instance.trainC(easyNumAtTrain);
        for(DataPointPair<Integer> dpp : easyNumAtTest.getAsDPPList())
            assertEquals(dpp.getPair().longValue(),
                    instance.classify(dpp.getDataPoint()).mostLikely());
    }

   
}
