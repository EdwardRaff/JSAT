/*
 * Copyright (C) 2015 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.classifiers.trees;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.datatransform.DataTransformProcess;
import jsat.datatransform.NumericalToHistogram;
import jsat.linear.DenseVector;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.utils.SystemInfo;
import jsat.utils.random.XORWOW;
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
public class RandomForestTest
{
 
    static DenseVector coefs = new DenseVector(new double[]{0.1, 0.9, -0.2, 0.4, -0.5});
    public RandomForestTest()
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
    public void testTrainC_RegressionDataSet()
    {
        System.out.println("train");
        for(boolean useCatFeatures : new boolean[]{true, false})
        {
            RandomForest instance = new RandomForest();

            RegressionDataSet train =  FixedProblems.getLinearRegression(1000, new XORWOW(), coefs);
            RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW(), coefs);

            RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train);
            if(useCatFeatures) {
              rme.setDataTransformProcess(new DataTransformProcess(new NumericalToHistogram.NumericalToHistogramTransformFactory()));
            }
            rme.evaluateTestSet(test);
            
            assertTrue(rme.getMeanError() <= test.getTargetValues().mean()*2.5);
        }
    }
    
    @Test
    public void testTrainC_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");

        for(boolean useCatFeatures : new boolean[]{true, false})
        {
            RandomForest instance = new RandomForest();
            
            ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

            RegressionDataSet train =  FixedProblems.getLinearRegression(1000, new XORWOW(), coefs);
            RegressionDataSet test = FixedProblems.getLinearRegression(100, new XORWOW(), coefs);

            RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train, ex);
            if(useCatFeatures) {
              rme.setDataTransformProcess(new DataTransformProcess(new NumericalToHistogram.NumericalToHistogramTransformFactory()));
            }
            rme.evaluateTestSet(test);

            assertTrue(rme.getMeanError() <= test.getTargetValues().mean()*2.5);

            ex.shutdownNow();
        }
    }
    
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");

        for(boolean useCatFeatures : new boolean[]{true, false})
        {
            RandomForest instance = new RandomForest();

            ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

            ClassificationDataSet train = FixedProblems.getCircles(1000, 1.0, 10.0, 100.0);
            //RF may not get boundry perfect, so use noiseless for testing
            ClassificationDataSet test = FixedProblems.getCircles(100, 0.0, new XORWOW(), 1.0, 10.0, 100.0);

            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
            if(useCatFeatures) {
              cme.setDataTransformProcess(new DataTransformProcess(new NumericalToHistogram.NumericalToHistogramTransformFactory()));
            }
            cme.evaluateTestSet(test);

            assertTrue(cme.getErrorRate() <= 0.001);

            ex.shutdownNow();
        }
    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        for(boolean useCatFeatures : new boolean[]{true, false})
        {
            RandomForest instance = new RandomForest();

            ClassificationDataSet train =  FixedProblems.getCircles(1000, 1.0, 10.0, 100.0);
            //RF may not get boundry perfect, so use noiseless for testing
            ClassificationDataSet test = FixedProblems.getCircles(100, 0.0, new XORWOW(), 1.0, 10.0, 100.0);

            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
            if(useCatFeatures) {
              cme.setDataTransformProcess(new DataTransformProcess(new NumericalToHistogram.NumericalToHistogramTransformFactory()));
            }
            cme.evaluateTestSet(test);

            assertTrue(cme.getErrorRate() <= 0.001);
        }
    }

    
    @Test
    public void testClone()
    {
        System.out.println("clone");
        for(boolean useCatFeatures : new boolean[]{true, false})
        {
            RandomForest instance = new RandomForest();
            
            ClassificationDataSet t1 = FixedProblems.getSimpleKClassLinear(100, 3);
            ClassificationDataSet t2 = FixedProblems.getSimpleKClassLinear(100, 6);
            if(useCatFeatures)
            {
                t1.applyTransform(new NumericalToHistogram(t1));
                t2.applyTransform(new NumericalToHistogram(t2));
            }

            instance = instance.clone();

            instance.trainC(t1);

            RandomForest result = instance.clone();
            for(int i = 0; i < t1.getSampleSize(); i++) {
              assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
            }
            result.trainC(t2);

            for(int i = 0; i < t1.getSampleSize(); i++) {
              assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());
            }

            for(int i = 0; i < t2.getSampleSize(); i++) {
              assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());
            }
        }
    }
    
}
