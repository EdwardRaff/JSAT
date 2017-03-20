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
package jsat.classifiers.bayesian;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.datatransform.DataTransformProcess;
import jsat.datatransform.NumericalToHistogram;
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
public class AODETest
{
    
    public AODETest()
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
    public void testSubEpochs()
    {
        System.out.println("getSubEpochs");
        AODE instance = new AODE();
        
        instance.setM(0.1);
        assertEquals(0.1, instance.getM(), 0.0);
        for (int i = -3; i < 0; i++)
            try
            {
                instance.setM(i);
                fail("Invalid value should have thrown an error");
            }
            catch (Exception ex)
            {

            }
    }
    
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        AODE instance = new AODE();
        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(10000, 3, RandomUtil.getRandom());
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(1000, 3, RandomUtil.getRandom());

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
        cme.setDataTransformProcess(new DataTransformProcess(new NumericalToHistogram()));
        cme.evaluateTestSet(test);
        
        assertTrue(cme.getErrorRate() <= 0.001);
        
        ex.shutdownNow();
    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        AODE instance = new AODE();
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(10000, 3, RandomUtil.getRandom());
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(1000, 3, RandomUtil.getRandom());

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
        cme.setDataTransformProcess(new DataTransformProcess(new NumericalToHistogram()));
        cme.evaluateTestSet(test);
        
        assertTrue(cme.getErrorRate() <= 0.001);
    }

    
    @Test
    public void testClone()
    {
        System.out.println("clone");
        
        ClassificationDataSet t1 = FixedProblems.getSimpleKClassLinear(10000, 3, RandomUtil.getRandom());
        ClassificationDataSet t2 = FixedProblems.getSimpleKClassLinear(10000, 6, RandomUtil.getRandom());
        t1.applyTransform(new NumericalToHistogram(t1));
        t2.applyTransform(new NumericalToHistogram(t2));
        
        AODE instance = new AODE();
        
        instance = instance.clone();
                
        instance.trainC(t1);

        AODE result = instance.clone();
        result.trainC(t2);
        
        for(int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());
        
        for(int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());
    }
    
}
