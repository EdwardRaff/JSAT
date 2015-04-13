/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
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

package jsat.classifiers.neuralnetwork;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.*;
import jsat.utils.SystemInfo;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class SOMTest
{
    
    public SOMTest()
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
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");

        SOM instance = new SOM(5, 5);
        instance.setMaxIterations(200);

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        ClassificationDataSet train = FixedProblems.getCircles(1000, 1.0, 10.0, 100.0);
        ClassificationDataSet test = FixedProblems.getCircles(100, 1.0, 10.0, 100.0);

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
        cme.evaluateTestSet(test);

        assertEquals(0, cme.getErrorRate(), 0.1);

        ex.shutdownNow();

    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");

        
        SOM instance = new SOM(5, 5);
        instance.setMaxIterations(50);
        
        ClassificationDataSet train = FixedProblems.getCircles(1000, 1.0, 10.0, 100.0);
        ClassificationDataSet test = FixedProblems.getCircles(100, 1.0, 10.0, 100.0);

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
        cme.evaluateTestSet(test);

        assertEquals(0, cme.getErrorRate(), 0.1);

    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        SOM instance = new SOM(5, 5);
        instance.setMaxIterations(50);
        
        ClassificationDataSet t1 = FixedProblems.getSimpleKClassLinear(500, 3);
        ClassificationDataSet t2 = FixedProblems.getSimpleKClassLinear(500, 6);

        instance = instance.clone();

        instance.trainC(t1);

        SOM result = instance.clone();
        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
        result.trainC(t2);

        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());

        for (int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());

    }
    
}
