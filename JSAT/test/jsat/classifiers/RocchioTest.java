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
package jsat.classifiers;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.utils.SystemInfo;
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
public class RocchioTest
{
    
    public RocchioTest()
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
        Rocchio instance = new Rocchio();
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(10000, 3);
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(1000, 3);

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        
        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
        cme.evaluateTestSet(test);
        
        assertTrue(cme.getErrorRate() <= 0.001);
    }
    
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        Rocchio instance = new Rocchio();
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(10000, 3);
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(1000, 3);

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
        cme.evaluateTestSet(test);
        
        assertTrue(cme.getErrorRate() <= 0.001);
    }

    @Test
    public void testClone()
    {
        System.out.println("clone");
        
        ClassificationDataSet t1 = FixedProblems.getSimpleKClassLinear(10000, 3);
        ClassificationDataSet t2 = FixedProblems.getSimpleKClassLinear(10000, 6);
        
        Rocchio instance = new Rocchio();
        
        instance = instance.clone();
                
        instance.trainC(t1);

        Rocchio result = instance.clone();
        for(int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
        result.trainC(t2);
        
        for(int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());
        
        for(int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());
    }
    
}
