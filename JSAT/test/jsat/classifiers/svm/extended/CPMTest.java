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
package jsat.classifiers.svm.extended;

import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
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
public class CPMTest
{
    
    public CPMTest()
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
     * Test of trainC method, of class AMM.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        //Hard to come up witha  good test problem for AMM, since it works better on higher dim problems
        System.out.println("trainC");
        CPM instance = new CPM(2, 20, 1.5, 50);
        
        ClassificationDataSet train = FixedProblems.getSimpleKClassLinear(10000, 2, RandomUtil.getRandom());
        ClassificationDataSet test = FixedProblems.getSimpleKClassLinear(1000, 2, RandomUtil.getRandom());

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
        cme.evaluateTestSet(test);
        
        assertTrue(cme.getErrorRate() <= 0.001);
    }

    /**
     * Test of clone method, of class AMM.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        
        ClassificationDataSet t1 = FixedProblems.getSimpleKClassLinear(10000, 2, RandomUtil.getRandom());
        ClassificationDataSet t2 = FixedProblems.getSimpleKClassLinear(10000, 2, RandomUtil.getRandom());
        
        CPM instance = new CPM(2, 20, 1.5, 50);
        
        instance = instance.clone();
                
        instance.trainC(t1);

        CPM result = instance.clone();
        result.trainC(t2);
        
        for(int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());
        
        for(int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());
    }
    
}
