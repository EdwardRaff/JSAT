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

package jsat.classifiers.boosting;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.*;
import jsat.classifiers.trees.DecisionStump;
import jsat.classifiers.trees.DecisionTree;
import jsat.classifiers.trees.TreePruner;
import jsat.datatransform.LinearTransform;
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
public class AdaBoostM1Test
{
    
    public AdaBoostM1Test()
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

        AdaBoostM1 instance = new AdaBoostM1(new DecisionStump(), 50);

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        ClassificationDataSet train = FixedProblems.getCircles(1000, .1, 10.0);
        ClassificationDataSet test = FixedProblems.getCircles(100, .1, 10.0);

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
        cme.evaluateTestSet(test);

        assertTrue(cme.getErrorRate() <= 0.15);

        ex.shutdownNow();
    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");

        AdaBoostM1 instance = new AdaBoostM1(new DecisionStump(), 50);

        ClassificationDataSet train = FixedProblems.getCircles(1000, .1, 10.0);
        ClassificationDataSet test = FixedProblems.getCircles(100, .1, 10.0);

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
        cme.evaluateTestSet(test);

        assertTrue(cme.getErrorRate() <= 0.15);

    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        AdaBoostM1 instance = new AdaBoostM1(new DecisionTree(10, 10, TreePruner.PruningMethod.NONE, 0.1), 50);

        ClassificationDataSet t1 = FixedProblems.getCircles(1000, 0.1, 10.0);
        ClassificationDataSet t2 = FixedProblems.getCircles(1000, 0.1, 10.0);
        
        t2.applyTransform(new LinearTransform(t2));

        int errors;
        
        instance = instance.clone();

        instance.trainC(t1);

        AdaBoostM1 result = instance.clone();
        
        errors = 0;
        for (int i = 0; i < t1.getSampleSize(); i++)
            errors += Math.abs(t1.getDataPointCategory(i) -  result.classify(t1.getDataPoint(i)).mostLikely());
        assertTrue(errors < 100);
        result.trainC(t2);

        for (int i = 0; i < t1.getSampleSize(); i++)
            errors += Math.abs(t1.getDataPointCategory(i) -  instance.classify(t1.getDataPoint(i)).mostLikely());
        assertTrue(errors < 100);

        for (int i = 0; i < t2.getSampleSize(); i++)
            errors += Math.abs(t2.getDataPointCategory(i) -  result.classify(t2.getDataPoint(i)).mostLikely());
        assertTrue(errors < 100);
    }
}
