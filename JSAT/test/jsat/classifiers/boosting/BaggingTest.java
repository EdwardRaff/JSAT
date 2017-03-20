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
import jsat.classifiers.trees.*;
import jsat.datatransform.LinearTransform;
import jsat.regression.RegressionDataSet;
import jsat.regression.RegressionModelEvaluation;
import jsat.regression.Regressor;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;
import org.junit.*;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class BaggingTest
{
    
    public BaggingTest()
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

        Bagging instance = new Bagging((Regressor)new DecisionTree());

        RegressionDataSet train = FixedProblems.getLinearRegression(5000, RandomUtil.getRandom());
        RegressionDataSet test = FixedProblems.getLinearRegression(100, RandomUtil.getRandom());

        RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train);
        rme.evaluateTestSet(test);

        assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.5);

    }

    @Test
    public void testTrainC_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");

        Bagging instance = new Bagging((Regressor)new DecisionTree());

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        RegressionDataSet train = FixedProblems.getLinearRegression(5000, RandomUtil.getRandom());
        RegressionDataSet test = FixedProblems.getLinearRegression(100, RandomUtil.getRandom());

        RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train, ex);
        rme.evaluateTestSet(test);

        assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.5);

        ex.shutdownNow();
    }

    
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");

        Bagging instance = new Bagging((Classifier)new DecisionTree());

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        ClassificationDataSet train = FixedProblems.getCircles(5000, .1, 10.0);
        ClassificationDataSet test = FixedProblems.getCircles(100, .1, 10.0);

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
        cme.evaluateTestSet(test);


        assertTrue(cme.getErrorRate() <= 0.05);

        ex.shutdownNow();
    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");

        Bagging instance = new Bagging((Classifier)new DecisionTree());

        ClassificationDataSet train = FixedProblems.getCircles(5000, .1, 10.0);
        ClassificationDataSet test = FixedProblems.getCircles(100, .1, 10.0);

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
        cme.evaluateTestSet(test);

        assertTrue(cme.getErrorRate() <= 0.05);

    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        Bagging instance = new Bagging((Classifier)new DecisionTree());

        ClassificationDataSet t1 = FixedProblems.getCircles(1000, 0.1, 10.0);
        ClassificationDataSet t2 = FixedProblems.getCircles(1000, 0.1, 10.0);
        
        t2.applyTransform(new LinearTransform(t2));

        int errors;
        
        instance = instance.clone();

        instance.trainC(t1);

        Bagging result = instance.clone();
        
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
