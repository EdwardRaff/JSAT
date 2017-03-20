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

package jsat.classifiers.knn;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.bayesian.NaiveBayesUpdateable;
import jsat.classifiers.svm.DCDs;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.regression.*;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;
import org.junit.*;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class LWLTest
{

    public LWLTest()
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

        LWL instance = new LWL((Regressor)new DCDs(), 30, new EuclideanDistance());

        RegressionDataSet train = FixedProblems.getLinearRegression(5000, RandomUtil.getRandom());
        RegressionDataSet test = FixedProblems.getLinearRegression(200, RandomUtil.getRandom());

        RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train);
        rme.evaluateTestSet(test);

        assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.3);

    }

    @Test
    public void testTrainC_RegressionDataSet_ExecutorService()
    {
        System.out.println("train");

        LWL instance = new LWL((Regressor)new DCDs(), 15, new EuclideanDistance());

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        RegressionDataSet train = FixedProblems.getLinearRegression(5000, RandomUtil.getRandom());
        RegressionDataSet test = FixedProblems.getLinearRegression(200, RandomUtil.getRandom());

        RegressionModelEvaluation rme = new RegressionModelEvaluation(instance, train, ex);
        rme.evaluateTestSet(test);

        assertTrue(rme.getMeanError() <= test.getTargetValues().mean() * 0.3);

        ex.shutdownNow();
    }

    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");

        LWL instance = new LWL(new NaiveBayesUpdateable(), 15, new EuclideanDistance());

        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);

        ClassificationDataSet train = FixedProblems.getCircles(5000, 1.0, 10.0, 100.0);
        ClassificationDataSet test = FixedProblems.getCircles(200, 1.0, 10.0, 100.0);

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train, ex);
        cme.evaluateTestSet(test);

        assertTrue(cme.getErrorRate() <= 0.001);

        ex.shutdownNow();
    }

    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");

        LWL instance = new LWL(new NaiveBayesUpdateable(), 15, new EuclideanDistance());

        ClassificationDataSet train = FixedProblems.getCircles(5000, 1.0, 10.0, 100.0);
        ClassificationDataSet test = FixedProblems.getCircles(200, 1.0, 10.0, 100.0);

        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(instance, train);
        cme.evaluateTestSet(test);

        assertTrue(cme.getErrorRate() <= 0.001);

    }

    @Test
    public void testClone()
    {
        System.out.println("clone");

        LWL instance = new LWL(new NaiveBayesUpdateable(), 15, new EuclideanDistance());

        ClassificationDataSet t1 = FixedProblems.getSimpleKClassLinear(100, 3);
        ClassificationDataSet t2 = FixedProblems.getSimpleKClassLinear(100, 6);

        instance = instance.clone();

        instance.trainC(t1);

        LWL result = instance.clone();
        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), result.classify(t1.getDataPoint(i)).mostLikely());
        result.trainC(t2);

        for (int i = 0; i < t1.getSampleSize(); i++)
            assertEquals(t1.getDataPointCategory(i), instance.classify(t1.getDataPoint(i)).mostLikely());

        for (int i = 0; i < t2.getSampleSize(); i++)
            assertEquals(t2.getDataPointCategory(i), result.classify(t2.getDataPoint(i)).mostLikely());

    }

}
