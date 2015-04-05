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

import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.parameters.Parameter;
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
public class EmphasisBoostTest
{
    
    public EmphasisBoostTest()
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
     * Test of getMaxIterations method, of class EmphasisBoost.
     */
    @Test
    public void testGetMaxIterations()
    {
        System.out.println("getMaxIterations");
        EmphasisBoost instance = null;
        int expResult = 0;
        int result = instance.getMaxIterations();
        assertEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of setMaxIterations method, of class EmphasisBoost.
     */
    @Test
    public void testSetMaxIterations()
    {
        System.out.println("setMaxIterations");
        int maxIterations = 0;
        EmphasisBoost instance = null;
        instance.setMaxIterations(maxIterations);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of getWeakLearner method, of class EmphasisBoost.
     */
    @Test
    public void testGetWeakLearner()
    {
        System.out.println("getWeakLearner");
        EmphasisBoost instance = null;
        Classifier expResult = null;
        Classifier result = instance.getWeakLearner();
        assertEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of setWeakLearner method, of class EmphasisBoost.
     */
    @Test
    public void testSetWeakLearner()
    {
        System.out.println("setWeakLearner");
        Classifier weakLearner = null;
        EmphasisBoost instance = null;
        instance.setWeakLearner(weakLearner);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of setLambda method, of class EmphasisBoost.
     */
    @Test
    public void testSetLambda()
    {
        System.out.println("setLambda");
        double lambda = 0.0;
        EmphasisBoost instance = null;
        instance.setLambda(lambda);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of getLambda method, of class EmphasisBoost.
     */
    @Test
    public void testGetLambda()
    {
        System.out.println("getLambda");
        EmphasisBoost instance = null;
        double expResult = 0.0;
        double result = instance.getLambda();
        assertEquals(expResult, result, 0.0);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of getScore method, of class EmphasisBoost.
     */
    @Test
    public void testGetScore()
    {
        System.out.println("getScore");
        DataPoint dp = null;
        EmphasisBoost instance = null;
        double expResult = 0.0;
        double result = instance.getScore(dp);
        assertEquals(expResult, result, 0.0);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of classify method, of class EmphasisBoost.
     */
    @Test
    public void testClassify()
    {
        System.out.println("classify");
        DataPoint data = null;
        EmphasisBoost instance = null;
        CategoricalResults expResult = null;
        CategoricalResults result = instance.classify(data);
        assertEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of trainC method, of class EmphasisBoost.
     */
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        ClassificationDataSet dataSet = null;
        ExecutorService threadPool = null;
        EmphasisBoost instance = null;
        instance.trainC(dataSet, threadPool);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of trainC method, of class EmphasisBoost.
     */
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet dataSet = null;
        EmphasisBoost instance = null;
        instance.trainC(dataSet);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of supportsWeightedData method, of class EmphasisBoost.
     */
    @Test
    public void testSupportsWeightedData()
    {
        System.out.println("supportsWeightedData");
        EmphasisBoost instance = null;
        boolean expResult = false;
        boolean result = instance.supportsWeightedData();
        assertEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of clone method, of class EmphasisBoost.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        EmphasisBoost instance = null;
        EmphasisBoost expResult = null;
        EmphasisBoost result = instance.clone();
        assertEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of getParameters method, of class EmphasisBoost.
     */
    @Test
    public void testGetParameters()
    {
        System.out.println("getParameters");
        EmphasisBoost instance = null;
        List<Parameter> expResult = null;
        List<Parameter> result = instance.getParameters();
        assertEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of getParameter method, of class EmphasisBoost.
     */
    @Test
    public void testGetParameter()
    {
        System.out.println("getParameter");
        String paramName = "";
        EmphasisBoost instance = null;
        Parameter expResult = null;
        Parameter result = instance.getParameter(paramName);
        assertEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }
    
}
