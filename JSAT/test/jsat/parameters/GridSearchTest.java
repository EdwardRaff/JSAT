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
package jsat.parameters;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.distributions.Distribution;
import jsat.distributions.Uniform;
import jsat.distributions.discrete.UniformDiscrete;
import jsat.linear.DenseVector;
import jsat.parameters.Parameter.WarmParameter;
import jsat.regression.RegressionDataSet;
import jsat.regression.Regressor;
import jsat.regression.WarmRegressor;
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
public class GridSearchTest
{
    static ExecutorService ex;
    ClassificationDataSet classData ;
    RegressionDataSet regData;
    
    public GridSearchTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
    }
    
    @AfterClass
    public static void tearDownClass()
    {
        ex.shutdown();
    }
    
    @Before
    public void setUp()
    {
        classData = new ClassificationDataSet(1, new CategoricalData[0], new CategoricalData(2));
        for(int i = 0; i < 100; i++)
            classData.addDataPoint(DenseVector.toDenseVec(1.0*i), 0);
        for(int i = 0; i < 100; i++)
            classData.addDataPoint(DenseVector.toDenseVec(-1.0*i), 1);
        
        regData = new RegressionDataSet(1, new CategoricalData[0]);
        for(int i = 0; i < 100; i++)
            regData.addDataPoint(DenseVector.toDenseVec(1.0*i), 0);
        for(int i = 0; i < 100; i++)
            regData.addDataPoint(DenseVector.toDenseVec(-1.0*i), 1);
        
    }
    
    @After
    public void tearDown()
    {
    }


    @Test
    public void testClassificationWarm()
    {
        System.out.println("setUseWarmStarts");
        GridSearch instance = new GridSearch((Classifier)new DumbModel(), 5);
        instance.setUseWarmStarts(true);
        
        instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
        instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);
        
        instance = instance.clone();
        instance.trainC(classData);
        instance = instance.clone();
        
        DumbModel model = (DumbModel) instance.getTrainedClassifier();
        assertEquals(1, model.param1);
        assertEquals(2, model.param2, 0.0);
        assertEquals(3, model.param3);
        assertTrue(model.wasWarmStarted);
    }
    
    @Test
    public void testClassificationWarmExec()
    {
        System.out.println("setUseWarmStarts");
        GridSearch instance = new GridSearch((Classifier)new DumbModel(), 5);
        instance.setUseWarmStarts(true);
        
        instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
        instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);
        
        instance = instance.clone();
        instance.trainC(classData, ex);
        instance = instance.clone();
        
        DumbModel model = (DumbModel) instance.getTrainedClassifier();
        assertEquals(1, model.param1);
        assertEquals(2, model.param2, 0.0);
        assertEquals(3, model.param3);
        assertTrue(model.wasWarmStarted);
    }
    
    @Test
    public void testClassification()
    {
        System.out.println("setUseWarmStarts");
        GridSearch instance = new GridSearch((Classifier)new DumbModel(), 5);
        instance.setUseWarmStarts(false);
        
        instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
        instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);
        
        instance = instance.clone();
        instance.trainC(classData);
        instance = instance.clone();
        
        DumbModel model = (DumbModel) instance.getTrainedClassifier();
        assertEquals(1, model.param1);
        assertEquals(2, model.param2, 0.0);
        assertEquals(3, model.param3);
        assertFalse(model.wasWarmStarted);
    }
    
    @Test
    public void testClassificationAutoAdd()
    {
        System.out.println("classificationAutoAdd");
        GridSearch instance = new GridSearch((Classifier)new DumbModel(), 5);
        instance.setUseWarmStarts(false);
        
        instance.autoAddParameters(classData);
        
        instance = instance.clone();
        instance.trainC(classData);
        instance = instance.clone();
        
        DumbModel model = (DumbModel) instance.getTrainedClassifier();
        assertEquals(1, model.param1);
        assertEquals(2, model.param2, 0.5);
        assertEquals(3, model.param3);
        assertFalse(model.wasWarmStarted);
    }
    
    @Test
    public void testClassificationExec()
    {
        System.out.println("setUseWarmStarts");
        GridSearch instance = new GridSearch((Classifier)new DumbModel(), 5);
        instance.setUseWarmStarts(false);
        
        instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
        instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);
        
        instance = instance.clone();
        instance.trainC(classData, ex);
        instance = instance.clone();
        
        DumbModel model = (DumbModel) instance.getTrainedClassifier();
        assertEquals(1, model.param1);
        assertEquals(2, model.param2, 0.0);
        assertEquals(3, model.param3);
        assertFalse(model.wasWarmStarted);
    }
    
    @Test
    public void testRegressionWarm()
    {
        System.out.println("setUseWarmStarts");
        GridSearch instance = new GridSearch((Regressor)new DumbModel(), 5);
        instance.setUseWarmStarts(true);
        
        instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
        instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);
        
        instance = instance.clone();
        instance.train(regData);
        instance = instance.clone();
        
        DumbModel model = (DumbModel) instance.getTrainedRegressor();
        assertEquals(1, model.param1);
        assertEquals(2, model.param2, 0.0);
        assertEquals(3, model.param3);
        assertTrue(model.wasWarmStarted);
    }
    
    @Test
    public void testRegressionWarmExec()
    {
        System.out.println("setUseWarmStarts");
        GridSearch instance = new GridSearch((Regressor)new DumbModel(), 5);
        instance.setUseWarmStarts(true);
        
        instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
        instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);
        
        instance = instance.clone();
        instance.train(regData, ex);
        instance = instance.clone();
        
        DumbModel model = (DumbModel) instance.getTrainedRegressor();
        assertEquals(1, model.param1);
        assertEquals(2, model.param2, 0.0);
        assertEquals(3, model.param3);
        assertTrue(model.wasWarmStarted);
    }
    
    @Test
    public void testRegression()
    {
        System.out.println("setUseWarmStarts");
        GridSearch instance = new GridSearch((Regressor)new DumbModel(), 5);
        instance.setUseWarmStarts(false);
        
        instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
        instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);
        
        instance = instance.clone();
        instance.train(regData);
        instance = instance.clone();
        
        DumbModel model = (DumbModel) instance.getTrainedRegressor();
        assertEquals(1, model.param1);
        assertEquals(2, model.param2, 0.0);
        assertEquals(3, model.param3);
        assertFalse(model.wasWarmStarted);
    }
    
    @Test
    public void testRegressionExec()
    {
        System.out.println("setUseWarmStarts");
        GridSearch instance = new GridSearch((Regressor)new DumbModel(), 5);
        instance.setUseWarmStarts(false);
        
        instance.addParameter("Param1", 0, 1, 2, 3, 4, 5);
        instance.addParameter("Param2", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        instance.addParameter("Param3", 0, 1, 2, 3, 4, 5);
        
        instance = instance.clone();
        instance.train(regData, ex);
        instance = instance.clone();
        
        DumbModel model = (DumbModel) instance.getTrainedRegressor();
        assertEquals(1, model.param1);
        assertEquals(2, model.param2, 0.0);
        assertEquals(3, model.param3);
        assertFalse(model.wasWarmStarted);
    }

    /**
     * This model is dumb. It always returns the same thing, unless the parameters are set to specific values. Then it returns a 2nd option as well. 
     */
    static class DumbModel implements WarmClassifier, WarmRegressor, Parameterized
    {
        int param1;
        double param2;
        int param3;
        boolean wasWarmStarted = false;

        public void setParam1(int param1)
        {
            this.param1 = param1;
        }

        public int getParam1()
        {
            return param1;
        }
        
        public Distribution guessParam1(DataSet d)
        {
            return new UniformDiscrete(0, 5);
        }

        public void setParam2(double param2)
        {
            this.param2 = param2;
        }

        public double getParam2()
        {
            return param2;
        }
        
        public Distribution guessParam2(DataSet d)
        {
            return new Uniform(0.0, 5.0);
        }

        @WarmParameter(prefLowToHigh = false)
        public void setParam3(int param3)
        {
            this.param3 = param3;
        }

        public int getParam3()
        {
            return param3;
        }
        
        public Distribution guessParam3(DataSet d)
        {
            return new UniformDiscrete(0, 5);
        }

        @Override
        public boolean warmFromSameDataOnly()
        {
            return false;
        }

        @Override
        public void trainC(ClassificationDataSet dataSet, Classifier warmSolution, ExecutorService threadPool)
        {
            wasWarmStarted = ((DumbModel)warmSolution).param3 == this.param3;
        }

        @Override
        public void trainC(ClassificationDataSet dataSet, Classifier warmSolution)
        {
            wasWarmStarted = ((DumbModel)warmSolution).param3 == this.param3;
        }

        @Override
        public CategoricalResults classify(DataPoint data)
        {
            //range check done so that RandomSearch can re-use this class for its own test
            boolean param2InRange = 1.5 < param2 && param2 < 2.5;
            if(param1 == 1 && param2InRange && param3 == 3)
                if(data.getNumericalValues().get(0) < 0)
                    return new CategoricalResults(new double[]{0.0, 1.0});
            return new CategoricalResults(new double[]{1.0, 0.0});
        }

        @Override
        public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
        {
            wasWarmStarted = false;
        }

        @Override
        public void trainC(ClassificationDataSet dataSet)
        {
            wasWarmStarted = false;
        }

        @Override
        public boolean supportsWeightedData()
        {
            return true;
        }

        @Override
        public void train(RegressionDataSet dataSet, Regressor warmSolution, ExecutorService threadPool)
        {
            wasWarmStarted = ((DumbModel)warmSolution).param3 == this.param3;
        }

        @Override
        public void train(RegressionDataSet dataSet, Regressor warmSolution)
        {
            wasWarmStarted = ((DumbModel)warmSolution).param3 == this.param3;
        }

        @Override
        public double regress(DataPoint data)
        {
            //range check done so that RandomSearch can re-use this class for its own test
            boolean param2InRange = 1.5 < param2 && param2 < 2.5;
            if(param1 == 1 && param2InRange && param3 == 3)
                if (data.getNumericalValues().get(0) < 0)
                    return 1;
            return 0;
        }

        @Override
        public void train(RegressionDataSet dataSet, ExecutorService threadPool)
        {
            wasWarmStarted = false;
        }

        @Override
        public void train(RegressionDataSet dataSet)
        {
            wasWarmStarted = false;
        }

        @Override
        public DumbModel clone()
        {
            DumbModel toRet = new DumbModel();
            toRet.param1 = this.param1;
            toRet.param2 = this.param2;
            toRet.param3 = this.param3;
            toRet.wasWarmStarted = this.wasWarmStarted;
            return toRet;
        }

        @Override
        public List<Parameter> getParameters()
        {
            return Parameter.getParamsFromMethods(this);
        }

        @Override
        public Parameter getParameter(String paramName)
        {
            return Parameter.toParameterMap(getParameters()).get(paramName);
        }
        
    }
}
