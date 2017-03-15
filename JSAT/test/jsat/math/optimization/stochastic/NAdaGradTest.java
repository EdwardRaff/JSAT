/*
 * Copyright (C) 2016 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.math.optimization.stochastic;

import java.util.Random;
import jsat.FixedProblems;
import jsat.SimpleDataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.linear.LinearSGD;
import jsat.datatransform.LinearTransform;
import jsat.distributions.TruncatedDistribution;
import jsat.distributions.Uniform;
import jsat.linear.DenseVector;
import jsat.linear.SubVector;
import jsat.linear.Vec;
import jsat.lossfunctions.LogisticLoss;
import jsat.math.FunctionVec;
import jsat.math.decayrates.NoDecay;
import jsat.math.optimization.RosenbrockFunction;
import jsat.utils.GridDataGenerator;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;
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
public class NAdaGradTest
{
    
    public NAdaGradTest()
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
    public void testUpdate_3args()
    {
        System.out.println("update");
        Random rand = RandomUtil.getRandom();
        Vec x0 = new DenseVector(10);
        for(int i = 0; i < x0.length(); i++)
            x0.set(i, rand.nextDouble());

        RosenbrockFunction f = new RosenbrockFunction();
        FunctionVec fp = f.getDerivative();
        double eta = 0.01;
        NAdaGrad instance = new NAdaGrad();
        instance.setup(x0.length());
        
        for(int i = 0; i < 100000; i++)
        {
            instance.update(x0, fp.f(x0).normalized(), eta);
            instance = instance.clone();
        }
        assertEquals(0.0, f.f(x0), 1e-1);
    }

    @Test
    public void testUpdate_5args()
    {
        System.out.println("update");
        Random rand = RandomUtil.getRandom();
        Vec xWithBias = new DenseVector(21);
        for(int i = 0; i < xWithBias.length(); i++)
            xWithBias.set(i, rand.nextDouble());
        
        Vec x0 = new SubVector(0, 20, xWithBias);

        RosenbrockFunction f = new RosenbrockFunction();
        FunctionVec fp = f.getDerivative();
        double eta = 0.01;
        
        
        NAdaGrad instance = new NAdaGrad();
        instance.setup(x0.length());
        
        for(int i = 0; i < 100000; i++)
        {
            double bias = xWithBias.get(20);
            Vec gradWithBias = fp.f(xWithBias);
            gradWithBias.normalize();
            double biasGrad = gradWithBias.get(20);
            Vec grad = new SubVector(0, 20, gradWithBias);
            double biasDelta = instance.update(x0, grad, eta, bias, biasGrad);
            xWithBias.set(20, bias-biasDelta);
            
            instance = instance.clone();
        }
        assertEquals(0.0, f.f(xWithBias), 1e-1);
        
    }
    
    @Test
    public void testUpdate_5args_scaled()
    {
        System.out.println("update");
        Random rand = RandomUtil.getRandom();
        
        //Test simple SGD LR on many different scalings of the data, all should work
        for(double scale = 0.0000001; scale <= 1000050; scale *= 10)
        {
            ClassificationDataSet train = FixedProblems.get2ClassLinear(2000, rand);
            ClassificationDataSet test = FixedProblems.get2ClassLinear(200, rand);
            
            LinearTransform transform = new LinearTransform(train, -1*scale, 1*scale);
            train.applyTransform(transform);
            test.applyTransform(transform);
            
            LinearSGD sgd = new LinearSGD(new LogisticLoss(), 0.5, new NoDecay(), 0.0, 0.0);
            sgd.setUseBias(true);
            sgd.setGradientUpdater(new NAdaGrad());
            
            ClassificationModelEvaluation cme = new ClassificationModelEvaluation(sgd, train);
            cme.evaluateTestSet(test);
            
            assertEquals(0.0, cme.getErrorRate(), 2.0/200);//should be 0 for all, AdaGrad or SGD would not be so sucesfull
        }
    }
    
}
