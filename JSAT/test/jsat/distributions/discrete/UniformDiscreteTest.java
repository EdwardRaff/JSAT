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
package jsat.distributions.discrete;

import jsat.linear.Vec;
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
public class UniformDiscreteTest
{
    static int[] testVals = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    public UniformDiscreteTest()
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
    public void testLogPmf()
    {
        System.out.println("logPmf");
        UniformDiscrete instance = new UniformDiscrete(2, 9);
        
        double[] expected_7 = new double[]
        {
            -Double.MAX_VALUE,-Double.MAX_VALUE,-2.07944154167984,-2.07944154167984,-2.07944154167984,-2.07944154167984,-2.07944154167984,-2.07944154167984,-2.07944154167984,-2.07944154167984,-Double.MAX_VALUE
        };
        
        for(int i = 0; i < expected_7.length; i++)
        {
            assertEquals(expected_7[i], instance.logPmf(testVals[i]), 1e-4);
        }
        
    }

    @Test
    public void testPmf()
    {
        System.out.println("pmf");
        UniformDiscrete instance = new UniformDiscrete(2, 9);
        
        double[] expected_7 = new double[]
        {
            0,0,0.125000000000000,0.125000000000000,0.125000000000000,0.125000000000000,0.125000000000000,0.125000000000000,0.125000000000000,0.125000000000000,0
        };
        
        for(int i = 0; i < expected_7.length; i++)
        {
            assertEquals(expected_7[i], instance.pmf(testVals[i]), 1e-4);
        }
    }

    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        UniformDiscrete instance = new UniformDiscrete(2, 9);
        
        double[] expected_7 = new double[]
        {
            0,0,0.125000000000000,0.250000000000000,0.375000000000000,0.500000000000000,0.625000000000000,0.750000000000000,0.875000000000000,1.00000000000000,1.00000000000000
        };
        
        for(int i = 0; i < expected_7.length; i++)
        {
            assertEquals(expected_7[i], instance.cdf(testVals[i]), 1e-4);
            
            //its hard to get the right value for the probabilities right on the line, so lets nudge them a little to make sure we map to the right spot
            double val;
            if(i == 0)
                val = instance.invCdf(expected_7[i]*.99);
            else
                val = instance.invCdf(expected_7[i-1]+(expected_7[i]-expected_7[i-1])*0.95);
            
            double expected;
            if(testVals[i] >= instance.max())
                expected = instance.max();
            else if(testVals[i] <= instance.min())
                expected = instance.min();
            else
                expected = testVals[i];
            assertEquals(expected, val, 1e-3);
        }
    }

    @Test
    public void testSummaryStats()
    {
        System.out.println("stats");
        UniformDiscrete instance = new UniformDiscrete(2, 9);
        //mean, median, variance, standard dev, skew
        double[] expected_7_5 = {5.50000000000000, 5.00000000000000, 5.25000000000000, 2.29128784747792, 0};
        
        
        assertEquals(expected_7_5[0], instance.mean(), 1e-4);
        assertEquals(expected_7_5[1], instance.median(), 1e-4);
        assertEquals(expected_7_5[2], instance.variance(), 1e-4);
        assertEquals(expected_7_5[3], instance.standardDeviation(), 1e-4);
        assertEquals(expected_7_5[4], instance.skewness(), 1e-4);
        
    }

    
    @Test
    public void testSample()
    {
        System.out.println("sample");
        UniformDiscrete instance = new UniformDiscrete(2, 9);
        
        Vec samples = instance.sampleVec(10000, RandomUtil.getRandom());
        
        assertEquals(instance.mean(), samples.mean(), 2e-1);
//        assertEquals(instance.median(), samples.median(), 2e-1);
        assertTrue(instance.median() == 5 || instance.median() == 6);//both are valid for this case
        assertEquals(instance.variance(), samples.variance(), 2e-1);
        assertEquals(instance.standardDeviation(), samples.standardDeviation(), 2e-1);
        assertEquals(instance.skewness(), samples.skewness(), 2e-1);
        
    }
    
}
