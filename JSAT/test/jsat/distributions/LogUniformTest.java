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
package jsat.distributions;

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
public class LogUniformTest
{
    static double[] testVals = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    public LogUniformTest()
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
     * Test of pdf method, of class LogUniform.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        
        LogUniform instance = new LogUniform(1e-2, 10);
        
        double[] expected = new double[]
        {
            0, 0.14476482730108394, 0.072382413650541971, 0.048254942433694648, 
            0.036191206825270986, 0.028952965460216789, 0.024127471216847324, 
            0.020680689614440563, 0.018095603412635493, 0.016084980811231549, 
            0.014476482730108394
        };
        
        for(int i = 0; i < testVals.length; i++)
            assertEquals(expected[i], instance.pdf(testVals[i]), 1e-10);
    }

    @Test
    public void testLogPdf()
    {
        System.out.println("pdf");
        
        LogUniform instance = new LogUniform(1e-2, 10);
        
        double[] expected = new double[]
        {
            -Double.MAX_VALUE,-1.9326447339160655,-2.62579191447601080,-3.0312570225841752,-3.3189390950359561,-3.5420826463501659,-3.7244042031441205,-3.8785548829713788,-4.0120862755959014,-4.1298693112522849,-4.2352298269101112
        };
        
        for(int i = 0; i < testVals.length; i++)
            assertEquals(expected[i], instance.logPdf(testVals[i]), 1e-10);
    }

    /**
     * Test of cdf method, of class LogUniform.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        
        LogUniform instance = new LogUniform(1e-2, 10);
        
        double[] expected = new double[]
        {
            0, 0.66666666666666667, 0.76700999855466040, 
            0.82570708490655415, 0.86735333044265413, 0.89965666811200627, 
            0.92605041679454788, 0.94836601333808561, 0.96769666233064786, 
            0.98474750314644162, 1.0000000000000000
        };
        
        for(int i = 0; i < testVals.length; i++)
            assertEquals(expected[i], instance.cdf(testVals[i]), 1e-10);
    }

    /**
     * Test of invCdf method, of class LogUniform.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        
        LogUniform instance = new LogUniform(1e-2, 10);
        
        double[] expected = new double[]
        {
            0.014125375446227543, 0.019952623149688796, 0.028183829312644538, 
            0.039810717055349725, 0.056234132519034908, 0.079432823472428150,
            0.11220184543019634, 0.15848931924611135, 0.22387211385683396, 
            0.31622776601683793, 0.44668359215096312, 0.63095734448019325, 
            0.89125093813374553, 1.2589254117941672, 1.7782794100389228, 
            2.5118864315095801, 3.5481338923357546, 5.0118723362727229, 
            7.0794578438413791
        };
        
        double[] ps = new double[]
        {
            0.05, 0.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7,
            .75, .8, .85, .9, .95
        };
        
        for(int i = 0; i < ps.length; i++)
            assertEquals(expected[i], instance.invCdf(ps[i]), 1e-10);
    }

    /**
     * Test of skewness method, of class LogUniform.
     */
    @Test
    public void testSampleAndStats()
    {
        System.out.println("skewness");
        LogUniform instance = new LogUniform(1e-2, 10);
        
        Vec samples = instance.sampleVec(10000, RandomUtil.getRandom());
        assertEquals(instance.mean(), samples.mean(), 0.1);
        assertEquals(instance.median(), samples.median(), 0.1);
        assertEquals(instance.variance(), samples.variance(), 0.3);
    }

    /**
     * Test of min method, of class LogUniform.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        LogUniform instance = new LogUniform(1e-2, 10);
        assertEquals(1e-2, instance.min(), 0.0);
    }

    /**
     * Test of max method, of class LogUniform.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        LogUniform instance = new LogUniform(1e-2, 10);
        assertEquals(10, instance.max(), 0.0);
    }
    
}
