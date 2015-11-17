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
public class ContinuousDistributionTest
{
    static private final ContinuousDistribution dumbNormal_0_1 = new ContinuousDistribution()
    {

        @Override
        public double pdf(double x)
        {
            return Normal.pdf(x, 0, 1);
        }

        @Override
        public String getDistributionName()
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public String[] getVariables()
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public double[] getCurrentVariableValues()
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public void setVariable(String var, double value)
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public ContinuousDistribution clone()
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public void setUsingData(Vec data)
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public double mode()
        {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

        @Override
        public double min()
        {
            return Double.NEGATIVE_INFINITY;
        }

        @Override
        public double max()
        {
            return Double.POSITIVE_INFINITY;
        }
    };
    
    public ContinuousDistributionTest()
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
     * Test of logPdf method, of class ContinuousDistribution.
     */
    @Test
    public void testLogPdf()
    {
        System.out.println("logPdf");
        Normal norm = new Normal();
        for(double i = -3; i < 3; i += 0.1)
            assertEquals(norm.logPdf(i), dumbNormal_0_1.logPdf(i), 0.01);
    }


    /**
     * Test of cdf method, of class ContinuousDistribution.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        Normal norm = new Normal();
        for(double i = -3; i < 3; i += 0.1)
        {
            assertEquals(norm.cdf(i), dumbNormal_0_1.cdf(i), 0.01);
        }
    }
   
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        Normal norm = new Normal();
        for(double p = 0.01; p < 1; p += 0.1)
        {
            assertEquals(norm.invCdf(p), dumbNormal_0_1.invCdf(p), 0.01);
        }
    }

    /**
     * Test of mean method, of class ContinuousDistribution.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        Normal norm = new Normal();
        assertEquals(norm.mean(), dumbNormal_0_1.mean(), 0.01);
    }

    /**
     * Test of variance method, of class ContinuousDistribution.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        Normal norm = new Normal();
        assertEquals(norm.variance(), dumbNormal_0_1.variance(), 0.01);
    }

    /**
     * Test of skewness method, of class ContinuousDistribution.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        Normal norm = new Normal();
        assertEquals(norm.skewness(), dumbNormal_0_1.skewness(), 0.01);
    }
    
}
