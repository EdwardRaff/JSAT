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
package jsat.math.optimization.oned;

import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.FunctionBase;
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
public class GoldenSearchTest
{
    private static final FunctionBase easyMin_at_0 = new FunctionBase()
    {

        @Override
        public double f(Vec x)
        {
            return 1+Math.pow(x.get(0), 2);
        }
    };
    
    public GoldenSearchTest()
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
     * Test of findMin method, of class GoldenSearch.
     */
    @Test
    public void testFindMin()
    {
        System.out.println("findMin");
        
        for(double tol = 2; tol >= 1e-8; tol/=2)
        {
            double result = GoldenSearch.findMin(-10, 10, easyMin_at_0, tol, 1000);
            assertEquals(0.0, result, tol*1.5);
        }
    }
    
}
