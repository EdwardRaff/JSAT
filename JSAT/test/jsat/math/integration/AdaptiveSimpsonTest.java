/*
 * Copyright (C) 2018 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.math.integration;

import jsat.math.Function1D;
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
public class AdaptiveSimpsonTest
{
    
    public AdaptiveSimpsonTest()
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
     * Test of integrate method, of class AdaptiveSimpson.
     */
    @Test
    public void testIntegrate()
    {
        System.out.println("integrate");
        
        double tol = 1e-10;
        double expect, result;
        
        result = AdaptiveSimpson.integrate((x)->Math.sin(x), tol, -1, Math.PI);
        expect = 1.5403023058681397174;
        assertEquals(expect, result, tol);
        
        
        result = AdaptiveSimpson.integrate((x)->x*Math.exp(-x), tol, -1, 5);
        expect = -0.040427681994512802580;
        assertEquals(expect, result, tol);
        
        
        result = AdaptiveSimpson.integrate((x)->Math.log(x), tol, 0.1, 50);
        expect = 145.93140878070670750;
        assertEquals(expect, result, tol);
    }
    
}
