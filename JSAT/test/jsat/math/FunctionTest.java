/*
 * Copyright (C) 2018 Edward Raff
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
package jsat.math;

import java.util.Random;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.optimization.RosenbrockFunction;
import jsat.utils.random.RandomUtil;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author edwardraff
 */
public class FunctionTest {
    
    public FunctionTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
    }
    
    @AfterClass
    public static void tearDownClass() {
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }


    /**
     * Test of forwardDifference method, of class Function.
     */
    @Test
    public void testForwardDifference() {
        System.out.println("forwardDifference");
        RosenbrockFunction f = new RosenbrockFunction();
        FunctionVec trueDeriv = f.getDerivative();
        FunctionVec approxDeriv = Function.forwardDifference(f);
        Random rand = RandomUtil.getRandom();
        
        
        for(int d = 2; d < 10; d++)
            for(int iter = 0; iter < 100; iter++)
            {
                Vec x = new DenseVector(d);
                for(int i = 0; i < x.length(); i++)
                    x.set(i, rand.nextDouble()*2-1);
                
                Vec trueD = trueDeriv.f(x);
                Vec approxD = approxDeriv.f(x);
                
                assertTrue(trueD.equals(approxD, 1e-1));
            }
        
    }

}
