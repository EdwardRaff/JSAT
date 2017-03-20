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

package jsat.linear.vectorcollection;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.linear.VecPaired;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.random.RandomUtil;
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
public class VectorArrayTest
{
    static List<Vec> simpleSet;
    
    public VectorArrayTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        simpleSet = new ArrayList<Vec>();
        for(int i = 0 ; i < 1000; i++)
            simpleSet.add(DenseVector.toDenseVec(i));
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
     * Test of search method, of class VectorArray.
     */
    @Test
    public void testSearch_Vec_double()
    {
        System.out.println("search");
        Random rand = RandomUtil.getRandom();
        
        VectorArray<Vec> vecCol = new VectorArray<Vec>(new EuclideanDistance());
        vecCol.addAll(simpleSet);
        
        for(int iters = 0; iters < 100; iters++)
            for(double range : new double[]{2.0, 5.0, 10.0})
            {
                int randIndex=  rand.nextInt(simpleSet.size());

                List<? extends VecPaired<Vec, Double>> found = vecCol.search(simpleSet.get(randIndex), range);

                int min = (int) Math.max(randIndex-range, 0);
                int max = (int) Math.min(randIndex+range, simpleSet.size()-1);

                for(Vec v : found)
                    assertTrue(min <= v.get(0) && v.get(0) <= max);
                assertEquals(1+max-min, found.size());

            }
    }

    /**
     * Test of search method, of class VectorArray.
     */
    @Test
    public void testSearch_Vec_int()
    {
        System.out.println("search");
        Random rand = RandomUtil.getRandom();
        
        VectorArray<Vec> vecCol = new VectorArray<Vec>(new EuclideanDistance());
        for(Vec v : simpleSet)
            vecCol.add(v);
        
        for(int numNeighbours = 1; numNeighbours < 100; numNeighbours++)
        {
            //get from the midle to avoid more complicated code to hangle edges
            int randIndex=  numNeighbours+rand.nextInt(simpleSet.size()-numNeighbours*2);
            
            List<? extends VecPaired<Vec, Double>> found = vecCol.search(simpleSet.get(randIndex), numNeighbours);
            
            int min =  Math.max(randIndex-(numNeighbours)/2, 0);
            int max = Math.min(randIndex+(numNeighbours)/2, simpleSet.size()-1);
            
            for(Vec v : found)
                assertTrue(min <= v.get(0) && v.get(0) <= max);
            assertEquals(numNeighbours, found.size());
            
        }
    }

    
}
