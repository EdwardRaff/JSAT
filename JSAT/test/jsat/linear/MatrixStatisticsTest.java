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
package jsat.linear;

import java.util.List;
import java.util.Random;
import jsat.distributions.multivariate.NormalM;
import jsat.utils.random.RandomUtil;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class MatrixStatisticsTest
{
    
    public MatrixStatisticsTest()
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
    public void testMCD()
    {
        Random rand = RandomUtil.getRandom();
        Vec mainMean = new DenseVector(5);
        Matrix mainCov = Matrix.random(5, 5, rand);
        mainCov = mainCov.multiplyTranspose(mainCov);
        
        Vec offMean = DenseVector.random(5, rand).multiply(1000);
        Matrix offCov = Matrix.random(5, 5, rand);
        offCov = offCov.multiplyTranspose(offCov).multiply(550);
        
        NormalM main = new NormalM(mainMean, mainCov);
        NormalM off = new NormalM(offMean, offCov);
        
        double mainRatio = 0.6;
        
        for(boolean parallel : new boolean[]{false,true})
        for(int N : new int[]{550, 1000, 1499, 3000})
        {
            int n_m = (int) (N*mainRatio);
            int n_o = (int) (N*(1-mainRatio));
            
            List<Vec> vecs = main.sample(n_m, rand);
            vecs.addAll(off.sample(n_o, rand));
            
            Vec estMean = new DenseVector(5);
            Matrix estCov = new DenseMatrix(5, 5);
            
            MatrixStatistics.FastMCD(estMean, estCov, vecs, parallel);
            
            assertTrue(estMean.equals(mainMean, 0.2));
            //Let cov be looser, but dif of 1.0 is still  way closer than the huge 55x of the offCov
            assertTrue(estCov.equals(mainCov, 0.5));
            
            //and confirm dumb estimate isn't very good
            estMean.zeroOut();
            MatrixStatistics.meanVector(estMean, vecs);
            estCov.zeroOut();
            MatrixStatistics.covarianceMatrix(estMean, estCov, vecs);
            
            assertFalse(estMean.equals(mainMean, 0.2));
            assertFalse(estCov.equals(mainCov, 1.0));
        }
    }
    
}
