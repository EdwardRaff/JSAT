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
package jsat.datatransform.visualization;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.EuclideanDistance;
import jsat.utils.SystemInfo;
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
public class MDSTest
{
    
    public MDSTest()
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
     * Test of transform method, of class MDS.
     */
    @Test
    public void testTransform_DataSet_ExecutorService()
    {
        System.out.println("transform");
        
        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        
        Random rand = RandomUtil.getRandom();
        MDS instance = new MDS();
        
        //create a small data set, and apply a random projection to a higher dimension
        //shouuld still be able to distances on the same scaling
        
        Matrix orig_dim = new DenseMatrix(20, 2);
        for (int i = 0; i < orig_dim.rows(); i++)
        {
            int offset = i % 2 == 0 ? -5 : 5;
            for (int j = 0; j < orig_dim.cols(); j++)
            {
                orig_dim.set(i, j, rand.nextDouble() + offset);
            }
        }
        
        Matrix s = Matrix.random(2, 4, rand);
        
        Matrix proj_data = orig_dim.multiply(s);
        
        SimpleDataSet proj = new SimpleDataSet(new CategoricalData[0], proj_data.cols());
        for(int i = 0; i < proj_data.rows(); i++)
            proj.add(new DataPoint(proj_data.getRow(i)));
        
        SimpleDataSet transformed_0 = instance.transform(proj, ex);
        SimpleDataSet transformed_1 = instance.transform(proj);
        
        for(SimpleDataSet transformed : new SimpleDataSet[]{transformed_0, transformed_1})
        {
        
            EuclideanDistance dist = new EuclideanDistance();

            for(int i = 0; i < orig_dim.rows(); i++)
                for(int j = 0; j < orig_dim.rows(); j++)
                {
                    Vec orig_i = orig_dim.getRowView(i);
                    Vec orig_j = orig_dim.getRowView(j);

                    Vec new_i = transformed.getDataPoint(i).getNumericalValues();
                    Vec new_j = transformed.getDataPoint(j).getNumericalValues();

                    double d_o = dist.dist(orig_i, orig_j);
                    double d_n = dist.dist(new_i, new_j);

                    //assert the magnitudes are about the same
                    if(d_o > 6)
                        assertTrue(d_n > 6);
                    else//do is small, we should also be small
                        assertTrue(d_o < 2);
                }
        }
        
        ex.shutdown();
    }
    
}
