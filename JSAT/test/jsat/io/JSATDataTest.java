/*
 * Copyright (C) 2015 Edward Raff
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
package jsat.io;

import java.io.*;
import java.util.Random;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.*;
import jsat.datatransform.DenseSparceTransform;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;
import jsat.text.GreekLetters;
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
 * @author Edward Raff
 */
public class JSATDataTest
{
    static SimpleDataSet simpleData;
    
    static SimpleDataSet byteIntegerData;
    
    public JSATDataTest()
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
        CategoricalData[] categories = new CategoricalData[3];
        categories[0] = new CategoricalData(2);
        categories[1] = new CategoricalData(3);
        categories[2] = new CategoricalData(5);
        
        categories[1].setCategoryName("I love " + GreekLetters.pi);//unicide to exercise non-ascii writer
        
        simpleData = new SimpleDataSet(categories, 20);
        
        Random rand = RandomUtil.getRandom();
        
        for(int i = 0; i < 10; i++)
        {
            int[] catVals = new int[categories.length];
            for(int j = 0; j < categories.length; j++)
                catVals[j] = rand.nextInt(categories[j].getNumOfCategories());
            
            double[] numeric = new double[simpleData.getNumNumericalVars()];
            for(int j = 0; j < numeric.length/3; j++)
            {
                numeric[rand.nextInt(numeric.length)] = rand.nextDouble();
            }
            
            simpleData.add(new DataPoint(new DenseVector(numeric), catVals, categories, rand.nextDouble()));
        }
        
        
        
        byteIntegerData = new SimpleDataSet(categories, 20);
        
        
        for(int i = 0; i < 10; i++)
        {
            int[] catVals = new int[categories.length];
            for(int j = 0; j < categories.length; j++)
                catVals[j] = rand.nextInt(categories[j].getNumOfCategories());
            
            double[] numeric = new double[simpleData.getNumNumericalVars()];
            for(int j = 0; j < numeric.length/3; j++)
            {
                numeric[rand.nextInt(numeric.length)] = rand.nextInt(Byte.MAX_VALUE);
            }
            
            byteIntegerData.add(new DataPoint(new DenseVector(numeric), catVals, categories, rand.nextInt(Byte.MAX_VALUE)));
        }
    }
    
    @After
    public void tearDown()
    {
    }
    
    @Test
    public void testReadWriteSimpleFPTypes() throws Exception
    {
        System.out.println("ReadWriteSimple");
        
        for(JSATData.FloatStorageMethod fpStoreMethod : JSATData.FloatStorageMethod.values())
        {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            JSATData.writeData(byteIntegerData, baos, fpStoreMethod);

            ByteArrayInputStream bin = new ByteArrayInputStream(baos.toByteArray());

            DataSet readBack = JSATData.load(bin);

            checkDataSet(byteIntegerData, readBack);

            byteIntegerData.applyTransform(new DenseSparceTransform(0.5));//sparcify our numeric values and try again

            baos = new ByteArrayOutputStream();
            JSATData.writeData(byteIntegerData, baos, fpStoreMethod);

            bin = new ByteArrayInputStream(baos.toByteArray());

            readBack = JSATData.load(bin);

            checkDataSet(byteIntegerData, readBack);
        }
    }

    /**
     * Test of writeData method, of class JSATData.
     */
    @Test
    public void testReadWriteSimple() throws Exception
    {
        System.out.println("ReadWriteSimple");
        
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        JSATData.writeData(simpleData, baos);
        
        ByteArrayInputStream bin = new ByteArrayInputStream(baos.toByteArray());
        
        DataSet readBack = JSATData.load(bin);
        
        checkDataSet(simpleData, readBack);
        
        simpleData.applyTransform(new DenseSparceTransform(0.5));//sparcify our numeric values and try again
        
        baos = new ByteArrayOutputStream();
        JSATData.writeData(simpleData, baos);
        
        bin = new ByteArrayInputStream(baos.toByteArray());
        
        readBack = JSATData.load(bin);
        
        checkDataSet(simpleData, readBack);
    }
    
    @Test
    public void testReadWriteClassification() throws Exception
    {
        System.out.println("ReadWriteClassification");
        
        
        //use the last categorical feature as the read target so that forcing as a standard dataset produces the same expected result as the original simple dataset
        ClassificationDataSet cds = simpleData.asClassificationDataSet(simpleData.getNumCategoricalVars()-1);
        
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        JSATData.writeData(cds, baos);
        
        ByteArrayInputStream bin = new ByteArrayInputStream(baos.toByteArray());
        
        
        //check classificaiton is right
        DataSet readBack = JSATData.load(bin);
        checkDataSet(cds, readBack);
        bin.reset();
        readBack = JSATData.loadClassification(bin);
        checkDataSet(cds, readBack);
        //check forcing as simple
        bin = new ByteArrayInputStream(baos.toByteArray());
        readBack = JSATData.load(bin, true);
        checkDataSet(simpleData, readBack);
        
        cds.applyTransform(new DenseSparceTransform(0.5));//sparcify our numeric values and try again
        simpleData.applyTransform(new DenseSparceTransform(0.5));
        
        baos = new ByteArrayOutputStream();
        JSATData.writeData(cds, baos);
        
        bin = new ByteArrayInputStream(baos.toByteArray());
        
        //check classificaiton is right
        readBack = JSATData.load(bin);
        checkDataSet(cds, readBack);
        bin.reset();
        readBack = JSATData.loadClassification(bin);
        checkDataSet(cds, readBack);
        //check forcing as simple
        bin = new ByteArrayInputStream(baos.toByteArray());
        readBack = JSATData.load(bin, true);
        checkDataSet(simpleData, readBack);
    }
    
    @Test
    public void testReadWriteRegression() throws Exception
    {
        System.out.println("ReadWriteRegression");
        
        
        //use the last categorical feature as the read target so that forcing as a standard dataset produces the same expected result as the original simple dataset
        RegressionDataSet rds = new RegressionDataSet(simpleData.getBackingList(), simpleData.getNumNumericalVars()-1);
        
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        JSATData.writeData(rds, baos);
        
        ByteArrayInputStream bin = new ByteArrayInputStream(baos.toByteArray());
        
        //check classificaiton is right
        DataSet readBack = JSATData.load(bin);
        checkDataSet(rds, readBack);
        bin.reset();
        readBack = JSATData.loadRegression(bin);
        checkDataSet(rds, readBack);
        //check forcing as simple
        bin = new ByteArrayInputStream(baos.toByteArray());
        readBack = JSATData.load(bin, true);
        checkDataSet(simpleData, readBack);
        
        rds.applyTransform(new DenseSparceTransform(0.5));//sparcify our numeric values and try again
        simpleData.applyTransform(new DenseSparceTransform(0.5));
        
        baos = new ByteArrayOutputStream();
        JSATData.writeData(rds, baos);
        
        bin = new ByteArrayInputStream(baos.toByteArray());
        
        //check classificaiton is right
        readBack = JSATData.load(bin);
        checkDataSet(rds, readBack);
        bin.reset();
        readBack = JSATData.loadRegression(bin);
        checkDataSet(rds, readBack);
        //check forcing as simple
        bin = new ByteArrayInputStream(baos.toByteArray());
        readBack = JSATData.load(bin, true);
        checkDataSet(simpleData, readBack);
    }

    private void checkDataSet(DataSet ogData, DataSet cpData)
    {
        assertEquals(ogData.getClass().getCanonicalName(), cpData.getClass().getCanonicalName());
        
        assertEquals(ogData.getNumNumericalVars(), cpData.getNumNumericalVars());
        assertEquals(ogData.getNumCategoricalVars(), cpData.getNumCategoricalVars());
        assertEquals(ogData.getSampleSize(), cpData.getSampleSize());
        
        CategoricalData[] og_cats = ogData.getCategories();
        CategoricalData[] cp_cats = ogData.getCategories();
        
        for(int i = 0; i < og_cats.length; i++)
        {
            assertEquals(og_cats[i].getCategoryName(), cp_cats[i].getCategoryName());
            assertEquals(og_cats[i].getNumOfCategories(), cp_cats[i].getNumOfCategories());
            for(int j = 0; j < og_cats[i].getNumOfCategories(); j++)
                assertEquals(og_cats[i].getOptionName(j), cp_cats[i].getOptionName(j));
        }
        
        //compare datapoint values
        for(int i = 0; i < ogData.getSampleSize(); i++)
        {
            DataPoint og = ogData.getDataPoint(i);
            DataPoint cp = cpData.getDataPoint(i);
            
            assertArrayEquals(og.getCategoricalValues(), cp.getCategoricalValues());
            Vec og_vec = og.getNumericalValues();
            Vec cp_vec = cp.getNumericalValues();
            assertTrue(og_vec.isSparse() == cp_vec.isSparse());
            assertTrue(og_vec.equals(cp_vec));
            
            assertEquals(og.getWeight(), cp.getWeight(), 0.0);
            
        }
        
        if(ogData instanceof ClassificationDataSet)
        {
            ClassificationDataSet ogC = (ClassificationDataSet) ogData;
            ClassificationDataSet cpC = (ClassificationDataSet) cpData;
            
            for(int i = 0; i < ogData.getSampleSize(); i++)
                assertEquals(ogC.getDataPointCategory(i), cpC.getDataPointCategory(i));
        }
        
        if(ogData instanceof RegressionDataSet)
        {
            RegressionDataSet ogR = (RegressionDataSet) ogData;
            RegressionDataSet cpR = (RegressionDataSet) cpData;
            
            for(int i = 0; i < ogData.getSampleSize(); i++)
                assertEquals(ogR.getTargetValue(i), cpR.getTargetValue(i), 0.0);
        }
        
    }

    
}
