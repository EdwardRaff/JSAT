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
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.*;

import jsat.linear.*;
import jsat.regression.RegressionDataSet;
import jsat.utils.DoubleList;
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
public class CSVTest
{
    
    public CSVTest()
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
    @SuppressWarnings("unchecked")
    public void testReadNumericOnly() throws Exception
    {
        System.out.println("read (numeric only features)");
        
        List<String> testLines = new ArrayList<String>();
        List<Double> expetedLabel = new DoubleList();
        List<Vec> expectedVec = new ArrayList<Vec>();
        
        
        testLines.add("-1,0.0, 3.0, 0.0,   0.0, 0.0");//normal line
        expetedLabel.add(-1.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 3.0, 0.0, 0.0, 0.0));
        
        testLines.add("1,3.0,0.0,0.0,0.0,0.0"); //line ends in a space
        expetedLabel.add(1.0);
        expectedVec.add(DenseVector.toDenseVec( 3.0, 0.0, 0.0, 0.0, 0.0));
        
        testLines.add("2,0.0, 3.0, 3.0, 1.0, 0.0");//normal line with many values
        expetedLabel.add(2.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 3.0, 3.0, 1.0, 0.0));
        
        testLines.add("-1, 0.0, 3.0, 0.0, 2.0, 0.0");//extra spaces in between
        expetedLabel.add(-1.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 3.0, 0.0, 2.0, 0.0));
        
        testLines.add("1,0.0, 0.0, 0.0, 0.0, 0.0    ");  ///empty line
        expetedLabel.add(1.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, 0.0, 0.0));
        
        testLines.add("  2  , 0.0, 0.0, 0.0, 0.0, 0.0"); // empty line with space 
        expetedLabel.add(2.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, 0.0, 0.0));
        
        testLines.add("3, 0.0, 0.0, 0.0, 0.0, 0.0");
        expetedLabel.add(3.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, 0.0, 0.0));
        
        testLines.add("3,0.0, 0.0, 0.0, 0.0, 0.0");
        expetedLabel.add(3.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, 0.0, 0.0));
        
        testLines.add("-1, 10.0, 0.0, \t 2.0, 0.0, 0.0"); //extra spaces at the end
        expetedLabel.add(-1.0);
        expectedVec.add(DenseVector.toDenseVec( 10.0, 0.0, 2.0, 0.0, 0.0));
        
        testLines.add("2, 0.0, 3.0, 3.0, 0.0, 1.0");//normal line with many values
        expetedLabel.add(2.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 3.0, 3.0, 0.0, 1.0));
        
        
        String[] newLines = new String[]{"\n", "\n\r", "\r\n", "\n\r\n"};
        
        char comment = '#';

        for (boolean endInNewLines : new boolean[]{true, false })
            for(boolean endsInComments : new boolean[]{true, false})
            for (String newLine : newLines)
                for (int i = 0; i < testLines.size(); i++)
                {
                    StringBuilder input = new StringBuilder();
                    for (int j = 0; j < i; j++)
                    {
                        input.append(testLines.get(j));
                        if(endsInComments)
                        {
                            input.append(comment);
                            input.append(testLines.get(j));
                        }
                        input.append(newLine);
                    }
                    input.append(testLines.get(i));
                    if (endsInComments)
                    {
                        input.append(comment);
                        input.append(testLines.get(i));
                    }
                    if (endInNewLines)
                        input.append(newLine);

                    @SuppressWarnings("unchecked")
                    RegressionDataSet regData = CSV.readR(0, new StringReader(input.toString()), ',', 0, comment, Collections.EMPTY_SET);
                    SimpleDataSet smpData = CSV.read(new StringReader(input.toString()), ',', 0, comment, Collections.EMPTY_SET);
                    ClassificationDataSet catData = CSV.readC(0, new StringReader(input.toString()), ',', 0, comment, Collections.EMPTY_SET);

                    assertEquals(i + 1, regData.getSampleSize());
                    for (int j = 0; j < i + 1; j++)
                    {
                        Vec ex_vec = expectedVec.get(j);
                        double ex_target = expetedLabel.get(j);
                        assertEquals(ex_target, regData.getTargetValue(j), 0.0);
                        assertTrue(ex_vec.equals(regData.getDataPoint(j).getNumericalValues()));
                        
                        //simpled dataset test
                        Vec ex_vec_inc_target = new ConcatenatedVec(new ConstantVector(ex_target, 1), ex_vec);
                        
                        assertTrue(ex_vec_inc_target.equals(smpData.getDataPoint(j).getNumericalValues()));
                        
                        //classification test
                        assertTrue(ex_vec.equals(catData.getDataPoint(j).getNumericalValues()));
                        int pred_indx = catData.getDataPointCategory(j);
                        assertEquals(Integer.toString((int) ex_target) , catData.getPredicting().getOptionName(pred_indx));
                    }
                }
    }

    
    @Test
    public void testReadNumericCatFeats() throws Exception
    {
        System.out.println("read");
        
        List<String> testLines = new ArrayList<String>();
        List<Double> expetedLabel = new DoubleList();
        List<Vec> expectedVec = new ArrayList<Vec>();
        List<int[]> expectedCats = new ArrayList<int[]>();
        
        CategoricalData[] cats = new CategoricalData[]
        {
            new CategoricalData(2),
            new CategoricalData(4),
        };
        cats[0].setOptionName("A", 0);
        cats[0].setOptionName("B", 1);
        
        
        cats[1].setOptionName("a", 0);
        cats[1].setOptionName("b", 1);
        cats[1].setOptionName("c", 2);
        cats[1].setOptionName("d", 3);
        
        CategoricalData predicting = new CategoricalData(4);
        predicting.setOptionName("-1", 0);
        predicting.setOptionName("1", 1);
        predicting.setOptionName("2", 2);
        predicting.setOptionName("3", 3);
        
        
        testLines.add("-1,0.0, A, 3.0, 0.0,   0.0, a, 0.0");//normal line
        expetedLabel.add(-1.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 3.0, 0.0, 0.0, 0.0));
        expectedCats.add(new int[]{0, 0});
        
        testLines.add("1,3.0,B,0.0,0.0,0.0,b,0.0"); //line ends in a space
        expetedLabel.add(1.0);
        expectedVec.add(DenseVector.toDenseVec( 3.0, 0.0, 0.0, 0.0, 0.0));
        expectedCats.add(new int[]{1, 1});
        
        testLines.add("2,0.0, A,3.0, 3.0, 1.0, c, 0.0");//normal line with many values
        expetedLabel.add(2.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 3.0, 3.0, 1.0, 0.0));
        expectedCats.add(new int[]{0, 2});
        
        testLines.add("-1, 0.0,A , 3.0, 0.0, 2.0, d, 0.0");//extra spaces in between
        expetedLabel.add(-1.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 3.0, 0.0, 2.0, 0.0));
        expectedCats.add(new int[]{0, 3});
        
        testLines.add("1,0.0,   B    , 0.0, 0.0, 0.0,     a   , 0.0    ");  ///empty line
        expetedLabel.add(1.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, 0.0, 0.0));
        expectedCats.add(new int[]{1, 0});
        
        testLines.add("  2  , 0.0, A, 0.0, 0.0, 0.0, b, 0.0"); // empty line with space 
        expetedLabel.add(2.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, 0.0, 0.0));
        expectedCats.add(new int[]{0, 1});
        
        testLines.add("3, 0.0, B, 0.0, 0.0, 0.0, b, 0.0");
        expetedLabel.add(3.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, 0.0, 0.0));
        expectedCats.add(new int[]{1, 1});
        
        testLines.add("3,0.0, B, 0.0, 0.0, 0.0, d , 0.0");
        expetedLabel.add(3.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, 0.0, 0.0));
        expectedCats.add(new int[]{1, 3});
        
        testLines.add("-1, 10.0, A, 0.0, \t 2.0, 0.0, c, 0.0"); //extra spaces at the end
        expetedLabel.add(-1.0);
        expectedVec.add(DenseVector.toDenseVec( 10.0, 0.0, 2.0, 0.0, 0.0));
        expectedCats.add(new int[]{0, 2});
        
        testLines.add("2, 0.0,    A,   3.0, 3.0, 0.0, c , 1.0");//normal line with many values
        expetedLabel.add(2.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 3.0, 3.0, 0.0, 1.0));
        expectedCats.add(new int[]{0, 2});
        
        testLines.add("3,0.0, B, 0.0, 0.0, 0.0, d , NaN");//Nan numeric strng
        expetedLabel.add(3.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, 0.0, Double.NaN));
        expectedCats.add(new int[]{1, 3});
        
        testLines.add("3,0.0, B, 0.0, 0.0, , d , 0.0");//Nan numeric empty string
        expetedLabel.add(3.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 0.0, 0.0, Double.NaN, 0.0));
        expectedCats.add(new int[]{1, 3});
        
        testLines.add("2, 0.0,    A,   3.0, 3.0, 0.0,  , 1.0");//missing cat feat
        expetedLabel.add(2.0);
        expectedVec.add(DenseVector.toDenseVec( 0.0, 3.0, 3.0, 0.0, 1.0));
        expectedCats.add(new int[]{0, -1});
        
        
        Set<Integer> cat_cols = new HashSet<Integer>(Arrays.asList(2, 6));
        //when read back in, cat features are at the end b/c we always write out cats at the end
        Set<Integer> cat_cols_reed_back = new HashSet<Integer>(Arrays.asList(6, 7));
        
        String[] newLines = new String[]{"\n", "\n\r", "\r\n", "\n\r\n"};
        
        char comment = '#';

        for (boolean endInNewLines : new boolean[]{true, false })
            for(boolean endsInComments : new boolean[]{true, false})
            for (String newLine : newLines)
                for (int i = 0; i < testLines.size(); i++)
                {
                    StringBuilder input = new StringBuilder();
                    for (int j = 0; j < i; j++)
                    {
                        input.append(testLines.get(j));
                        if(endsInComments)
                        {
                            input.append(comment);
                            input.append(testLines.get(j));
                        }
                        input.append(newLine);
                    }
                    input.append(testLines.get(i));
                    if (endsInComments)
                    {
                        input.append(comment);
                        input.append(testLines.get(i));
                    }
                    if (endInNewLines)
                        input.append(newLine);

                    @SuppressWarnings("unchecked")
                    RegressionDataSet regData = CSV.readR(0, new StringReader(input.toString()), ',', 0, comment, cat_cols);
                    SimpleDataSet smpData = CSV.read(new StringReader(input.toString()), ',', 0, comment, cat_cols);
                    @SuppressWarnings("unchecked")
                    ClassificationDataSet catData = CSV.readC(0, new StringReader(input.toString()), ',', 0, comment, cat_cols);
                    
                    //Confirm that string -> data -> string -> data keeps everything the same
                    StringWriter data_strng_out = new StringWriter();
                    CSV.write(smpData, data_strng_out);
                    SimpleDataSet readBackIn = CSV.read(new StringReader(data_strng_out.toString()), 0, cat_cols_reed_back);
                    compareDataSetPoints(smpData, readBackIn);

                    assertEquals(i + 1, regData.getSampleSize());
                    for (int j = 0; j < i + 1; j++)
                    {
                        Vec ex_vec = expectedVec.get(j);
                        double ex_target = expetedLabel.get(j);
                        int[] ex_cats = expectedCats.get(j);
                        
                        int[] cats_vals;
                        
                        String[] ex_cats_s = new String[ex_cats.length];
                        for(int k = 0; k < ex_cats.length; k++)
                            ex_cats_s[k] = cats[k].getOptionName(ex_cats[k]);
                        
                        assertEquals(ex_target, regData.getTargetValue(j), 0.0);
                        assertTrue(ex_vec.equals(regData.getDataPoint(j).getNumericalValues()));
                        cats_vals = regData.getDataPoint(j).getCategoricalValues();
                        for(int k = 0; k < cats_vals.length; k++)
                            assertEquals(ex_cats_s[k], regData.getCategories()[k].getOptionName(cats_vals[k]));
                        
                        //simpled dataset test
                        Vec ex_vec_inc_target = new ConcatenatedVec(new ConstantVector(ex_target, 1), ex_vec);
                        
                        assertTrue(ex_vec_inc_target.equals(smpData.getDataPoint(j).getNumericalValues()));
                        cats_vals = smpData.getDataPoint(j).getCategoricalValues();
                        for(int k = 0; k < cats_vals.length; k++)
                            assertEquals(ex_cats_s[k], smpData.getCategories()[k].getOptionName(cats_vals[k]));
                        
                        //classification test
                        assertTrue(ex_vec.equals(catData.getDataPoint(j).getNumericalValues()));
                        int pred_indx = catData.getDataPointCategory(j);
                        assertEquals(Integer.toString((int) ex_target) , catData.getPredicting().getOptionName(pred_indx));
                        cats_vals = catData.getDataPoint(j).getCategoricalValues();
                        for(int k = 0; k < cats_vals.length; k++)
                            assertEquals(ex_cats_s[k], catData.getCategories()[k].getOptionName(cats_vals[k]));
                    }
                }
    }
    
    @Test
    public void testWriteRead()
    {
        CategoricalData[] cats = new CategoricalData[]
        {
            new CategoricalData(2),
            new CategoricalData(4),
            new CategoricalData(3),
        };
        cats[0].setOptionName("A", 0);
        cats[0].setOptionName("B", 1);
        
        
        cats[1].setOptionName("a", 0);
        cats[1].setOptionName("b", 1);
        cats[1].setOptionName("c", 2);
        cats[1].setOptionName("d", 3);
        
        cats[2].setOptionName("hello", 0);
        cats[2].setOptionName("hello_world", 1);
        cats[2].setOptionName("whats up?", 2);

        
        SimpleDataSet truth_data = new SimpleDataSet(cats, 3);
        Random rand = RandomUtil.getRandom();
        for (int i = 0; i < 100; i++)
        {
            DenseVector dv = new DenseVector(3);
            int[] vals = new int[3];
            for (int j = 0; j < 3; j++)
            {
                dv.set(j, rand.nextInt(20));
                vals[j] = rand.nextInt(cats[j].getNumOfCategories());
            }
            truth_data.add(new DataPoint(dv, vals, cats));
        }

        for (int lines_to_skip = 0; lines_to_skip < 10; lines_to_skip++)
        {
            StringBuilder extraLines = new StringBuilder();
            
            for(int i = 0; i < lines_to_skip; i++)
            {
                for(int j = 0; j < rand.nextInt(1000)+1; j++)
                    extraLines.append(i);
                extraLines.append("\n");
            }
            
            try
            {
                StringWriter writter = new StringWriter();
                CSV.write(truth_data, writter, ',');
                SimpleDataSet simpleIn = CSV.read(new StringReader(extraLines.toString()+writter.toString()), ',', lines_to_skip, '#', new HashSet<Integer>(Arrays.asList(3, 4, 5)));
                compareDataSetPoints(truth_data, simpleIn);
            }
            catch (IOException ex)
            {
                Logger.getLogger(CSVTest.class.getName()).log(Level.SEVERE, null, ex);
            }

            try
            {
                for (int indx = 0; indx < 3; indx++)
                {
                    StringWriter writter = new StringWriter();
                    ClassificationDataSet trutch_c = truth_data.asClassificationDataSet(indx);
                    CSV.write(trutch_c, writter, ',');
                    ClassificationDataSet in = CSV.readC(0, new StringReader(extraLines.toString()+writter.toString()), ',', lines_to_skip, '#', new HashSet<Integer>(Arrays.asList(4, 5)));
                    compareDataSetPoints(trutch_c, in);
                    for (int i = 0; i < trutch_c.getSampleSize(); i++)
                    {
                        String exp_s = trutch_c.getPredicting().getOptionName(trutch_c.getDataPointCategory(i));
                        String found_s = in.getPredicting().getOptionName(in.getDataPointCategory(i));
                        assertEquals(exp_s, found_s);
                    }
                }
            }
            catch (IOException ex)
            {
                Logger.getLogger(CSVTest.class.getName()).log(Level.SEVERE, null, ex);
            }

            try
            {
                for (int indx = 0; indx < 3; indx++)
                {
                    StringWriter writter = new StringWriter();
                    RegressionDataSet trutch_r = truth_data.asRegressionDataSet(indx);
                    CSV.write(trutch_r, writter, ',');
                    RegressionDataSet in = CSV.readR(0, new StringReader(extraLines.toString()+writter.toString()), ',', lines_to_skip, '#', new HashSet<Integer>(Arrays.asList(3, 4, 5)));
                    compareDataSetPoints(trutch_r, in);
                    assertTrue(trutch_r.getTargetValues().equals(in.getTargetValues()));
                }
            }
            catch (IOException ex)
            {
                Logger.getLogger(CSVTest.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    private void compareDataSetPoints(DataSet<?> truth_data, DataSet<?> simpleIn)
    {
        assertEquals(truth_data.getSampleSize(), simpleIn.getSampleSize());
        assertEquals(truth_data.getNumCategoricalVars(), simpleIn.getNumCategoricalVars());
        assertEquals(truth_data.getNumNumericalVars(), simpleIn.getNumNumericalVars());
        
        for(int i = 0;i < truth_data.getSampleSize(); i++)
        {
            DataPoint exp = truth_data.getDataPoint(i);
            DataPoint found = simpleIn.getDataPoint(i);
            assertTrue(exp.getNumericalValues().equals(found.getNumericalValues()));
            
            
            for(int k = 0; k < truth_data.getNumCategoricalVars(); k++)
            {
                String exp_s = truth_data.getCategories()[k].getOptionName(exp.getCategoricalValue(k));
                String found_s = simpleIn.getCategories()[k].getOptionName(found.getCategoricalValue(k));
                assertEquals(exp_s, found_s);
            }
        }
    }
}
