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
import jsat.DataSet;
import jsat.classifiers.*;
import jsat.linear.Vec;
import jsat.regression.RegressionDataSet;
import jsat.utils.DoubleList;
import jsat.utils.StringUtils;
import static java.lang.Character.isWhitespace;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import jsat.SimpleDataSet;
import jsat.linear.*;
import jsat.utils.*;

/**
 * Provides a reader and writer for CSV style datasets. This CSV reader supports
 * comments in CSVs (must begin with a single character) and categorical
 * features (columns must be specified when calling). Any number of newlines
 * will be treated as a single newline separating two rows.<br>
 * <br>
 * When reading and writing a CSV, if the delimiter or comment markers are not
 * specified - the defaults will be used {@link #DEFAULT_DELIMITER} and
 * {@link #DEFAULT_COMMENT} respectively.<br>
 * <br>
 * The CSV loader will treat empty columns as missing values for both numeric
 * and categorical features. A value of "NaN" in a numeric column will also be
 * treated as a missing value. Once loaded, missing values for numeric features
 * are encoded as {@link Double#NaN} and as <i>-1</i> for categorical features.
 *
 * @author Edward Raff
 */
public class CSV
{
    public static final char DEFAULT_DELIMITER = ',';
    public static final char DEFAULT_COMMENT = '#';

    private CSV()
    {
    }
    
    /**
     * Reads in a CSV dataset as a regression dataset.
     *
     * @param numeric_target_column the column index (starting from zero) of the
     * feature that will be the target regression value
     * @param path the reader for the CSV content
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return the regression dataset from the given CSV file
     * @throws IOException
     */
    public static RegressionDataSet readR(int numeric_target_column, Path path, int lines_to_skip, Set<Integer> cat_cols) throws IOException
    {
        return readR(numeric_target_column, path, DEFAULT_DELIMITER, lines_to_skip, DEFAULT_COMMENT, cat_cols);
    }
    
    /**
     * Reads in a CSV dataset as a regression dataset.
     *
     * @param numeric_target_column the column index (starting from zero) of the
     * feature that will be the target regression value
     * @param reader the reader for the CSV content
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return the regression dataset from the given CSV file
     * @throws IOException
     */
    public static RegressionDataSet readR(int numeric_target_column, Reader reader, int lines_to_skip, Set<Integer> cat_cols) throws IOException
    {
        return readR(numeric_target_column, reader, DEFAULT_DELIMITER, lines_to_skip, DEFAULT_COMMENT, cat_cols);
    }
    
    /**
     * Reads in a CSV dataset as a regression dataset.
     *
     * @param numeric_target_column the column index (starting from zero) of the
     * feature that will be the target regression value
     * @param path the CSV file to read
     * @param delimiter the delimiter to separate columns, usually a comma
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param comment the character used to indicate the start of a comment.
     * Once this character is reached, anything at and after the character will
     * be ignored.
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return the regression dataset from the given CSV file
     * @throws IOException
     */
    public static RegressionDataSet readR(int numeric_target_column, Path path, char delimiter, int lines_to_skip, char comment, Set<Integer> cat_cols) throws IOException
    {
        BufferedReader br = Files.newBufferedReader(path, Charset.defaultCharset());
        RegressionDataSet ret = readR(numeric_target_column, br, delimiter, lines_to_skip, comment, cat_cols);
        br.close();
        return ret;
    }
    
    /**
     * Reads in a CSV dataset as a regression dataset.
     *
     * @param numeric_target_column the column index (starting from zero) of the
     * feature that will be the target regression value
     * @param reader the reader for the CSV content
     * @param delimiter the delimiter to separate columns, usually a comma
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param comment the character used to indicate the start of a comment.
     * Once this character is reached, anything at and after the character will
     * be ignored.
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return the regression dataset from the given CSV file
     * @throws IOException
     */
    public static RegressionDataSet readR(int numeric_target_column, Reader reader, char delimiter, int lines_to_skip, char comment, Set<Integer> cat_cols) throws IOException
    {
        return (RegressionDataSet) readCSV(reader, lines_to_skip, delimiter, comment, cat_cols, numeric_target_column, -1);
    }
    
    /**
     * Reads in a CSV dataset as a classification dataset. Comments assumed to
     * start with the "#" symbol.
     * 
     * @param classification_target the column index (starting from zero) of the
     * feature that will be the categorical target value
     * @param path the CSV file to read
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return the classification dataset from the given CSV file
     * @throws IOException 
     */
    public static ClassificationDataSet readC(int classification_target, Path path, int lines_to_skip, Set<Integer> cat_cols) throws IOException
    {
        return readC(classification_target, path, DEFAULT_DELIMITER, lines_to_skip, DEFAULT_COMMENT, cat_cols);
    }
    
    /**
     * Reads in a CSV dataset as a classification dataset. Comments assumed to
     * start with the "#" symbol.
     * 
     * @param classification_target the column index (starting from zero) of the
     * feature that will be the categorical target value
     * @param reader the reader for the CSV content
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return the classification dataset from the given CSV file
     * @throws IOException 
     */
    public static ClassificationDataSet readC(int classification_target, Reader reader, int lines_to_skip, Set<Integer> cat_cols) throws IOException
    {
        return readC(classification_target, reader, DEFAULT_DELIMITER, lines_to_skip, DEFAULT_COMMENT, cat_cols);
    }
    
    /**
     * Reads in a CSV dataset as a classification dataset.
     * 
     * @param classification_target the column index (starting from zero) of the
     * feature that will be the categorical target value
     * @param reader the reader for the CSV content
     * @param delimiter the delimiter to separate columns, usually a comma
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param comment the character used to indicate the start of a comment.
     * Once this character is reached, anything at and after the character will
     * be ignored.
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return the classification dataset from the given CSV file
     * @throws IOException 
     */
    public static ClassificationDataSet readC(int classification_target, Reader reader, char delimiter, int lines_to_skip, char comment, Set<Integer> cat_cols) throws IOException
    {
        return (ClassificationDataSet) readCSV(reader, lines_to_skip, delimiter, comment, cat_cols, -1, classification_target);
    }
    
    /**
     * Reads in a CSV dataset as a classification dataset.
     * 
     * @param classification_target the column index (starting from zero) of the
     * feature that will be the categorical target value
     * @param path the CSV file
     * @param delimiter the delimiter to separate columns, usually a comma
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param comment the character used to indicate the start of a comment.
     * Once this character is reached, anything at and after the character will
     * be ignored.
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return the classification dataset from the given CSV file
     * @throws IOException 
     */
    public static ClassificationDataSet readC(int classification_target, Path path, char delimiter, int lines_to_skip, char comment, Set<Integer> cat_cols) throws IOException
    {
        BufferedReader br = Files.newBufferedReader(path, Charset.defaultCharset());
        ClassificationDataSet ret = readC(classification_target, br, delimiter, lines_to_skip, comment, cat_cols);
        br.close();
        return ret;
    }
    
    /**
     * Reads in the given CSV dataset as a simple CSV file
     * @param path the CSV file
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return a simple dataset of the given CSV file
     * @throws IOException 
     */
    public static SimpleDataSet read(Path path, int lines_to_skip, Set<Integer> cat_cols) throws IOException
    {
        return read(path, DEFAULT_DELIMITER, lines_to_skip, DEFAULT_COMMENT, cat_cols);
    }
    
    /**
     * Reads in the given CSV dataset as a simple CSV file
     * @param reader the reader for the CSV content
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return a simple dataset of the given CSV file
     * @throws IOException 
     */
    public static SimpleDataSet read(Reader reader, int lines_to_skip, Set<Integer> cat_cols) throws IOException
    {
        return read(reader, DEFAULT_DELIMITER, lines_to_skip, DEFAULT_COMMENT, cat_cols);
    }
    
    /**
     * Reads in the given CSV dataset as a simple CSV file
     * @param path the CSV file to read
     * @param delimiter the delimiter to separate columns, usually a comma
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param comment the character used to indicate the start of a comment.
     * Once this character is reached, anything at and after the character will
     * be ignored.
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return a simple dataset of the given CSV file
     * @throws IOException 
     */
    public static SimpleDataSet read(Path path, char delimiter, int lines_to_skip, char comment, Set<Integer> cat_cols) throws IOException
    {
        BufferedReader br = Files.newBufferedReader(path, Charset.defaultCharset());
        SimpleDataSet ret = read(br, delimiter, lines_to_skip, comment, cat_cols);
        br.close();
        return ret;
    }
    
    /**
     * Reads in the given CSV dataset as a simple CSV file
     * @param reader the reader for the CSV content
     * @param delimiter the delimiter to separate columns, usually a comma
     * @param lines_to_skip the number of lines to skip when reading in the CSV
     * (used to skip header information)
     * @param comment the character used to indicate the start of a comment.
     * Once this character is reached, anything at and after the character will
     * be ignored.
     * @param cat_cols a set of the indices to treat as categorical features.
     * @return a simple dataset of the given CSV file
     * @throws IOException 
     */
    public static SimpleDataSet read(Reader reader, char delimiter, int lines_to_skip, char comment, Set<Integer> cat_cols) throws IOException
    {
        return (SimpleDataSet) readCSV(reader, lines_to_skip, delimiter, comment, cat_cols, -1, -1);
    }
    
    private static DataSet<?> readCSV(Reader reader, int lines_to_skip, char delimiter, char comment, Set<Integer> cat_col, int numeric_target, int cat_target) throws IOException
    {
        StringBuilder processBuffer = new StringBuilder(20);
        StringBuilder charBuffer = new StringBuilder(1024);
        char[] read_buffer = new char[1024];
        
        /**
         * The target values if doing regression
         */
        DoubleList regressionTargets = new DoubleList();
        /**
         * The target values if doing classification
         */
        IntList catTargets = new IntList();
        
        /**
         * Fist mapping is for each column that contains categorical variables. 
         * The value map is a mapping from each string to its index, based on order seen. 
         */
        Map<Integer, Map<String, Integer>> seenCats = new HashMap<Integer, Map<String, Integer>>();
        for(int col : cat_col)
            if(col != cat_target)
            seenCats.put(col, new HashMap<String, Integer>());
        /**
         * a mapping from each string to its index, based on order seen, for the target class
         */
        Map<String, Integer> seenCats_target = new HashMap<String, Integer>();
        
        /**
         * 
         */
        Map<Integer, Integer> cat_indx_to_csv_column = new HashMap<Integer, Integer>();

        
        STATE state = STATE.INITIAL;
        int position = 0;
        
        /**
         * Negative value used to indicate that we don't know how many columns
         * there are yet. Once we process a single row, we set the number of
         * columns seen so we can sanity check
         */
        int totalCols = -1;
        DoubleList numericFeats = new DoubleList();
        IntList catFeats = new IntList();
        int cur_column = 0;
        
        List<Vec> all_vecs = new ArrayList<Vec>();
        List<int[]> all_cats = new ArrayList<int[]>();
        
        while(true)
        {
            
            while(charBuffer.length()-position <= 1)//make sure we have chars to handle
            {
                //move everything to the front
                charBuffer.delete(0, position);
                position = 0;
                
                int read = reader.read(read_buffer);
                if(read < 0)
                    break;
                charBuffer.append(read_buffer, 0, read);
            }
            
            if(charBuffer.length()-position == 0)//EOF, no more chars
            {
                //Look at the last state we were in before EOF
                if(state == STATE.NEWLINE)
                {
                    //nothing to do and everything already processed, just return
                    break;
                }
                else if(state == STATE.COMMENT)
                {
                    break;///nothing to do, values should have already been added once we transition to comment state
                }
                else if(state == STATE.VALUE)//line ended in the middle of processing
                {
                    charBuffer.append("\n");//append the wanted newline and let it run thought like normal
                }
                else
                    throw new RuntimeException();
                
            }
            
            //Normal processing of states
            char ch = charBuffer.charAt(position);
            switch(state)
            {
                case INITIAL:
                    if(lines_to_skip > 0)
                        state =  STATE.SKIPPING_ROWS;
                    else
                        state = STATE.VALUE;
                    break;
                case COMMENT://comment behaves basically the same as SKIPPING ROWS
                case SKIPPING_ROWS:
                    if(isNewLine(ch))
                    {
                        if(state == STATE.SKIPPING_ROWS)
                            lines_to_skip--;
                        state = STATE.NEWLINE;
                    }
                    else
                    {
                        //keep moving till we hit a new line
                        position++;
                    }
                    break;
                case VALUE:
                    
                    if(ch == delimiter || isNewLine(ch) || ch == comment )
                    {
                        //trim all the white space from the end of what we have been reading
                        while(processBuffer.length() > 0 && isWhitespace(processBuffer.charAt(processBuffer.length()-1)))
                            processBuffer.setLength(processBuffer.length()-1);
                        
                        //clean up the value we are looking at
                        if(cat_col.contains(cur_column) || cur_column == cat_target)
                        {
                            Map<String, Integer> map = (cur_column == cat_target) ? seenCats_target : seenCats.get(cur_column);
                            String cat_op = processBuffer.toString();
                            processBuffer.setLength(0);
                            
                            int val;
                            if(cat_op.length() == 0)
                                val = -1;
                            else
                            {
                                if(!map.containsKey(cat_op))
                                    map.put(cat_op, map.size());
                                val = map.get(cat_op);
                            }
                       
                            if (cur_column == cat_target)
                                if (val == -1)
                                    throw new RuntimeException("Categorical column can't have missing values!");
                                else
                                    catTargets.add(val);
                            else
                                catFeats.add(val);
                            

                            if(cur_column != cat_target)
                                cat_indx_to_csv_column.put(catFeats.size()-1, cur_column);
                        }
                        else//numeric feature
                        {
                            double val;
                            if(processBuffer.length() == 0)
                                val = Double.NaN;
                            else
                                val = StringUtils.parseDouble(processBuffer, 0, processBuffer.length());
                            processBuffer.setLength(0);
                            if(cur_column == numeric_target)
                            {
                                regressionTargets.add(val);
                            }
                            else//normal storage
                            {
                                numericFeats.add(val);
                            }
                        }
                        
                        //now do the state transitions
                        if(ch == delimiter)
                            state = STATE.DELIMITER;
                        else
                        {
                            if(ch == comment)
                                state = STATE.COMMENT;
                            else
                                state = STATE.NEWLINE;
                            
                            if(totalCols < 0)
                                totalCols = cur_column+1;
                            else if(totalCols != cur_column+1)
                                throw new RuntimeException("Inconsistent number of columns in CSV");
                            
                            //add out stuff to the list 
                            all_vecs.add(new DenseVector(numericFeats));
                            int[] cat_vals = new int[catFeats.size()];
                            for(int i = 0; i <cat_vals.length; i++)
                                cat_vals[i] = catFeats.getI(i);
                            all_cats.add(cat_vals);
                            
                            numericFeats.clear();
                            catFeats.clear();
                        }
                    }
                    else//process a character value
                    {
                        if(processBuffer.length() == 0 && Character.isWhitespace(ch))
                        {
                            //don't add leading whitespace to the buffer, just move to next char
                            position++;
                        }
                        else//normal value, add to buffer and increment to next char
                        {
                            processBuffer.append(ch);
                            position++;
                        }
                    }

                    break;
                case DELIMITER:
                    
                    if(ch == delimiter)
                    {
                        position++;
                        cur_column++;
                        state = STATE.VALUE;
                    }
                    else
                        throw new RuntimeException("BAD CSV");//how did we get here?
                    
                    break;
                case NEWLINE:
                    cur_column = 0;
                    if (isNewLine(ch))
                        position++;
                    else//now we move to next state
                    {
                        if (lines_to_skip > 0)
                        {
                            //keep skipping until we are out of lines to skip
                            state = STATE.SKIPPING_ROWS;
                        }
                        else
                        {
                            state = STATE.VALUE;
                        }
                    }
                    break;
            }
        }
        
        //ok, we read everything in - clean up time on the categorical features
        
        /**
         * we will sort each set of seen options so that we get the same feature
         * index ordering regardless of the order they occurred in the data
         */
        Map<Integer, Map<Integer, Integer>> cat_true_index = new HashMap<Integer, Map<Integer, Integer>>();
        
        Map<Integer, CategoricalData> catDataMap = new HashMap<Integer, CategoricalData>();
        if(cat_target >= 0)//added so it gets processed easily below
            seenCats.put(cat_target, seenCats_target);
        CategoricalData target_data = null;
        for( Map.Entry<Integer, Map<String, Integer>> main_entry : seenCats.entrySet())
        {
            HashMap<Integer, Integer> translator = new HashMap<Integer, Integer>();
            int col = main_entry.getKey();
            Map<String, Integer> catsSeen = main_entry.getValue();
            List<String> sortedOrder = new ArrayList<String>(catsSeen.keySet());
            Collections.sort(sortedOrder);
            
            CategoricalData cd = new CategoricalData(sortedOrder.size());
            if(col != cat_target)
                catDataMap.put(col, cd);
            else
                target_data = cd;
            for(int i = 0; i < sortedOrder.size(); i++)
            {
                translator.put(catsSeen.get(sortedOrder.get(i)), i);
                cd.setOptionName(sortedOrder.get(i), i);
            }
            
            cat_true_index.put(col, translator);
        }
        
        //go through and convert everything
        for(int[] cat_vals : all_cats)
        {
            for(int i = 0; i < cat_vals.length; i++)
            {
                if(cat_vals[i] >= 0)//if -1 its a missing value
                    cat_vals[i] = cat_true_index.get(cat_indx_to_csv_column.get(i)).get(cat_vals[i]);
            }
        }
        
        if(cat_target >= 0)//clean up the target value as well 
        {
            Map<Integer, Integer> translator = cat_true_index.get(cat_target);
            for(int i = 0; i < catTargets.size(); i++)
                catTargets.set(i, translator.get(catTargets.get(i)));
        }
        
        //collect the categorical variable headers
        CategoricalData[] cat_array = new CategoricalData[catDataMap.size()];
            for(int i = 0; i < cat_array.length; i++)
                cat_array[i]= catDataMap.get(cat_indx_to_csv_column.get(i));
            
        if(cat_target >= 0)
        {
            ClassificationDataSet d = new ClassificationDataSet(totalCols - cat_array.length-1, cat_array, target_data);
            for (int i = 0; i < all_vecs.size(); i++)
                d.addDataPoint(all_vecs.get(i), all_cats.get(i), catTargets.getI(i));

            return d;
        }
        else if (numeric_target >= 0)
        {
            RegressionDataSet d = new RegressionDataSet(totalCols - cat_array.length - 1, cat_array);
            for (int i = 0; i < all_vecs.size(); i++)
                d.addDataPoint(all_vecs.get(i), all_cats.get(i), regressionTargets.getD(i));

            return d;
        }
        else
        {
            SimpleDataSet d = new SimpleDataSet(cat_array, totalCols - cat_array.length);
            for (int i = 0; i < all_vecs.size(); i++)
                d.add(new DataPoint(all_vecs.get(i), all_cats.get(i), cat_array));

            return d;
        }
        
    }
    
    /**
     * Writes out the given dataset as a CSV file. If the given dataset is a
     * regression or classification dataset, the target feature that is being
     * predicted will always be written out as the first index in the CSV. <br>
     * After that, all numeric features will be written out in order, followed
     * by the categorical features.
     *
     * @param data the dataset object to save as a CSV file
     * @param path the path to write the CSV to
     * @throws IOException 
     */
    public static void write(DataSet<?> data, Path path) throws IOException
    {
        write(data, path, DEFAULT_DELIMITER);
    }
    
    /**
     * Writes out the given dataset as a CSV file. If the given dataset is a
     * regression or classification dataset, the target feature that is being
     * predicted will always be written out as the first index in the CSV. <br>
     * After that, all numeric features will be written out in order, followed
     * by the categorical features.
     *
     * @param data the dataset object to save as a CSV file
     * @param writer the output writer to write the CSV to
     * @throws IOException 
     */
    public static void write(DataSet<?> data, Writer writer) throws IOException
    {
        write(data, writer, DEFAULT_DELIMITER);
    }
    
    /**
     * Writes out the given dataset as a CSV file. If the given dataset is a
     * regression or classification dataset, the target feature that is being
     * predicted will always be written out as the first index in the CSV. <br>
     * After that, all numeric features will be written out in order, followed
     * by the categorical features.
     *
     * @param data the dataset object to save as a CSV file
     * @param path the path to write the CSV to
     * @param delimiter the delimiter between column values, normally a comma
     * @throws IOException 
     */
    public static void write(DataSet<?> data, Path path, char delimiter) throws IOException
    {
        BufferedWriter bw = Files.newBufferedWriter(path, Charset.defaultCharset());
        write(data, bw, delimiter);
        bw.close();
    }
    
    /**
     * Writes out the given dataset as a CSV file. If the given dataset is a
     * regression or classification dataset, the target feature that is being
     * predicted will always be written out as the first index in the CSV. <br>
     * After that, all numeric features will be written out in order, followed
     * by the categorical features.
     *
     * @param data the dataset object to save as a CSV file
     * @param writer the output writer to write the CSV to
     * @param delimiter the delimiter between column values, normally a comma
     * @throws IOException 
     */
    public static void write(DataSet<?> data, Writer writer, char delimiter) throws IOException
    {
        //first, create safe categorical feature names to write out
        String[][] catNamesToUse = getSafeNames(data.getCategories(), delimiter);
        String[] classNames = null;
        if(data instanceof ClassificationDataSet)
            classNames = getSafeNames(new CategoricalData[]{((ClassificationDataSet)data).getPredicting()}, delimiter)[0];
        
        //write out every data point
        for(int i = 0; i < data.getSampleSize(); i++)
        {
            if(i > 0)//write newline first
                writer.write('\n');
            boolean nothingWrittenYet = true;
            
            //target feature always goes at the front
            if(data instanceof ClassificationDataSet)
            {
                int targetClass = ((ClassificationDataSet)data).getDataPointCategory(i);
                writer.write(classNames[targetClass]);
                nothingWrittenYet = false;
            }
            else if(data instanceof RegressionDataSet)
            {
                double targetVal = ((RegressionDataSet)data).getTargetValue(i);
                writer.write(Double.toString(targetVal));
                nothingWrittenYet = false;
            }
            
            DataPoint dp = data.getDataPoint(i);
            Vec v =dp.getNumericalValues();
            int[] c = dp.getCategoricalValues();
            
            
            //write out numeric features first
            for(int j = 0; j < v.length(); j++)
            {
                if(!nothingWrittenYet)
                    writer.write(delimiter);
                
                //bellow handles NaN correctly, rint will just return NaN and then toString prints "NaN"
                double val = v.get(j);
                if(Math.rint(val) == val)//cast to long before writting to save space
                    writer.write(Long.toString((long) val));
                else
                    writer.write(Double.toString(val));
                nothingWrittenYet = false;
            }
            //then categorical features, useing the safe names we constructed earlier
            for(int j = 0; j < c.length; j++)
            {
                if(!nothingWrittenYet)
                    writer.write(delimiter);
                if(c[j] >= 0)
                    writer.write(catNamesToUse[j][c[j]]);
                //else, its negative - which is missing, so not writing anything out should result in the correct behavior
                nothingWrittenYet = false;
            }
        }
        
        writer.flush();
    }


    private static String[][] getSafeNames(CategoricalData[] cats, char delimiter)
    {
        String[][] catNamesToUse = new String[cats.length][];
        final char delim_replacement;
        if(delimiter == '_')//avoid setting the replacment to the deliminater value itself!
            delim_replacement = '-';
        else
            delim_replacement = '_';
        for(int i = 0; i < catNamesToUse.length; i++)
        {
            catNamesToUse[i] = new String[cats[i].getNumOfCategories()];
            for(int j = 0; j < catNamesToUse[i].length; j++)
            {
                String name = cats[i].getOptionName(j).trim();
                
                if(name.contains(String.valueOf(delimiter)))
                    name = name.replace(delimiter, delim_replacement);
                
                catNamesToUse[i][j] = name;
            }
        }
        return catNamesToUse;
    }
    
    private static boolean isNewLine(char ch)
    {
        return ch =='\n' || ch == '\r';
    }
    
    /**
     * Simple state machine used to parse CSV files
     */
    private enum STATE
    {
        /**
         * Initial state, doesn't actually do anything
         */
        INITIAL,
        /**
         * Used when we start and want to skip some fixed number of rows in the file
         */
        SKIPPING_ROWS,
        VALUE,
        DELIMITER,
        NEWLINE,
        /**
         * When we encounter the comment start character, run till we hit the end of the line
         */
        COMMENT,
    }
}
