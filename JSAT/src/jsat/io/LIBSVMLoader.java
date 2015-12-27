package jsat.io;

import java.io.*;
import java.util.*;
import jsat.DataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.datatransform.DenseSparceTransform;
import jsat.linear.*;
import jsat.regression.RegressionDataSet;
import jsat.utils.DoubleList;
import jsat.utils.StringUtils;

/**
 * Loads a LIBSVM data file into a {@link DataSet}. LIVSM files do not indicate 
 * whether or not the target variable is supposed to be numerical or 
 * categorical, so two different loading methods are provided. For a LIBSVM file
 * to be loaded correctly, it must match the LIBSVM spec without extensions. 
 * <br><br>
 * Each line should begin with a numeric value. This is either a regression 
 * target or a class label. <br>
 * Then, for each non zero value in the data set, a space should precede an 
 * integer value index starting from 1 followed by a colon ":" followed by a 
 * numeric feature value. <br> The single space at the beginning should be the 
 * only space. There should be no double spaces in the file. 
 * <br><br>
 * LIBSVM files do not explicitly specify the length of data vectors. This can 
 * be problematic if loading a testing and training data set, if the data sets 
 * do not include the same highest index as a non-zero value, the data sets will
 * have incompatible vector lengths. To resolve this issue, use the loading 
 * methods that include the optional {@code vectorLength} parameter to specify 
 * the length before hand. 
 * 
 * @author Edward Raff
 */
public class LIBSVMLoader
{
    private static boolean fastLoad = true;

    private LIBSVMLoader()
    {
    }
    
    /*
     * LIBSVM format is sparse
     * <VAL> <1 based Index>:<Value>
     * 
     */
    
    /**
     * Loads a new regression data set from a LIBSVM file, assuming the label is
     * a numeric target value to predict
     * 
     * @param file the file to load
     * @return a regression data set
     * @throws FileNotFoundException if the file was not found
     * @throws IOException if an error occurred reading the input stream
     */
    public static RegressionDataSet loadR(File file) throws FileNotFoundException, IOException
    {
        return loadR(file, 0.5);
    }
    
    /**
     * Loads a new regression data set from a LIBSVM file, assuming the label is
     * a numeric target value to predict
     * 
     * @param file the file to load
     * @param sparseRatio the fraction of non zero values to qualify a data 
     * point as sparse
     * @return a regression data set
     * @throws FileNotFoundException if the file was not found
     * @throws IOException if an error occurred reading the input stream
     */
    public static RegressionDataSet loadR(File file, double sparseRatio) throws FileNotFoundException, IOException
    {
        return loadR(file, sparseRatio, -1);
    }
    
    /**
     * Loads a new regression data set from a LIBSVM file, assuming the label is
     * a numeric target value to predict
     * 
     * @param file the file to load
     * @param sparseRatio the fraction of non zero values to qualify a data 
     * point as sparse
     * @param vectorLength the pre-determined length of each vector. If given a 
     * negative value, the largest non-zero index observed in the data will be 
     * used as the length. 
     * @return a regression data set
     * @throws FileNotFoundException if the file was not found
     * @throws IOException if an error occurred reading the input stream
     */
    public static RegressionDataSet loadR(File file, double sparseRatio, int vectorLength) throws FileNotFoundException, IOException
    {
        return loadR(new FileReader(file), sparseRatio, vectorLength);
    }
    
    /**
     * Loads a new regression data set from a LIBSVM file, assuming the label is
     * a numeric target value to predict
     * 
     * @param isr the input stream for the file to load
     * @param sparseRatio the fraction of non zero values to qualify a data 
     * point as sparse
     * @return a regression data set
     * @throws IOException if an error occurred reading the input stream
     */
    public static RegressionDataSet loadR(InputStreamReader isr, double sparseRatio) throws IOException
    {
        return loadR(isr, sparseRatio, -1);
    }
    
    /**
     * Loads a new regression data set from a LIBSVM file, assuming the label is
     * a numeric target value to predict.
     * 
     * @param reader the reader for the file to load
     * @param sparseRatio the fraction of non zero values to qualify a data 
     * point as sparse
     * @param vectorLength the pre-determined length of each vector. If given a 
     * negative value, the largest non-zero index observed in the data will be 
     * used as the length. 
     * @return a regression data set
     * @throws IOException 
     */
    public static RegressionDataSet loadR(Reader reader, double sparseRatio, int vectorLength) throws IOException
    {
        return (RegressionDataSet) loadG(reader, sparseRatio, vectorLength, false);
    }
    
    /**
     * Loads a new classification data set from a LIBSVM file, assuming the 
     * label is a nominal target value
     * 
     * @param file the file to load
     * @return a classification data set
     * @throws FileNotFoundException if the file was not found
     * @throws IOException if an error occurred reading the input stream
     */
    public static ClassificationDataSet loadC(File file) throws FileNotFoundException, IOException
    {
        return loadC(new FileReader(file), 0.5);
    }
    
    /**
     * Loads a new classification data set from a LIBSVM file, assuming the 
     * label is a nominal target value
     * 
     * @param file the file to load
     * @param sparseRatio the fraction of non zero values to qualify a data 
     * point as sparse
     * @return a classification data set
     * @throws FileNotFoundException if the file was not found
     * @throws IOException if an error occurred reading the input stream
     */
    public static ClassificationDataSet loadC(File file, double sparseRatio) throws FileNotFoundException, IOException
    {
        return loadC(file, sparseRatio, -1);
    }
    
    /**
     * Loads a new classification data set from a LIBSVM file, assuming the 
     * label is a nominal target value
     * 
     * @param file the file to load
     * @param sparseRatio the fraction of non zero values to qualify a data 
     * point as sparse
     * @param vectorLength the pre-determined length of each vector. If given a 
     * negative value, the largest non-zero index observed in the data will be 
     * used as the length. 
     * @return a classification data set
     * @throws FileNotFoundException if the file was not found
     * @throws IOException if an error occurred reading the input stream
     */
    public static ClassificationDataSet loadC(File file, double sparseRatio, int vectorLength) throws FileNotFoundException, IOException
    {
        return loadC(new FileReader(file), sparseRatio, vectorLength);
    }
    
    /**
     * Loads a new classification data set from a LIBSVM file, assuming the 
     * label is a nominal target value 
     * 
     * @param isr the input stream for the file to load
     * @param sparseRatio the fraction of non zero values to qualify a data 
     * point as sparse
     * @return a classification data set
     * @throws IOException if an error occurred reading the input stream
     */
    public static ClassificationDataSet loadC(InputStreamReader isr, double sparseRatio) throws IOException
    {
        return loadC(isr, sparseRatio, -1);
    }
    
    /**
     * Loads a new classification data set from a LIBSVM file, assuming the 
     * label is a nominal target value 
     * 
     * @param reader the input stream for the file to load
     * @param sparseRatio the fraction of non zero values to qualify a data 
     * point as sparse
     * @param vectorLength the pre-determined length of each vector. If given a 
     * negative value, the largest non-zero index observed in the data will be 
     * used as the length. 
     * @return a classification data set
     * @throws IOException if an error occurred reading the input stream
     */
    public static ClassificationDataSet loadC(Reader reader, double sparseRatio, int vectorLength) throws IOException
    {
        return (ClassificationDataSet) loadG(reader, sparseRatio, vectorLength, true);
    }
    
    /**
     * Generic loader for both Classification and Regression interpretations. 
     * @param reader
     * @param sparseRatio
     * @param vectorLength
     * @param classification {@code true} to treat as classification,
     * {@code false} to treat as regression
     * @return
     * @throws IOException 
     */
    private static DataSet loadG(Reader reader, double sparseRatio, int vectorLength, boolean classification) throws IOException
    {
        StringBuilder processBuffer = new StringBuilder(20);
        StringBuilder charBuffer = new StringBuilder(1024);
        char[] buffer = new char[1024];
        List<SparseVector> sparceVecs = new ArrayList<SparseVector>();
        /**
         * The category "label" for each value loaded in
         */
        List<Double> labelVals = new DoubleList();
        Map<Double, Integer> possibleCats = new HashMap<Double, Integer>();
        int maxLen= 1;
        
        STATE state = STATE.INITIAL;
        int position = 0;
        SparseVector tempVec = new SparseVector(1, 1);
        /**
         * The index that we have parse out of a non zero pair
         */
        int indexProcessing = -1;
        while(true)
        {
            
            while(charBuffer.length()-position <= 1)//make sure we have chars to handle
            {
                //move everything to the front
                charBuffer.delete(0, position);
                position = 0;
                
                int read = reader.read(buffer);
                if(read < 0)
                    break;
                charBuffer.append(buffer, 0, read);
            }
            
            if(charBuffer.length()-position == 0)//EOF, no more chars
            {
                if(state == STATE.LABEL)//last line was empty
                {
                    double label = Double.parseDouble(processBuffer.toString());

                    if (!possibleCats.containsKey(label) && classification)
                        possibleCats.put(label, possibleCats.size());
                    labelVals.add(label);
                    
                    sparceVecs.add(new SparseVector(maxLen, 0));
                }
                else if(state == STATE.WHITESPACE_AFTER_LABEL)//last line was empty, but we have already eaten the label
                {
                    sparceVecs.add(new SparseVector(maxLen, 0));
                }
                else if(state == STATE.FEATURE_VALUE || state == STATE.WHITESPACE_AFTER_FEATURE)//line ended after a value pair
                {
                    //process the last value pair & insert into vec
                    double value = StringUtils.parseDouble(processBuffer, 0, processBuffer.length());
                    processBuffer.delete(0, processBuffer.length());

                    maxLen = Math.max(maxLen, indexProcessing+1);
                    tempVec.setLength(maxLen);
                    if (value != 0)
                        tempVec.set(indexProcessing, value);
                    sparceVecs.add(tempVec.clone());
                }
                else if(state == STATE.NEWLINE)
                {
                    //nothing to do and everything already processed, just return
                    break;
                }
                else
                    throw new RuntimeException();
                //we may have ended on a line, and have a sparse vec to add before returning
                break;
            }
            
            char ch = charBuffer.charAt(position);
            switch(state)
            {
                case INITIAL:
                    state = STATE.LABEL;
                    break;
                case LABEL:
                    if (Character.isDigit(ch)  || ch == '.' || ch == 'E' || ch == 'e' || ch == '-' || ch == '+')
                    {
                        processBuffer.append(ch);
                        position++;
                    }
                    else if (Character.isWhitespace(ch))//this gets spaces and new lines
                    {
                        double label = Double.parseDouble(processBuffer.toString());

                        if (!possibleCats.containsKey(label) && classification)
                            possibleCats.put(label, possibleCats.size());
                        labelVals.add(label);

                        //clean up and move to new state
                        processBuffer.delete(0, processBuffer.length());
                        
                        if (ch == '\n' || ch == '\r')//empty line, so add a zero vector
                        {
                            tempVec.zeroOut();
                            sparceVecs.add(new SparseVector(maxLen, 0));
                            state = STATE.NEWLINE;
                        }
                        else//just white space
                        {
                            tempVec.zeroOut();
                            state = STATE.WHITESPACE_AFTER_LABEL;
                        }
                    }
                    else
                        throw new RuntimeException("Invalid LIBSVM file");
                    break;
                case WHITESPACE_AFTER_LABEL:
                    if (Character.isDigit(ch))//move to next state
                    {
                        state = STATE.FEATURE_INDEX;
                    }
                    else if (Character.isWhitespace(ch))
                    {
                        if (ch == '\n' || ch == '\r')
                        {
                            tempVec.zeroOut();
                            sparceVecs.add(new SparseVector(maxLen, 0));///no features again, add zero vec
                            state = STATE.NEWLINE;
                        }
                        else//normal whie space
                            position++;
                    }
                    else
                        throw new RuntimeException();
                    break;
                case FEATURE_INDEX:
                    if (Character.isDigit(ch))
                    {
                        processBuffer.append(ch);
                        position++;
                    }
                    else if(ch == ':')
                    {
                        indexProcessing = StringUtils.parseInt(processBuffer, 0, processBuffer.length())-1;
                        processBuffer.delete(0, processBuffer.length());
                        
                        
                        state = STATE.FEATURE_VALUE;
                        position++;
                    }
                    else
                        throw new RuntimeException();
                    break;
                case FEATURE_VALUE:
                    //we need to accept all the values that may be part of a float value
                    if (Character.isDigit(ch) || ch == '.' || ch == 'E' || ch == 'e' || ch == '-' || ch == '+')
                    {
                        processBuffer.append(ch);
                        position++;
                    }
                    else
                    {
                        double value = StringUtils.parseDouble(processBuffer, 0, processBuffer.length());
                        processBuffer.delete(0, processBuffer.length());
   
                        maxLen = Math.max(maxLen, indexProcessing+1);
                        tempVec.setLength(maxLen);
                        if (value != 0)
                            tempVec.set(indexProcessing, value);
                        
                        if (Character.isWhitespace(ch))
                            state = STATE.WHITESPACE_AFTER_FEATURE;
                        else
                            throw new RuntimeException();
                    }
                    
                    break;
                case WHITESPACE_AFTER_FEATURE:
                    if (Character.isDigit(ch))
                        state = STATE.FEATURE_INDEX;
                    else if (Character.isWhitespace(ch))
                    {
                        if (ch == '\n' || ch == '\r')
                        {
                            sparceVecs.add(tempVec.clone());
                            tempVec.zeroOut();
                            state = STATE.NEWLINE;
                        }
                        else
                            position++;
                    }
                    break;
                case NEWLINE:
                    if (ch == '\n' || ch == '\r')
                        position++;
                    else
                    {
                        state = STATE.LABEL;
                    }
                    break;
            }
        }
        
        if (vectorLength > 0)
            if (maxLen > vectorLength)
                throw new RuntimeException("Length given was " + vectorLength + ", but observed length was " + maxLen);
            else
                maxLen = vectorLength;

        if(classification)
        {
            CategoricalData predicting = new CategoricalData(possibleCats.size());

            //Give categories a unique ordering to avoid loading issues based on the order categories are presented
            List<Double> allCatKeys = new DoubleList(possibleCats.keySet());
            Collections.sort(allCatKeys);
            for(int i = 0; i < allCatKeys.size(); i++)
                possibleCats.put(allCatKeys.get(i), i);

            ClassificationDataSet cds = new ClassificationDataSet(maxLen, new CategoricalData[0], predicting);
            for(int i = 0; i < labelVals.size(); i++)
            {
                SparseVector vec = sparceVecs.get(i);
                vec.setLength(maxLen);
                cds.addDataPoint(vec, new int[0], possibleCats.get(labelVals.get(i)));
            }

            cds.applyTransform(new DenseSparceTransform(sparseRatio));

            return cds;
        }
        else//regression
        {
            RegressionDataSet rds = new RegressionDataSet(maxLen, new CategoricalData[0]);
            for(int i = 0; i < sparceVecs.size(); i++)
            {
                SparseVector sv = sparceVecs.get(i);
                sv.setLength(maxLen);
                rds.addDataPoint(sv, new int[0], labelVals.get(i));
            }

            rds.applyTransform(new DenseSparceTransform(sparseRatio));

            return rds;
        }
    }
    
    /**
     * Writes out the given classification data set as a LIBSVM data file
     * @param data the data set to write to a file
     * @param os the output stream to write to. The stream will not be closed or
     * flushed by this method
     */
    public static void write(ClassificationDataSet data, OutputStream os)
    {
        PrintWriter writer = new PrintWriter(os);
        for(int i = 0; i < data.getSampleSize(); i++)
        {
            int pred = data.getDataPointCategory(i);
            Vec vals = data.getDataPoint(i).getNumericalValues();
            writer.write(pred + " ");
            for(IndexValue iv : vals)
            {
                double val = iv.getValue();
                if(Math.rint(val) == val)//cast to long before writting to save space
                    writer.write((iv.getIndex()+1) + ":" + (long)val + " ");//+1 b/c 1 based indexing
                else
                    writer.write((iv.getIndex()+1) + ":" + val + " ");//+1 b/c 1 based indexing
            }
            writer.write("\n");
        }
        writer.flush();
        writer.close();
    }
    
    /**
     * Writes out the given regression data set as a LIBSVM data file
     * @param data the data set to write to a file
     * @param os the output stream to write to. The stream will not be closed or
     * flushed by this method
     */
    public static void write(RegressionDataSet data, OutputStream os)
    {
        PrintWriter writer = new PrintWriter(os);
        for(int i = 0; i < data.getSampleSize(); i++)
        {
            double pred = data.getTargetValue(i);
            Vec vals = data.getDataPoint(i).getNumericalValues();
            writer.write(pred + " ");
            for(IndexValue iv : vals)
            {
                double val = iv.getValue();
                if(Math.rint(val) == val)//cast to long before writting to save space
                    writer.write((iv.getIndex()+1) + ":" + (long)val + " ");//+1 b/c 1 based indexing
                else
                    writer.write((iv.getIndex()+1) + ":" + val + " ");//+1 b/c 1 based indexing
            }
            writer.write("\n");
        }
        writer.flush();
        writer.close();
    }
    
    /**
     * Simple state machine used to parse LIBSVM files
     */
    private enum STATE
    {
        /**
         * Initial state, doesn't actually do anything
         */
        INITIAL,
        LABEL,
        WHITESPACE_AFTER_LABEL,
        FEATURE_INDEX,
        FEATURE_VALUE,
        WHITESPACE_AFTER_FEATURE,
        NEWLINE,
    }
    
}
