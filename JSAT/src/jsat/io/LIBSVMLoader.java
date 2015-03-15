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
        StringBuilder builder = new StringBuilder(1024);
        char[] buffer = new char[1024];
        List<SparseVector> sparseVecs = new ArrayList<SparseVector>();
        List<Double> targets = new DoubleList();
        int maxLen=1;
        
        
        int charsRead;
        int pos = 0;
        while(true)
        {
            //skip new lines and fill buffer
            while(true)
            {
                if(pos < builder.length() && (builder.charAt(pos) == '\n' || builder.charAt(pos) == '\r'))
                    pos++;
                if(pos >= builder.length())
                {
                    charsRead = reader.read(buffer);
                    if(charsRead >= 0)
                        builder.append(buffer, 0, charsRead);
                    else
                        break;
                }
                else
                    break;
            }
            if(pos == builder.length())//end of the file
                break;
            //now pos should be at the begining of a line, which should start with a key
            int spaceLoc = findCharOrEOL(builder, buffer, reader, ' ', pos);
            
            //we now have the key
            
            double target = Double.parseDouble(builder.subSequence(pos, spaceLoc).toString());
            pos = spaceLoc+1;//move to next char
            getMoreChars(pos, builder, reader, buffer);//incase we hit the end
            
            targets.add(target);
            if(builder.charAt(pos) == '\n' || builder.charAt(pos) == '\r')//new line, we had a line that was empty
            {
                //add the zero vector 
                sparseVecs.add(new SparseVector(0, 0));
                pos++;//now on the new line
                builder.delete(0, pos);
                pos = 0;
                continue;
            }
            //else, sart parsing the non zero values
            SparseVector sv = tempSparseVecs.get();
            sv.zeroOut();
            
            while(builder.charAt(pos) != '\n' && builder.charAt(pos) != '\r')//keep going till we hit EOL
            {
                int colonPos = findCharOrEOL(builder, buffer, reader, ':', pos);
                int index = StringUtils.parseInt(builder, pos, colonPos)-1;
                pos = colonPos+1;//should now be the start of a float
                int endPos = findCharOrEOL(builder, buffer, reader, ' ', pos);
                double value;
                if(endPos < 0)//we hit EOF, so assume the rest is our float
                {
                    if(fastLoad)
                        value = StringUtils.parseDouble(builder, pos, builder.length());
                    else
                        value = Double.parseDouble(builder.subSequence(pos, builder.length()).toString());
                }
                else
                {
                    if(fastLoad)
                        value = StringUtils.parseDouble(builder, pos, endPos);
                    else
                        value = Double.parseDouble(builder.subSequence(pos, endPos).toString());
                }

                //set and adjust
                maxLen = Math.max(maxLen, index+1);
                sv.setLength(maxLen);
                sv.set(index, value);
                //move and adjust buffer
                pos = endPos+1;
                getMoreChars(pos, builder, reader, buffer);
                if(pos == builder.length())//we are EOF
                {
                    if (pos > 0)
                        builder.delete(0, pos - 1);
                    break;
                }
                builder.delete(0, pos);
                pos = 0;
                
            }
            
            sparseVecs.add(sv.clone());
        }
        
        RegressionDataSet rds = new RegressionDataSet(maxLen, new CategoricalData[0]);
        for(int i = 0; i < sparseVecs.size(); i++)
        {
            SparseVector sv = sparseVecs.get(i);
            sv.setLength(maxLen);
            rds.addDataPoint(sv, new int[0], targets.get(i));
        }
        
        rds.applyTransform(new DenseSparceTransform(sparseRatio));
        
        return rds;
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
        StringBuilder builder = new StringBuilder(1024);
        char[] buffer = new char[1024];
        List<SparseVector> sparceVecs = new ArrayList<SparseVector>();
        List<Double> cats = new ArrayList<Double>();
        Map<Double, Integer> possibleCats = new HashMap<Double, Integer>();
        int maxLen=1;
        
        int charsRead;
        int pos = 0;
        while(true)
        {
            //skip new lines and fill buffer
            while(true)
            {
                if(pos < builder.length() && (builder.charAt(pos) == '\n' || builder.charAt(pos) == '\r'))
                    pos++;
                if(pos >= builder.length())
                {
                    charsRead = reader.read(buffer);
                    if(charsRead >= 0)
                        builder.append(buffer, 0, charsRead);
                    else
                        break;
                }
                else
                    break;
            }
            if(pos >= builder.length())//end of the file
                break;
            //now pos should be at the begining of a line, which should start with a key
            int spaceLoc = findCharOrEOL(builder, buffer, reader, ' ', pos);
            
            //we now have the key
            
            double cat = Double.parseDouble(builder.subSequence(pos, spaceLoc).toString());
            pos = spaceLoc+1;//move to next char
            getMoreChars(pos, builder, reader, buffer);//incase we hit the end
            if(!possibleCats.containsKey(cat))
                possibleCats.put(cat, possibleCats.size());
            cats.add(cat);
            if(builder.charAt(pos) == '\n' || builder.charAt(pos) == '\r')//new line, we had a line that was empty
            {
                //add the zero vector 
                sparceVecs.add(new SparseVector(0, 0));
                pos++;//now on the new line
                builder.delete(0, pos);
                pos = 0;
                continue;
            }
            //else, sart parsing the non zero values
            SparseVector sv = tempSparseVecs.get();
            sv.zeroOut();
            
            while(builder.charAt(pos) != '\n' && builder.charAt(pos) != '\r')//keep going till we hit EOL
            {
                int colonPos = findCharOrEOL(builder, buffer, reader, ':', pos);
                int index = StringUtils.parseInt(builder, pos, colonPos)-1;
                pos = colonPos+1;//should now be the start of a float
                int endPos = findCharOrEOL(builder, buffer, reader, ' ', pos);
                double value;
                if(endPos < 0)//we hit EOF, so assume the rest is our float
                {
                    if(fastLoad)
                        value = StringUtils.parseDouble(builder, pos, builder.length());
                    else
                        value = Double.parseDouble(builder.subSequence(pos, builder.length()).toString());
                }
                else
                {
                    if(fastLoad)
                        value = StringUtils.parseDouble(builder, pos, endPos);
                    else
                        value = Double.parseDouble(builder.subSequence(pos, endPos).toString());
                }

                //set and adjust
                maxLen = Math.max(maxLen, index+1);
                sv.setLength(maxLen);
                sv.set(index, value);
                //move and adjust buffer
                pos = endPos+1;
                getMoreChars(pos, builder, reader, buffer);
                if(pos == builder.length())//we are EOF
                {
                    if (pos > 0)
                        builder.delete(0, pos - 1);
                    break;
                }
                builder.delete(0, pos);
                pos = 0;
                
            }
            
            sparceVecs.add(sv.clone());
        }
        
        CategoricalData predicting = new CategoricalData(possibleCats.size());
        
        if(vectorLength > 0)
            maxLen = vectorLength;
        
        //Give categories a unique ordering to avoid loading issues based on the order categories are presented
        List<Double> allCatKeys = new DoubleList(possibleCats.keySet());
        Collections.sort(allCatKeys);
        for(int i = 0; i < allCatKeys.size(); i++)
            possibleCats.put(allCatKeys.get(i), i);
        
        ClassificationDataSet cds = new ClassificationDataSet(maxLen, new CategoricalData[0], predicting);
        for(int i = 0; i < cats.size(); i++)
        {
            SparseVector vec = sparceVecs.get(i);
            vec.setLength(maxLen);
            cds.addDataPoint(vec, new int[0], possibleCats.get(cats.get(i)));
        }
        
        cds.applyTransform(new DenseSparceTransform(sparseRatio));
        
        return cds;
    }

    /**
     * Loads more characters into the builder if needed
     * @param pos the current position in the builder
     * @param builder the builder used as the current window
     * @param isr the reader to get characters from
     * @param buffer the buffer to load characters into that are then copied into the builder
     * @throws IOException 
     */
    protected static void getMoreChars(int pos, StringBuilder builder, Reader isr, char[] buffer) throws IOException
    {
        int charsRead;
        while(pos >= builder.length())
        {
            charsRead = isr.read(buffer);
            if (charsRead >= 0)
                builder.append(buffer, 0, charsRead);
            else
                break;
        }
    }

    /**
     * Finds the first occurrence of the given character or a new line is encountered
     * @param builder the builder used as the current window
     * @param buffer the buffer to load characters into that are then copied into the builder
     * @param isr the reader to get characters from
     * @param findMe the character to find
     * @param start the position in the builder to start the search from
     * @return
     * @throws IOException 
     */
    protected static int findCharOrEOL(StringBuilder builder, char[] buffer, Reader isr, char findMe, int start) throws IOException
    {
        //first find the key
        int pos = start;
        int bytesRead;
        while(true)
        {
            
            if (builder.length() <= pos)
                if ((bytesRead = isr.read(buffer)) < 0)
                    return -1;
                else if (bytesRead == 0)
                    continue;//try again, we need to read something before we can continue
                else
                {
                    builder.append(buffer, 0, bytesRead);
                }
            //we don't inc pos in the while loop b/c if we have to continue on a read of zero bytes we will skip when we shouldn't
            if(builder.charAt(pos) != findMe && builder.charAt(pos) != '\n' && builder.charAt(pos) != '\r')
                pos++;
            else
                break;
        }
        return pos;
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
                writer.write((iv.getIndex()+1) + ":" + iv.getValue() + " ");//+1 b/c 1 based indexing
            writer.write("\n");
        }
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
                writer.write((iv.getIndex()+1) + ":" + iv.getValue() + " ");//+1 b/c 1 based indexing
            writer.write("\n");
        }
    }
    
    /**
     * Use thread local of sparse vectors to initialize construction. This way 
     * we avoid unnecessary object allocation - one base vec will increase to 
     * the needed size. Then a copy with only the needed space is added instead
     * of the thread local vector. 
     */
    private static final ThreadLocal<SparseVector> tempSparseVecs = new ThreadLocal<SparseVector>()
    {

        @Override
        protected SparseVector initialValue()
        {
            return new SparseVector(1);
        }
        
    };
    
}
