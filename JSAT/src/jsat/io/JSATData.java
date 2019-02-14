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
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import jsat.*;
import jsat.classifiers.*;
import jsat.linear.*;
import jsat.regression.RegressionDataSet;
import jsat.utils.DoubleList;
import jsat.utils.IntList;

/**
 * JSAT Data Loader provides a simple binary file format for storing and reading
 * datasets. All datasets can always be read back in as a {@link SimpleDataSet},
 * and {@link ClassificationDataSet} and {@link RegressionDataSet} datasets can
 * be read back in as their original types.<br>
 * <br>
 * For well behaved datasets (where most numeric features are integer values),
 * an uncompressed JSAT Dataset file may be larger than a similar
 * {@link ARFFLoader ARFF} or {@link LIBSVMLoader LIBSVM} file. This is because
 * JSAT always uses 32 or 64 bits (4 or 8 bytes) for every value, where values
 * stored as a string could use as little as 1 or 2 bytes for simple values.
 * However, JSAT's storage will be consistent - data which uses the floating
 * point values (such as the number "0.25098039215686274") will use additional
 * bytes in the human readable ARFF and LIBSVM formats that are not necessary,
 * where the binary JSAT format will stay the same size.<br>
 * <br>
 * Even when JSAT produces larger files, since it is a simple binary format,
 * reading and writing will usually be significantly faster.
 * <br>
 * Additional storage savings can be obtained by using the
 * {@link GZIPOutputStream} when storing a dataset, and then decompressed when
 * read back in using {@link GZIPInputStream}.
 * <br>
 * 
 * 
 *
 * @author Edward Raff
 */
public class JSATData
{

    private JSATData()
    {
    }
    
    
    public static final byte[] MAGIC_NUMBER = new byte[]
    {
        'J', 'S', 'A', 'T', '_', '0', '0'
    };
    public static enum DatasetTypeMarker
    {
        STANDARD,
        REGRESSION,
        CLASSIFICATION;
    }
    public static enum FloatStorageMethod
    {
        AUTO 
        {
            @Override
            protected void writeFP(double value, DataOutputStream out) throws IOException
            {
                //Auto dosn't actually write! Auto just means figure out the best
                throw new UnsupportedOperationException("Not supported ."); 
            }
            
            @Override
            protected double readFP(DataInputStream in) throws IOException
            {
                //Auto dosn't actually write! Auto just means figure out the best
                throw new UnsupportedOperationException("Not supported ."); 
            }
            
            @Override
            protected boolean noLoss(double orig) 
            {
                return true;
            }
            
        },
        FP64 
        {
            @Override
            protected void writeFP(double value, DataOutputStream out) throws IOException
            {
                out.writeDouble(value);
            }
            
            @Override
            protected double readFP(DataInputStream in) throws IOException
            {
                return in.readDouble();
            }
            
            @Override
            protected boolean noLoss(double orig) 
            {
                return true;
            }
        },
        FP32 
        {
            @Override
            protected void writeFP(double value, DataOutputStream out) throws IOException
            {
                out.writeFloat((float) value);
            }
            
            @Override
            protected double readFP(DataInputStream in) throws IOException
            {
                return in.readFloat();
            }
            
            @Override
            protected boolean noLoss(double orig) 
            {
                //below can only return true if ther eis no loss in storing these values as 32 bit floats instead of doubles
                float f_o = (float) orig;
                return Double.valueOf(f_o)-orig == 0.0;
            }
        },
        SHORT 
        {
            @Override
            protected void writeFP(double value, DataOutputStream out) throws IOException
            {
                out.writeShort(Math.min(Math.max((int)value, Short.MIN_VALUE), Short.MAX_VALUE));
            }
            
            @Override
            protected double readFP(DataInputStream in) throws IOException
            {
                return in.readShort();
            }
            
            @Override
            protected boolean noLoss(double orig) 
            {
                return Short.MIN_VALUE <= orig && orig <= Short.MAX_VALUE && orig == Math.rint(orig);
            }
        },
        BYTE 
        {
            @Override
            protected void writeFP(double value, DataOutputStream out) throws IOException
            {
                out.writeByte(Math.min(Math.max((int)value, Byte.MIN_VALUE), Byte.MAX_VALUE));
            }
            
            @Override
            protected double readFP(DataInputStream in) throws IOException
            {
                return in.readByte();
            }
            
            @Override
            protected boolean noLoss(double orig) 
            {
                return Byte.MIN_VALUE <= orig && orig <= Byte.MAX_VALUE && orig == Math.rint(orig);
            }
        },
        U_BYTE
        {
            @Override
            protected void writeFP(double value, DataOutputStream out) throws IOException
            {
                out.writeByte(Math.min(Math.max((int)value, 0), 255));
            }
            
            @Override
            protected double readFP(DataInputStream in) throws IOException
            {
                return in.readByte() & 0xff;
            }
            
            @Override
            protected boolean noLoss(double orig) 
            {
                return 0 <= orig && orig <= 255 && orig == Math.rint(orig);
            }
        };
        
        abstract protected void writeFP(double value, DataOutputStream out) throws IOException;
        
        abstract protected double readFP(DataInputStream in) throws IOException;
        
        abstract protected boolean noLoss(double orig);
        
        static public <Type extends DataSet<Type>> FloatStorageMethod getMethod(DataSet<Type> data, FloatStorageMethod method)
        {
            if (method == FloatStorageMethod.AUTO)//figure out what storage method to use! 
            {
                EnumSet<FloatStorageMethod> storageCandidates = EnumSet.complementOf(EnumSet.of(FloatStorageMethod.AUTO));

                //loop through all the data and remove invalid candidates
                for(int i = 0; i < data.size(); i++)
                {
                    DataPoint dp = data.getDataPoint(i);
                    for (IndexValue iv : dp.getNumericalValues())
                    {
                        Iterator<FloatStorageMethod> iter = storageCandidates.iterator();
                        while (iter.hasNext())
                        {
                            if (!iter.next().noLoss(iv.getValue()))
                                iter.remove();
                        }
                        if (storageCandidates.size() == 1)
                            break;
                    }

                    Iterator<FloatStorageMethod> iter = storageCandidates.iterator();
                    while (iter.hasNext())
                    {
                        if (!iter.next().noLoss(data.getWeight(i)))
                            iter.remove();
                    }
                    if (storageCandidates.size() == 1)
                        break;
                }
                
                if(data instanceof RegressionDataSet)
                {
                    for(IndexValue iv : ((RegressionDataSet)data).getTargetValues())
                    {
                        Iterator<FloatStorageMethod> iter = storageCandidates.iterator();
                        while (iter.hasNext())
                        {
                            if (!iter.next().noLoss(iv.getValue()))
                                iter.remove();
                        }
                        if (storageCandidates.size() == 1)
                            break;
                    }
                }
                
                if(storageCandidates.contains(BYTE))
                    return BYTE;
                else if(storageCandidates.contains(U_BYTE))
                    return U_BYTE;
                else if(storageCandidates.contains(SHORT))
                    return SHORT;
                else if(storageCandidates.contains(FP32))
                    return FP32;
                return FP64;
                
                
            }
            else
                return method;
        }
    }
    
    public static final byte STRING_ENCODING_ASCII = 0;
    public static final byte STRING_ENCODING_UTF_16 = 1;
    
    /**
     * This method writes out a JSAT dataset to a binary format that can be read
     * in again later, and could be read in other languages.<br>
     * <br>
     * The format that is used will understand both
     * {@link ClassificationDataSet} and {@link RegressionDataSet} datasets as
     * special cases, and will store the target values in the binary file. When
     * read back in, they can be returned as their original dataset type, or
     * treated as normal fields as a {@link SimpleDataSet}.<br>
     * The storage format chosen for floating point values will chose a method
     * that results in no loss of precision when reading the data back in.
     *
     * @param <Type>
     * @param dataset the dataset to write out to a binary file
     * @param outRaw the raw output stream, the caller should provide a buffered
     * stream.
     * @throws IOException
     */
    public static <Type extends DataSet<Type>> void writeData(DataSet<Type> dataset, OutputStream outRaw) throws IOException
    {
        writeData(dataset, outRaw, FloatStorageMethod.AUTO);
    }
    
    /**
     * This method writes out a JSAT dataset to a binary format that can be read
     * in again later, and could be read in other languages.<br>
     * <br>
     * The format that is used will understand both
     * {@link ClassificationDataSet} and {@link RegressionDataSet} datasets as
     * special cases, and will store the target values in the binary file. When
     * read back in, they can be returned as their original dataset type, or
     * treated as normal fields as a {@link SimpleDataSet}.
     *
     * @param <Type>
     * @param dataset the dataset to write out to a binary file
     * @param outRaw the raw output stream, the caller should provide a buffered
     * stream.
     * @param fpStore the storage method of storing floating point values, which
     * may result in a loss of precision depending on the method chosen.
     * @throws IOException
     */
    public static <Type extends DataSet<Type>> void writeData(DataSet<Type> dataset, OutputStream outRaw, FloatStorageMethod fpStore) throws IOException
    {
        fpStore = FloatStorageMethod.getMethod(dataset, fpStore);
        
        DataWriter.DataSetType type;
        CategoricalData predicting;
        if(dataset instanceof ClassificationDataSet)
        {
            type = DataWriter.DataSetType.CLASSIFICATION;
            predicting = ((ClassificationDataSet)dataset).getPredicting();
        }
        else if(dataset instanceof RegressionDataSet)
        {
            type = DataWriter.DataSetType.REGRESSION;
            predicting = null;
        }
        else
        {
            type = DataWriter.DataSetType.SIMPLE;
            predicting = null;
        }
        
        DataWriter dw = getWriter(outRaw, dataset.getCategories(), dataset.getNumNumericalVars(), predicting, fpStore, type);
        
        //write out all the datapoints
        for(int i = 0; i < dataset.size(); i++)
        {
            double label = 0;
            if (dataset instanceof ClassificationDataSet)
                label = ((ClassificationDataSet) dataset).getDataPointCategory(i);
            else if (dataset instanceof RegressionDataSet)
                label = ((RegressionDataSet) dataset).getTargetValue(i);
            dw.writePoint(dataset.getWeight(i), dataset.getDataPoint(i), label);
        }
        
        dw.finish();
        outRaw.flush();
    }
    
    /**
     * Returns a DataWriter object which can be used to stream a set of arbitrary datapoints into the given output stream. This works in a thread safe manner. 
     * 
     * @param out the location to store all the data
     * @param catInfo information about the categorical features to be written
     * @param dim information on how many numeric features exist
     * @param predicting information on the class label, may be {@code null} if not a classification dataset
     * @param fpStore the format floating point values should be stored as
     * @param type what type of data set (simple, classification, regression) to be written
     * @return the DataWriter that the actual points can be streamed through
     * @throws IOException 
     */
    public static DataWriter getWriter(OutputStream out, CategoricalData[] catInfo, int dim, final CategoricalData predicting, final FloatStorageMethod fpStore, DataWriter.DataSetType type) throws IOException
    {
        return new DataWriter(out, catInfo, dim, type)
        {
            @Override
            protected void writeHeader(CategoricalData[] catInfo, int dim, DataWriter.DataSetType type, OutputStream out) throws IOException
            {
                DataOutputStream data_out = new DataOutputStream(out);
        
                data_out.write(JSATData.MAGIC_NUMBER);

                int numNumeric = dim;
                int numCat = catInfo.length;

                DatasetTypeMarker marker = DatasetTypeMarker.STANDARD;
                if(type == type.REGRESSION)
                {
                    numNumeric++;
                    marker = DatasetTypeMarker.REGRESSION;
                }
                if(type == type.CLASSIFICATION)
                {
                    numCat++;
                    marker = DatasetTypeMarker.CLASSIFICATION;
                }

                data_out.writeByte(marker.ordinal());
                data_out.writeByte(fpStore.ordinal());
                data_out.writeInt(numNumeric);
                data_out.writeInt(numCat);
                data_out.writeInt(-1);//-1 used to indicate a potentially variable number of files

                for(CategoricalData category : catInfo)
                {
                    //first, whats the name of the i'th category
                    writeString(category.getCategoryName(), data_out);

                    data_out.writeInt(category.getNumOfCategories());//output the number of categories 
                    for(int i = 0; i < category.getNumOfCategories(); i++)//the option names
                        writeString(category.getOptionName(i), data_out);
                }
                //extra for classification dataset
                if(type == DataWriter.DataSetType.CLASSIFICATION)
                {
                    CategoricalData category = predicting;
                    //first, whats the name of the i'th category
                    writeString(category.getCategoryName(), data_out);

                    data_out.writeInt(category.getNumOfCategories());//output the number of categories 
                    for(int i = 0; i < category.getNumOfCategories(); i++)//the option names
                        writeString(category.getOptionName(i), data_out);
                }
                data_out.flush();
            }
            
            @Override
            protected void pointToBytes(double weight, DataPoint dp, double label, ByteArrayOutputStream byteOut)
            {
                try
                {
                    DataOutputStream data_out = new DataOutputStream(byteOut);
                    fpStore.writeFP(weight, data_out);
                    for(int val : dp.getCategoricalValues())
                        data_out.writeInt(val);
                    if(type == DataWriter.DataSetType.CLASSIFICATION)
                        data_out.writeInt((int) label);
                    
                    Vec numericVals = dp.getNumericalValues();
                    
                    data_out.writeBoolean(numericVals.isSparse());
                    if(numericVals.isSparse())
                    {
                        if(type == DataWriter.DataSetType.REGRESSION)
                            data_out.writeInt(numericVals.nnz()+1);//+1 for the target value, which may actually be zero...
                        else
                            data_out.writeInt(numericVals.nnz());
                        
                        for(IndexValue iv : numericVals)
                        {
                            data_out.writeInt(iv.getIndex());
                            fpStore.writeFP(iv.getValue(), data_out);
                        }
                    }
                    else
                    {
                        for(int j = 0; j < numericVals.length(); j++)
                            fpStore.writeFP(numericVals.get(j), data_out);
                    }
                    
                    //append the target value
                    if(type == DataWriter.DataSetType.REGRESSION)
                    {
                        /*
                        * if dense, we only need to just add the extra double. If
                        * sparse, we do the index and then the double.
                        */
                        if (numericVals.isSparse())
                            data_out.writeInt(numericVals.length());
                        
                        fpStore.writeFP(label, data_out);
                    }
                    data_out.flush();
                }
                catch (IOException ex)
                {
                    Logger.getLogger(JSATData.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        };
    }
    
    /**
     * This loads a JSAT dataset from an input stream, and will not do any of
     * its own buffering. The DataSet will be returned as either a
     * {@link SimpleDataSet}, {@link ClassificationDataSet}, or
     * {@link RegressionDataSet} depending on what type of dataset was
     * originally written out.<br>
     *
     * @param inRaw the input stream, caller should buffer it
     * @return a dataset
     * @throws IOException 
     */
    public static DataSet<?> load(InputStream inRaw) throws IOException
    {
        return load(inRaw, DataStore.DEFAULT_STORE.emptyClone());
    }
    
    /**
     * This loads a JSAT dataset from an input stream, and will not do any of
     * its own buffering. The DataSet will be returned as either a
     * {@link SimpleDataSet}, {@link ClassificationDataSet}, or
     * {@link RegressionDataSet} depending on what type of dataset was
     * originally written out.<br>
     *
     * @param inRaw the input stream, caller should buffer it
     * @param backingStore the data store to put all datapoints in
     * @return a dataset
     * @throws IOException 
     */
    public static DataSet<?> load(InputStream inRaw, DataStore backingStore) throws IOException
    {
        return load(inRaw, false, backingStore);
    }
    
    /**
     * Loads in a JSAT dataset as a {@link SimpleDataSet}. So long as the input
     * stream is valid, this will not fail.
     *
     * @param inRaw the input stream, caller should buffer it
     * @return a SimpleDataSet object
     * @throws IOException 
     */
    public static SimpleDataSet loadSimple(InputStream inRaw) throws IOException
    {
        return loadSimple(inRaw, new RowMajorStore());
    }
    
    /**
     * Loads in a JSAT dataset as a {@link SimpleDataSet}. So long as the input
     * stream is valid, this will not fail.
     *
     * @param inRaw the input stream, caller should buffer it
     * @param backingStore the data store to put all data points in
     * @return a SimpleDataSet object
     * @throws IOException 
     */
    public static SimpleDataSet loadSimple(InputStream inRaw, DataStore backingStore) throws IOException
    {
        return (SimpleDataSet) load(inRaw, true, backingStore);
    }
    
    /**
     * Loads in a JSAT dataset as a {@link ClassificationDataSet}. An exception
     * will be thrown if the original dataset in the file was not a
     * {@link ClassificationDataSet}.
     *
     * @param inRaw the input stream, caller should buffer it
     * @return a ClassificationDataSet object
     * @throws IOException 
     * @throws ClassCastException if the original dataset was a not a ClassificationDataSet
     */
    public static ClassificationDataSet loadClassification(InputStream inRaw) throws IOException
    {
        return loadClassification(inRaw, new RowMajorStore());
    }
    
    /**
     * Loads in a JSAT dataset as a {@link ClassificationDataSet}. An exception
     * will be thrown if the original dataset in the file was not a
     * {@link ClassificationDataSet}.
     *
     * @param inRaw the input stream, caller should buffer it
     * @param backingStore the data store to put all data points in
     * @return a ClassificationDataSet object
     * @throws IOException 
     * @throws ClassCastException if the original dataset was a not a ClassificationDataSet
     */
    public static ClassificationDataSet loadClassification(InputStream inRaw, DataStore backingStore) throws IOException
    {
        return (ClassificationDataSet) load(inRaw, backingStore);
    }
    
    /**
     * Loads in a JSAT dataset as a {@link RegressionDataSet}. An exception
     * will be thrown if the original dataset in the file was not a
     * {@link RegressionDataSet}.
     *
     * @param inRaw the input stream, caller should buffer it
     * @return a RegressionDataSet object
     * @throws IOException 
     * @throws ClassCastException if the original dataset was a not a RegressionDataSet
     */
    public static RegressionDataSet loadRegression(InputStream inRaw) throws IOException
    {
        return loadRegression(inRaw, new RowMajorStore());
    }
    
    /**
     * Loads in a JSAT dataset as a {@link RegressionDataSet}. An exception
     * will be thrown if the original dataset in the file was not a
     * {@link RegressionDataSet}.
     *
     * @param inRaw the input stream, caller should buffer it
     * @param backingStore the data store to put all data points in
     * @return a RegressionDataSet object
     * @throws IOException 
     * @throws ClassCastException if the original dataset was a not a RegressionDataSet
     */
    public static RegressionDataSet loadRegression(InputStream inRaw, DataStore backingStore) throws IOException
    {
        return (RegressionDataSet) load(inRaw, backingStore);
    }
    
    /**
     * This loads a JSAT dataset from an input stream, and will not do any of
     * its own buffering. The DataSet will be returned as either a
     * {@link SimpleDataSet}, {@link ClassificationDataSet}, or
     * {@link RegressionDataSet} depending on what type of dataset was
     * originally written out.<br>
     * <br>
     * This method supports forcing the load to return a {@link SimpleDataSet}. 
     * <br>
     * This method uses a {@link RowMajorStore} store for the data. 
     *
     * @param inRaw the input stream, caller should buffer it
     * @param forceAsStandard {@code true} for for the dataset to be loaded as a
     * {@link SimpleDataSet}, otherwise it will be determined based on the input
     * streams contents.
     * @return a dataset
     * @throws IOException 
     */
    @SuppressWarnings("unchecked")
    protected static DataSet<?> load(InputStream inRaw, boolean forceAsStandard) throws IOException
    {
        return load(inRaw, forceAsStandard, DataStore.DEFAULT_STORE.emptyClone());
    }
    /**
     * This loads a JSAT dataset from an input stream, and will not do any of
     * its own buffering. The DataSet will be returned as either a
     * {@link SimpleDataSet}, {@link ClassificationDataSet}, or
     * {@link RegressionDataSet} depending on what type of dataset was
     * originally written out.<br>
     * <br>
     * This method supports forcing the load to return a {@link SimpleDataSet}. 
     *
     * @param inRaw the input stream, caller should buffer it
     * @param forceAsStandard {@code true} for for the dataset to be loaded as a
     * {@link SimpleDataSet}, otherwise it will be determined based on the input
     * streams contents.
     * @param store the backing mechanism to store all the data in and use for the returned dataset object
     * @return a dataset
     * @throws IOException 
     */
    @SuppressWarnings("unchecked")
    protected static DataSet<?> load(InputStream inRaw, boolean forceAsStandard, DataStore store) throws IOException
    {
        DataInputStream in = new DataInputStream(inRaw);
        
        byte[] magic_number = new byte[MAGIC_NUMBER.length];
        in.readFully(magic_number);
        String magic = new String(magic_number, "US-ASCII");
        
        if(!magic.startsWith("JSAT_"))
            throw new RuntimeException("data does not contain magic number");
        
        DatasetTypeMarker marker = DatasetTypeMarker.values()[in.readByte()];
        FloatStorageMethod fpStore = FloatStorageMethod.values()[in.readByte()];
        
        int numNumeric = in.readInt();
        int numCat = in.readInt();
        int N = in.readInt();
        
        if(forceAsStandard)
            marker = DatasetTypeMarker.STANDARD;
        
        if(marker == DatasetTypeMarker.CLASSIFICATION)
            numCat--;
        else if(marker == DatasetTypeMarker.REGRESSION)
            numNumeric--;
        
        CategoricalData[] categories = new CategoricalData[numCat];
        CategoricalData predicting = null;//may not be used
        
        for(int i = 0; i < categories.length; i++)
        {
            //first, whats the name of the i'th category
            String name = readString(in);
            int k = in.readInt();//output the number of categories 
            
            categories[i] = new CategoricalData(k);
            categories[i].setCategoryName(name);
            
            for(int j = 0; j < k; j++)//the option names
                categories[i].setOptionName(readString(in), j);
        }
        
        if(marker == DatasetTypeMarker.CLASSIFICATION)
        {
            //first, whats the name of the i'th category
            String name = readString(in);
            int k = in.readInt();//output the number of categories 
            
            predicting = new CategoricalData(k);
            predicting.setCategoryName(name);
            
            for(int j = 0; j < k; j++)//the option names
                predicting.setOptionName(readString(in), j);
        }
        
        //used for both numeric and categorical target storage
        DoubleList targets = new DoubleList();
        DoubleList weights = new DoubleList();
        store.setCategoricalDataInfo(categories);
        store.setNumNumeric(numNumeric);
        
        //read in all the data points
        if(N < 0)
            N = Integer.MAX_VALUE;
        try
        {
            for(int i = 0; i < N; i++)
            {
            
                double weight = fpStore.readFP(in);//in.readDouble();
                int[] catVals = new int[numCat];
                double target = 0;

                for(int j = 0; j < catVals.length; j++)
                    catVals[j] = in.readInt();

                if(marker ==  DatasetTypeMarker.CLASSIFICATION)
                {
                    //int can be stored losselessly in a double, so this is safe
                    target = in.readInt();
                }

                boolean sparse = in.readBoolean();
                Vec numericVals;


                if(sparse)
                {
                    int nnz = in.readInt();
                    if(marker == DatasetTypeMarker.REGRESSION)
                        nnz--;//don't count the target value
                    int[] indicies = new int[nnz];
                    double[] values = new double[nnz];
                    for(int j = 0; j < nnz; j++)
                    {
                        indicies[j] = in.readInt();
                        values[j] = fpStore.readFP(in);
                    }
                    numericVals = new SparseVector(indicies, values, numNumeric, nnz);
                }
                else
                {
                    numericVals = new DenseVector(numNumeric);
                    for(int j = 0; j < numNumeric; j++)
                        numericVals.set(j, fpStore.readFP(in));
                }

                //get the target value 
                if(marker == DatasetTypeMarker.REGRESSION)
                {
                    /* 
                     * if dense, we only need to just add the extra double. If 
                     * sparse, we do the index and then the double. 
                     */
                    if (numericVals.isSparse())
                        in.readInt();//don't care, its the last index value - so its the target

                    target = fpStore.readFP(in);
                }

                DataPoint dp = new DataPoint(numericVals, catVals, categories);
		weights.add(weight);
                store.addDataPoint(dp);

                switch(marker)
                {
                    case CLASSIFICATION:
                    case REGRESSION:
                        targets.add(target);
                    default:
                        break;
                }
            
            }
        }
        catch (EOFException eo)
        {
            //No problem
        }
        
        in.close();
        
	DataSet toRet;
        switch(marker)
        {
            case CLASSIFICATION:
                IntList targets_i = IntList.view(targets.stream().mapToInt(Double::intValue).toArray());
                toRet =  new ClassificationDataSet(store, targets_i, predicting);
		break;
            case REGRESSION:
                toRet =  new RegressionDataSet(store, targets);
		break;
            default:
                toRet =  new SimpleDataSet(store);
        }
	for(int i = 0; i < weights.size(); i++)
	    toRet.setWeight(i, weights.getD(i));
	return toRet;
    }
    
    private static void writeString(String s, DataOutputStream out) throws IOException
    {
        boolean isAscii = true;
        for(int i = 0; i < s.length() && isAscii; i++)
            if(s.charAt(i) >= 256 || s.charAt(i) <= 0)
                isAscii = false;

        if(isAscii)
        {
            out.writeByte(STRING_ENCODING_ASCII);
            out.writeInt(s.length());//number of bytes of the string
            for(int i = 0; i < s.length(); i++)
                out.writeByte(s.charAt(i));
        }
        else//write as UTF-8
        {
            byte[] bytes = s.getBytes("UTF-16");
            out.writeByte(STRING_ENCODING_UTF_16);
            out.writeInt(bytes.length);//number of bytes of the string
            out.write(bytes);
        }
    }
    
    private static String readString(DataInputStream in) throws IOException
    {
        StringBuilder builder = new StringBuilder();
        byte encoding = in.readByte();
        int bytesToRead = in.readInt();
        
        switch(encoding)
        {
            case STRING_ENCODING_ASCII:
                for (int i = 0; i < bytesToRead; i++)
                    builder.append(Character.toChars(in.readByte()));
                return builder.toString();
            case STRING_ENCODING_UTF_16:
                byte[] bytes = new byte[bytesToRead];
                in.readFully(bytes);
                return new String(bytes, "UTF-16");
            default:
                throw new RuntimeException("Unkown string encoding value " + encoding);
        }
        
    }
}
