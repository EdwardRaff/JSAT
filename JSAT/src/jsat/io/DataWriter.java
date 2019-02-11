/*
 * Copyright (C) 2017 edraff
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

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;

/**
 * This interface defines a contract by which datapoints may be written out to a
 * dataset file (such as a CSV) in an incremental fashion. This object is thread
 * safe for all methods.
 *
 * @author Edward Raff
 */
public abstract class DataWriter implements Closeable
{
    /**
     * Use a 10MB local buffer for each thread
     */
    private static final int LOCAL_BUFFER_SIZE = 1024*1024*10;
    /**
     * The list of all local buffers, used to make sure all data makes it out when {@link #finish() } is called. 
     */
    protected List<ByteArrayOutputStream> all_buffers = Collections.synchronizedList(new ArrayList<>());
    /**
     * The destination to ultimately write the dataset to
     */
    protected final OutputStream out;
    /**
     * The type of dataset to be written out
     */
    public final DataSetType type;
    /**
     * the categorical feature information for the whole corpus
     */
    public final CategoricalData[] catInfo;
    /**
     * the number of numeric features for the whole corpus
     */
    public final int dim;
    
    public DataWriter(OutputStream out, CategoricalData[] catInfo, int dim, DataSetType type) throws IOException
    {
        this.out = out;
        this.type = type;
        this.catInfo = catInfo;
        this.dim = dim;
        writeHeader(catInfo, dim, type, out);
    }
    
    
    /**
     * The local buffers that writing will be done to. 
     */
    protected ThreadLocal<ByteArrayOutputStream> local_baos = new ThreadLocal<ByteArrayOutputStream>()
    {
        @Override
        protected ByteArrayOutputStream initialValue()
        {
            ByteArrayOutputStream baos = new ByteArrayOutputStream(LOCAL_BUFFER_SIZE);
            all_buffers.add(baos);
            return baos;
        }
    };
    
    
    abstract protected void writeHeader(CategoricalData[] catInfo, int dim, DataSetType type, OutputStream out) throws IOException;
    
    /**
     * Write out the given data point to the output stream
     * @param dp the data point to write to the file
     * @param label The associated label for this dataum. If {@link #type} is a
     * {@link DataSetType#SIMPLE} set, this value will be ignored. If
     * {@link DataSetType#CLASSIFICATION}, the value will be assumed to be an
     * integer class label.
     * @throws java.io.IOException
     */
    public void writePoint(DataPoint dp, double label) throws IOException
    {
	writePoint(1.0, dp, label);
    }
    
    /**
     * Write out the given data point to the output stream
     * @param weight weight of the given data point to write out
     * @param dp the data point to write to the file
     * @param label The associated label for this dataum. If {@link #type} is a
     * {@link DataSetType#SIMPLE} set, this value will be ignored. If
     * {@link DataSetType#CLASSIFICATION}, the value will be assumed to be an
     * integer class label.
     * @throws java.io.IOException
     */
    public void writePoint(double weight, DataPoint dp, double label) throws IOException
    {
        ByteArrayOutputStream baos = local_baos.get();
        pointToBytes(weight, dp, label, baos);
        if(baos.size() >= LOCAL_BUFFER_SIZE)//We've got a big chunk of data, lets dump it
            synchronized(out)
            {
                baos.writeTo(out);
                baos.reset();
            }
    }
    
    /**
     * This method converts a datapoint into the sequence of bytes used by the underlying file format. 
     * @param weight weight of the given data point to write out
     * @param dp the data point to be converted to set of bytes
     * @param label the label of the point to convert to the set of bytes
     * @param byteOut the location to write the bytes to.
     */
    abstract protected void pointToBytes(double weight, DataPoint dp, double label, ByteArrayOutputStream byteOut);
    
    /**
     * To be called after all threads are done calling {@link #writePoint(jsat.classifiers.DataPoint, double) }. 
     */
    public synchronized void finish() throws IOException
    {
        synchronized(out)
        {
            for(ByteArrayOutputStream baos : all_buffers)
            {
                baos.writeTo(out);
                baos.reset();
            }
            out.flush();
        }
        
    }

    @Override
    public void close() throws IOException
    {
        finish();
        out.close();
    }
    
    public static enum DataSetType
    {
        SIMPLE, CLASSIFICATION, REGRESSION
    }
}
