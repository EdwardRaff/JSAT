package jsat.text;

import java.util.List;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.text.tokenizer.Tokenizer;
import jsat.text.wordweighting.WordWeighting;
import jsat.utils.IntList;

/**
 * This class provides a framework for loading classification datasets made of 
 * text documents as vectors. This extension uses 
 * {@link #addOriginalDocument(java.lang.String, int) } instead so that the 
 * original documents have a class label associated with them. 
 * {@link #getDataSet() } then returns a classification data set, where the 
 * class label for each data point is the label provided when 
 * <tt>addOriginalDocument</tt> was called. 
 * <br>
 * New vectors created with {@link #newText(java.lang.String) } are inherently 
 * not part of the original data set, so do not need or receive a class label.
 * 
 * @author Edward Raff
 */
public abstract class ClassificationTextDataLoader extends TextDataLoader
{
    /**
     * The list of the true class labels for the data that was loaded before 
     * {@link #finishAdding() } was called. 
     */
    protected List<Integer> classLabels;
    /**
     * The information about the class label that would be predicted for a 
     * classification data set.
     */
    protected CategoricalData labelInfo;
    
    
    public ClassificationTextDataLoader(Tokenizer tokenizer, WordWeighting weighting)
    {
        super(tokenizer, weighting);
        classLabels = new IntList();
    }
    
    /**
     * The classification label data stored in {@link #labelInfo} must be set 
     * if the text loader is to return a classification data set. As such, this 
     * abstract class exists to force the user to set it, in this way they can 
     * not forget. <br>
     * This will be called in {@link #getDataSet() } just before 
     * {@link #initialLoad() } is called. 
     */
    protected abstract void setLabelInfo();

    /**
     * Should use {@link #addOriginalDocument(java.lang.String, int) } instead. 
     * @param text the text of the data to add
     */
    @Override
    protected void addOriginalDocument(String text)
    {
        throw new UnsupportedOperationException("addOriginalDocument(String"
                + " text, int label) should be used instead");
    }
    
    /**
     * To be called by the {@link #initialLoad() } method. 
     * It will take in the text and add a new document 
     * vector to the data set. Once all text documents 
     * have been loaded, this method should never be 
     * called again. 
     * 
     * @param text the text of the document to add
     * @param label the classification label for this document
     */
    protected void addOriginalDocument(String text, int label)
    {
        if(label >= labelInfo.getNumOfCategories())
            throw new RuntimeException("Invalid label given");
        super.addOriginalDocument(text);
        classLabels.add(label);
    }
    
    @Override
    public ClassificationDataSet getDataSet()
    {
        if(!noMoreAdding)
        {
            setLabelInfo();
            initialLoad();
            finishAdding();
        }
        
        ClassificationDataSet cds = 
                new ClassificationDataSet(vectors.get(0).length(), 
                new CategoricalData[]{}, labelInfo);
        for(int i = 0; i < vectors.size(); i++)
            cds.addDataPoint(vectors.get(i), new int[]{}, classLabels.get(i));
        
        return cds;
    }
}
