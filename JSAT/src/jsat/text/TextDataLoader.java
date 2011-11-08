
package jsat.text;

import jsat.text.wordweighting.WordWeighting;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.List;
import java.util.Set;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.SparceVector;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.vectorcollection.VectorArray;
import jsat.math.Function;
import jsat.math.IndexFunction;
import jsat.text.stemming.Stemmer;
import jsat.text.tokenizer.Tokenizer;
import static java.lang.Math.*;
/**
 * This class provides a framework for loading datasets made of Text documents as vectors. 
 * 
 * 
 * 
 * @author Edward Raff 
 */
public abstract class TextDataLoader
{
    protected DistanceMetric distMetric;
    protected VectorArray<SparceVector> vectors;
    protected Tokenizer tokenizer;
    
    /**
     * Maps words to their associated index in an array
     */
    protected Map<String, Integer> wordIndex;
    protected ArrayList<Integer> termDocumentFrequencys;
    private WordWeighting weighting;
    
    
    private boolean noMoreAdding;
    private int currentLength = 0;
    private int documents;

    public TextDataLoader(DistanceMetric distMetric, Tokenizer tokenizer, WordWeighting weighting)
    {
        this.distMetric = distMetric;
        this.vectors = new VectorArray<SparceVector>(this.distMetric);
        this.tokenizer = tokenizer;
        
        this.wordIndex = new Hashtable<String, Integer>();
        this.termDocumentFrequencys = new ArrayList<Integer>();
        this.weighting = weighting;
        noMoreAdding = false;
    }
    
    /**
     * This method will initialLoad all the text documents from their source. 
     * For each document, {@link #addOriginalDocument(java.lang.String) } 
     * should be called with the text of the document. <br>
     * Once all documents have been added, {@link #finishAdding() } should be called, 
     * so that post processing steps can be applied. 
     */
    public abstract void initialLoad();
    
    /**
     * To be called by the {@link #initialLoad() } method. 
     * It will take in the text and add a new document 
     * vector to the data set. Once all text documents 
     * have been loaded, this method should never be 
     * called again. 
     * 
     * @param text the text of the document to add
     */
    protected void addOriginalDocument(String text)
    {
        if(noMoreAdding)
            throw new RuntimeException("Initial data set has been finalized");
        List<String> words = tokenizer.tokenize(text);
        
        /**
         * Words we have already seen in this document, for keeping track of document occurances
         */
        Set<String> seenWords = new HashSet<String>();
        
        SparceVector vec = new SparceVector(currentLength+1);//+1 to avoid issues when its length is zero, will be corrected in finalization step anyway
        for(String word : words)
        {
            if(!wordIndex.containsKey(word))//this word has never been seen before!
            {
                wordIndex.put(word, currentLength++);
                termDocumentFrequencys.add(1);
                vec.setLength(currentLength);
                vec.set(currentLength-1, 1.0);
            }
            else//this word has been seen before
            {
                int indx = wordIndex.get(word);
                if(!seenWords.contains(word))
                    termDocumentFrequencys.set(indx, termDocumentFrequencys.get(indx)+1);
                vec.increment(indx, 1.0);
            }
            
            seenWords.add(word);
        }
        
        vectors.add(vec);
        documents++;
    }
    
    protected void finishAdding()
    {
        noMoreAdding = true;
        
        for(SparceVector vec : vectors)
        {
            //Make sure all the vectors have the same length
            vec.setLength(currentLength);
            vec.applyIndexFunction(weighting);
        }
    }
    
    public SimpleDataSet getDataSet()
    {
        List<DataPoint> dataPoints= new ArrayList<DataPoint>(vectors.size());
        
        for(SparceVector vec : vectors)
            dataPoints.add(new DataPoint(vec, new int[0], new CategoricalData[0]));
        
        return new SimpleDataSet(dataPoints);
    }
    
    /**
     * To be called after all original texts have been loaded. 
     * 
     * @param text the text of the document to create a document vector from
     * @return the sparce vector representing this document 
     */
    public SparceVector newText(String text)
    {
        if(!noMoreAdding)
            throw new RuntimeException("Initial documents have not yet loaded");
        List<String> words = tokenizer.tokenize(text);
        
        SparceVector vec = new SparceVector(currentLength);
        
        for( String word : words)
        {
            if(wordIndex.containsKey(word))//Could also call retainAll on words before looping. Worth while to investigate 
            {
                int index = wordIndex.get(word);
                vec.increment(index, 1.0);
            }
        }
        
        vec.applyIndexFunction(weighting);
        
        return vec;
    }
}
