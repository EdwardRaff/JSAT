
package jsat.text;

import java.util.*;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.SparseVector;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.vectorcollection.VectorArray;
import jsat.text.tokenizer.Tokenizer;
import jsat.text.wordweighting.WordWeighting;
import jsat.utils.IntList;

/**
 * This class provides a framework for loading datasets made of Text documents as vectors. 
 * 
 * 
 * 
 * @author Edward Raff 
 */
public abstract class TextDataLoader
{
    protected List<SparseVector> vectors;
    protected Tokenizer tokenizer;
    
    /**
     * Maps words to their associated index in an array
     */
    protected Map<String, Integer> wordIndex;
    protected List<String> allWords;
    protected List<Integer> termDocumentFrequencys;
    private WordWeighting weighting;
    
    
    private boolean noMoreAdding;
    private int currentLength = 0;
    private int documents;

    public TextDataLoader(Tokenizer tokenizer, WordWeighting weighting)
    {
        this.vectors = new ArrayList<SparseVector>();
        this.tokenizer = tokenizer;
        
        this.wordIndex = new HashMap<String, Integer>();
        this.termDocumentFrequencys = new IntList();
        this.weighting = weighting;
        this.allWords = new ArrayList<String>();
        noMoreAdding = false;
    }
    
    /**
     * This method will load all the text documents that make up the original 
     * data set from their source. For each document, 
     * {@link #addOriginalDocument(java.lang.String) } should be called with the
     * text of the document. <br>
     * Once all documents have been added, {@link #finishAdding() } should be 
     * called, so that post processing steps can be applied. <br>
     * New document vectors can be obtained after loading by calling 
     * {@link #newText(java.lang.String) }. 
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
        
        SparseVector vec = new SparseVector(currentLength+1, words.size());//+1 to avoid issues when its length is zero, will be corrected in finalization step anyway
        for(String word : words)
        {
            if(!wordIndex.containsKey(word))//this word has never been seen before!
            {
                allWords.add(word);
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
        
        weighting.setWeight(vectors, termDocumentFrequencys);
        for(SparseVector vec : vectors)
        {
            //Make sure all the vectors have the same length
            vec.setLength(currentLength);
            //Unlike normal index functions, WordWeighting needs to use the vector to do some set up first
            weighting.applyTo(vec);
        }
    }
    
    /**
     * Returns a new data set containing the original data points that were 
     * loaded with this loader. 
     * 
     * @return an appropriate data set for this loader
     */
    public DataSet getDataSet()
    {
        List<DataPoint> dataPoints= new ArrayList<DataPoint>(vectors.size());
        
        for(SparseVector vec : vectors)
            dataPoints.add(new DataPoint(vec, new int[0], new CategoricalData[0]));
        
        return new SimpleDataSet(dataPoints);
    }
    
    /**
     * To be called after all original texts have been loaded. 
     * 
     * @param text the text of the document to create a document vector from
     * @return the sparce vector representing this document 
     */
    public SparseVector newText(String text)
    {
        if(!noMoreAdding)
            throw new RuntimeException("Initial documents have not yet loaded");
        List<String> words = tokenizer.tokenize(text);
        
        SparseVector vec = new SparseVector(currentLength);
        
        for( String word : words)
        {
            if(wordIndex.containsKey(word))//Could also call retainAll on words before looping. Worth while to investigate 
            {
                int index = wordIndex.get(word);
                vec.increment(index, 1.0);
            }
        }
        
        weighting.applyTo(vec);
        
        return vec;
    }
    
    public String getWordForIndex(int index)
    {
        if(index >= 0 && index < allWords.size())
            return allWords.get(index);
        else
            return null;
    }
}
