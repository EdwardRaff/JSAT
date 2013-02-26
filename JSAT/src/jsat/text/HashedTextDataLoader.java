package jsat.text;

import java.util.*;
import java.util.Map.Entry;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.text.tokenizer.Tokenizer;
import jsat.text.wordweighting.WordWeighting;
import jsat.utils.IntList;

/**
 * This class provides a framework for loading datasets made of Text documents 
 * as hashed feature vectors. 
 * 
 * @author Edward Raff
 */
abstract public class HashedTextDataLoader implements TextVectorCreator
{
    private final int dimensionSize;
    private Tokenizer tokenizer;
    private WordWeighting weighting;
    
    protected List<SparseVector> vectors;
    private int[] termDocumentFrequencys;
    protected boolean noMoreAdding;
    private int documents;
    
    /**
     * Temporary work space to use for tokenization
     */
    protected StringBuilder workSpace;
    /**
     * Temporary storage space to use for tokenization
     */
    protected List<String> storageSpace;
    /**
     * Temporary space to use when creating vectors
     */
    protected Map<String, Integer> wordCounts;
    
    private TextVectorCreator tvc;
    
    public HashedTextDataLoader(Tokenizer tokenizer, WordWeighting weighting)
    {
        this(1<<22, tokenizer, weighting);
    }

    public HashedTextDataLoader(int dimensionSize, Tokenizer tokenizer, WordWeighting weighting)
    {
        this.dimensionSize = dimensionSize;
        this.tokenizer = tokenizer;
        this.weighting = weighting;
        this.termDocumentFrequencys = new int[dimensionSize];
        this.vectors = new ArrayList<SparseVector>();
        this.tvc = new HashedTextVectorCreator(dimensionSize, tokenizer, weighting);
        
        noMoreAdding = false;
    }
    
    /**
     * This method will load all the text documents that make up the original 
     * data set from their source. For each document, 
     * {@link #addOriginalDocument(java.lang.String) } should be called with the
     * text of the document. <br>
     * This method will be called when {@link #getDataSet() } is called for the 
     * first time. <br>
     * New document vectors can be obtained after loading by calling 
     * {@link #newText(java.lang.String) }. 
     */
    protected abstract void initialLoad();

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
        if(workSpace == null)
        {
            workSpace = new StringBuilder();
            storageSpace = new ArrayList<String>();
        }
        
        workSpace.setLength(0);
        storageSpace.clear();
        
        
        tokenizer.tokenize(text, workSpace, storageSpace);
        /**
         * Create a new one every 50 so that we dont waist iteration time 
         * on many null elements when we occasionally load in an abnormally
         * large document 
         */
        if(documents % 50 == 0)
            wordCounts = new HashMap<String, Integer>(storageSpace.size());
        
        for(String word : storageSpace)
        {
            Integer count = wordCounts.get(word);
            if(count == null)
                wordCounts.put(word, 1);
            else
                wordCounts.put(word, count+1);
        }
        
        SparseVector vec = new SparseVector(dimensionSize, wordCounts.size());
        for(Iterator<Entry<String, Integer>> iter = wordCounts.entrySet().iterator(); iter.hasNext();)
        {
            Entry<String, Integer> entry = iter.next();
            String word = entry.getKey();
            
            int index = Math.abs(word.hashCode()) % dimensionSize;
            vec.set(index, entry.getValue());
            termDocumentFrequencys[index] += entry.getValue();
            iter.remove();
        }
        
        vectors.add(vec);
        documents++;
    }
    
    /**
     * Once all original documents have been added, this method is called so 
     * that post processing steps can be applied. 
     */
    protected void finishAdding()
    {
        noMoreAdding = true;
        
        workSpace = null;
        storageSpace = null;
        wordCounts = null;
        
        weighting.setWeight(vectors, IntList.unmodifiableView(termDocumentFrequencys, dimensionSize));
        for(SparseVector vec : vectors)
            weighting.applyTo(vec);
        termDocumentFrequencys = null;
    }
    
    /**
     * Returns a new data set containing the original data points that were 
     * loaded with this loader. 
     * 
     * @return an appropriate data set for this loader
     */
    public DataSet getDataSet()
    {
        if(!noMoreAdding)
        {
            initialLoad();
            finishAdding();
        }
        
        List<DataPoint> dataPoints= new ArrayList<DataPoint>(vectors.size());
        
        for(SparseVector vec : vectors)
            dataPoints.add(new DataPoint(vec, new int[0], new CategoricalData[0]));
        
        return new SimpleDataSet(dataPoints);
    }

    @Override
    public Vec newText(String input)
    {
        return getTextVectorCreator().newText(input);
    }

    @Override
    public Vec newText(String input, StringBuilder workSpace, List<String> storageSpace)
    {
        return getTextVectorCreator().newText(input, workSpace, storageSpace);
    }
        
    /**
     * Returns the {@link TextVectorCreator} used by this data loader to convert
     * documents into vectors. 
     * 
     * @return the text vector creator used by this class
     */
    public TextVectorCreator getTextVectorCreator()
    {
        if(!noMoreAdding)
            throw new RuntimeException("Initial documents have not yet loaded");
        return tvc;
    }
    
}
