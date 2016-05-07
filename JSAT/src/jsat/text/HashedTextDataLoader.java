package jsat.text;

import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicIntegerArray;
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
 * as hashed feature vectors. Text is broken up into a sequence of tokens using
 * a {@link Tokenizer}, that must be provided. The weights used will be
 * determined by some {@link WordWeighting word weighting scheme}. <br>
 * The user adds documents to the initial dataset using the {@link #addOriginalDocument(java.lang.String)
 * } method. The {@link #finishAdding() } must be called when no more documents
 * are left to add, at which point class will take care of calling the {@link WordWeighting#setWeight(java.util.List, java.util.List)
 * } method to configure the word weighting used with the original data
 * added.<br>
 * <br>
 * After the initial dataset is loaded, new strings can be converted to vectors
 * using the {@link #newText(java.lang.String) } method. This should only be
 * called after {@link #finishAdding() }.<br>
 * <br>
 * Instance of this class will keep a reference to all originally added vectors.
 * To transform new texts into vectors without keeping references to all of the
 * original vectors, the {@link #getTextVectorCreator() } will return an object
 * that perform the transformation.
 * 
 * @author Edward Raff
 */
abstract public class HashedTextDataLoader implements TextVectorCreator
{

    private static final long serialVersionUID = 8513621180409278670L;
    private final int dimensionSize;
    /**
     * Tokenizer to apply to input strings
     */
    private Tokenizer tokenizer;
    private WordWeighting weighting;

    /**
     * List of original vectors
     */
    protected List<SparseVector> vectors;
    private AtomicIntegerArray termDocumentFrequencys;
    protected boolean noMoreAdding;
    private volatile int documents;
    
    /**
     * Temporary work space to use for tokenization
     */
    protected ThreadLocal<StringBuilder> workSpace;
    /**
     * Temporary storage space to use for tokenization
     */
    protected ThreadLocal<List<String>> storageSpace;
    /**
     * Temporary space to use when creating vectors
     */
    protected ThreadLocal<Map<String, Integer>> wordCounts;
    
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
        this.termDocumentFrequencys = new AtomicIntegerArray(dimensionSize);
        this.vectors = new ArrayList<SparseVector>();
        this.tvc = new HashedTextVectorCreator(dimensionSize, tokenizer, weighting);
        
        noMoreAdding = false;
        this.workSpace = new ThreadLocal<StringBuilder>();
        this.storageSpace = new ThreadLocal<List<String>>();
        this.wordCounts = new ThreadLocal<Map<String, Integer>>();
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
     * called again. <br>
     * This method is thread safe.
     * 
     * @param text the text of the document to add
     * @return the index of the created document for the given text. Starts from
     * zero and counts up.
     */
    protected int addOriginalDocument(String text)
    {
        if(noMoreAdding)
            throw new RuntimeException("Initial data set has been finalized");
        StringBuilder localWorkSpace = workSpace.get();
        List<String> localStorageSpace = storageSpace.get();
        Map<String, Integer> localWordCounts = wordCounts.get();
        if(localWorkSpace == null)
        {
            localWorkSpace = new StringBuilder();
            localStorageSpace = new ArrayList<String>();
            localWordCounts = new LinkedHashMap<String, Integer>();
            workSpace.set(localWorkSpace);
            storageSpace.set(localStorageSpace);
            wordCounts.set(localWordCounts);
        }
        
        localWorkSpace.setLength(0);
        localStorageSpace.clear();
        
        
        tokenizer.tokenize(text, localWorkSpace, localStorageSpace);
        
        for(String word : localStorageSpace)
        {
            Integer count = localWordCounts.get(word);
            if(count == null)
                localWordCounts.put(word, 1);
            else
                localWordCounts.put(word, count+1);
        }
        
        SparseVector vec = new SparseVector(dimensionSize, localWordCounts.size());
        for(Iterator<Entry<String, Integer>> iter = localWordCounts.entrySet().iterator(); iter.hasNext();)
        {
            Entry<String, Integer> entry = iter.next();
            String word = entry.getKey();
            //XXX This code generates a hashcode and then computes the absolute value of that hashcode. If the hashcode is Integer.MIN_VALUE, then the result will be negative as well (since Math.abs(Integer.MIN_VALUE) == Integer.MIN_VALUE). 
            int index = Math.abs(word.hashCode()) % dimensionSize;
            vec.set(index, entry.getValue());
            termDocumentFrequencys.addAndGet(index, entry.getValue());
            iter.remove();
        }
        
        synchronized(vectors)
        {
            vectors.add(vec);
            return documents++;
        }
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
        
        final int[] frqs = new int[dimensionSize];
        for(int i = 0; i < termDocumentFrequencys.length(); i++)
            frqs[i] = termDocumentFrequencys.get(i);
        weighting.setWeight(vectors, IntList.unmodifiableView(frqs, dimensionSize));
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
