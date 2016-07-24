
package jsat.text;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.datatransform.RemoveAttributeTransform;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.text.tokenizer.Tokenizer;
import jsat.text.wordweighting.WordWeighting;
import jsat.utils.IntList;
import jsat.utils.IntSet;

/**
 * This class provides a framework for loading datasets made of Text documents
 * as vectors. Text is broken up into a sequence of tokens using a
 * {@link Tokenizer}, that must be provided. The weights used will be determined
 * by some {@link WordWeighting word weighting scheme}. <br>
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
public abstract class TextDataLoader implements TextVectorCreator
{

    private static final long serialVersionUID = -657253682338792871L;
    /**
     * List of original vectors
     */
    protected final List<SparseVector> vectors;
    /**
     * Tokenizer to apply to input strings
     */
    protected Tokenizer tokenizer;
    
    /**
     * Maps words to their associated index in an array
     */
    protected ConcurrentHashMap<String, Integer> wordIndex;
    /**
     * list of all word tokens encountered in order of first observation
     */
    protected List<String> allWords;
    /**
     * The map of integer counts of how many times each word token was seen. Key
     * is the index of the word, value is the number of times it was seen. Using
     * a map instead of a list so that it can be updated in a efficient thread
     * safe way
     */
    protected ConcurrentHashMap<Integer, AtomicInteger> termDocumentFrequencys;
    private WordWeighting weighting;
    
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
    
    /**
     * true when {@link #finishAdding() } is called, and no new original 
     * documents can be inserted
     */
    protected boolean noMoreAdding;
    private final AtomicInteger currentLength = new AtomicInteger(0);
    private volatile int documents;

    /**
     * Creates a new loader for text datasets
     * @param tokenizer the tokenization method to break up strings with
     * @param weighting the scheme to set the weights for feature vectors. 
     */
    public TextDataLoader(Tokenizer tokenizer, WordWeighting weighting)
    {
        this.vectors = new ArrayList<SparseVector>();
        this.tokenizer = tokenizer;
        
        this.wordIndex = new ConcurrentHashMap<String, Integer>();
        this.termDocumentFrequencys = new ConcurrentHashMap<Integer, AtomicInteger>();
        this.weighting = weighting;
        this.allWords = new ArrayList<String>();
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
    public abstract void initialLoad();
    
    /**
     * To be called by the {@link #initialLoad() } method. 
     * It will take in the text and add a new document 
     * vector to the data set. Once all text documents 
     * have been loaded, this method should never be 
     * called again. <br>
     * <br>
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
        localWordCounts.clear();
        
        tokenizer.tokenize(text, localWorkSpace, localStorageSpace);
        
        for(String word : localStorageSpace)
        {
            Integer count = localWordCounts.get(word);
            if(count == null)
                localWordCounts.put(word, 1);
            else
                localWordCounts.put(word, count+1);
        }
        
        SparseVector vec = new SparseVector(currentLength.get()+1, localWordCounts.size());//+1 to avoid issues when its length is zero, will be corrected in finalization step anyway
        for(Iterator<Map.Entry<String, Integer>> iter = localWordCounts.entrySet().iterator(); iter.hasNext();)
        {
            Map.Entry<String, Integer> entry = iter.next();
            String word = entry.getKey();
            
            int ms_to_sleep = 1;
            while(!addWord(word, vec, entry.getValue()))//try in a loop, expoential back off 
            {
                try
                {
                    Thread.sleep(ms_to_sleep);
                    ms_to_sleep = Math.min(100, ms_to_sleep*2);
                }
                catch (InterruptedException ex)
                {
                    Logger.getLogger(TextDataLoader.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }
        localWordCounts.clear();
        
        synchronized(vectors)
        {
            vectors.add(vec);
            return documents++;
        }
    }

    /**
     * Does the work to add a given word to the sparse vector. May not succeed
     * in race conditions when two ore more threads are trying to add a word at
     * the same time.
     *
     * @param word the word to add to the vector
     * @param vec the location to store the word occurrence
     * @param entry the number of times the word occurred
     * @return {@code true} if the word was successfully added. {@code false} if
     * the word wasn't added due to a race- and should be tried again
     */
    private boolean addWord(String word, SparseVector vec, Integer value)
    {
        Integer indx = wordIndex.get(word);
        if(indx == null)//this word has never been seen before!
        {
            Integer index_for_new_word;
            if((index_for_new_word = wordIndex.putIfAbsent(word, -1)) == null)//I won the race to insert this word into the map
            {
                /*
                * we need to do this increment after words to avoid a race
                * condition where two people incrment currentLength for the
                * same word, as that will throw off other word additions
                * before we can fix the problem
                */
                index_for_new_word = currentLength.getAndIncrement();
                wordIndex.put(word, index_for_new_word);//overwrite with correct value
            }
            if(index_for_new_word < 0)
                return false;
            
            //possible race on tdf as well when two threads found same new word at the same time
            AtomicInteger termCount = new AtomicInteger(0), tmp = null;
            tmp = termDocumentFrequencys.putIfAbsent(index_for_new_word, termCount);
            if(tmp != null)
                termCount = tmp;
            termCount.incrementAndGet();
            
            int newLen = Math.max(index_for_new_word+1, vec.length());
            vec.setLength(newLen);
            vec.set(index_for_new_word, value);
        }
        else//this word has been seen before
        {
            if(indx < 0)
                return false;
            
            AtomicInteger toInc = termDocumentFrequencys.get(indx);
            if (toInc == null)
            {
                //wordIndex and termDocumnetFrequences are not updated
                //atomicly together, so could get index but not have tDF ready
                toInc = termDocumentFrequencys.putIfAbsent(indx, new AtomicInteger(1));
                if (toInc == null)//other person finished adding before we "fixed" via putIfAbsent
                    toInc = termDocumentFrequencys.get(indx);
            }
            toInc.incrementAndGet();
            
            if (vec.length() <= indx)//happens when another thread sees the word first and adds it, then get check and find it- but haven't increased our vector legnth
                vec.setLength(indx+1);
            vec.set(indx, value);
        }
        
        return true;
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
        
        int finalLength = currentLength.get();
        int[] frqs = new int[finalLength];
        for(Map.Entry<Integer, AtomicInteger> entry : termDocumentFrequencys.entrySet())
            frqs[entry.getKey()] = entry.getValue().get();
        for(SparseVector vec : vectors)
        {
            //Make sure all the vectors have the same length
            vec.setLength(finalLength);
        }
        weighting.setWeight(vectors, IntList.view(frqs, finalLength));
        
        System.out.println("Final Length: " + finalLength);
        for(SparseVector vec : vectors)
        {
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
    
    /**
     * To be called after all original texts have been loaded. 
     * 
     * @param text the text of the document to create a document vector from
     * @return the sparce vector representing this document 
     */
    @Override
    public Vec newText(String text)
    {
        if(!noMoreAdding)
            throw new RuntimeException("Initial documents have not yet loaded");
        return getTextVectorCreator().newText(text);
    }

    @Override
    public Vec newText(String input, StringBuilder workSpace, List<String> storageSpace)
    {
        if(!noMoreAdding)
            throw new RuntimeException("Initial documents have not yet loaded");
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
        else if(tvc == null)
            tvc = new BasicTextVectorCreator(tokenizer, wordIndex, weighting);
        return tvc;
    }
    
    /**
     * Returns the original token for the given index in the data set
     * @param index the numeric feature index
     * @return the word token associated with the index
     */
    public String getWordForIndex(int index)
    {
        //lazy population of allWords array
        if(allWords.size() != wordIndex.size())//we added since this was done
        {
            while(allWords.size() < wordIndex.size())
                allWords.add("");
            for(Map.Entry<String, Integer> entry : wordIndex.entrySet())
                allWords.set(entry.getValue(), entry.getKey());
        }
        if(index >= 0 && index < allWords.size())
            return allWords.get(index);
        else
            return null;
    }
    
    /**
     * Return the number of times a token has been seen in the document
     * @param index the numeric feature index 
     * @return the total occurrence count for the feature
     */
    public int getTermFrequency(int index)
    {
        return termDocumentFrequencys.get(index).get();
    }
    
    /**
     * Creates a new transform factory to remove all features for tokens that 
     * did not occur a certain number of times
     * @param minCount the minimum number of occurrences to be kept as a feature
     * @return a transform factory for removing features that did not occur 
     * often enough
     */
    @SuppressWarnings("unchecked")
    public RemoveAttributeTransform getMinimumOccurrenceDTF(int minCount)
    {
        
        final Set<Integer> numericToRemove = new IntSet();
        for(int i = 0; i < termDocumentFrequencys.size(); i++)
            if(termDocumentFrequencys.get(i).get() < minCount)
                numericToRemove.add(i);
        
        return new RemoveAttributeTransform(Collections.EMPTY_SET, numericToRemove);
    }
}
