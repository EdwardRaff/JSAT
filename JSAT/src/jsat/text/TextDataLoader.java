
package jsat.text;

import java.util.*;

import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.datatransform.RemoveAttributeTransform.RemoveAttributeTransformFactory;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.text.tokenizer.Tokenizer;
import jsat.text.wordweighting.WordWeighting;
import jsat.utils.IntList;
import jsat.utils.IntSet;

/**
 * This class provides a framework for loading datasets made of Text documents 
 * as vectors. 
 * 
 * @author Edward Raff 
 */
public abstract class TextDataLoader implements TextVectorCreator
{

	private static final long serialVersionUID = -657254682338792871L;
	/**
     * List of original vectors
     */
    protected List<SparseVector> vectors;
    /**
     * Tokenizer to apply to input strings
     */
    protected Tokenizer tokenizer;
    
    /**
     * Maps words to their associated index in an array
     */
    protected Map<String, Integer> wordIndex;
    /**
     * list of all word tokens encountered in order of first observation
     */
    protected List<String> allWords;
    /**
     * The list of integer counts of how many times each word token was seen
     */
    protected List<Integer> termDocumentFrequencys;
    private WordWeighting weighting;
    
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
    
    /**
     * true when {@link #finishAdding() } is called, and no new original 
     * documents can be inserted
     */
    protected boolean noMoreAdding;
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
     * called again. 
     * 
     * @param text the text of the document to add
     */
    protected void addOriginalDocument(String text)
    {
        if(noMoreAdding) {
          throw new RuntimeException("Initial data set has been finalized");
        }
        if(workSpace == null)
        {
            workSpace = new StringBuilder();
            storageSpace = new ArrayList<String>();
            wordCounts = new HashMap<String, Integer>();
        }

        workSpace.setLength(0);
        storageSpace.clear();
        
        tokenizer.tokenize(text, workSpace, storageSpace);
        /**
         * Create a new one every 50 so that we dont waist iteration time 
         * on many null elements when we occasionally load in an abnormally
         * large document 
         */
        if(documents % 50 == 0) {
          wordCounts = new HashMap<String, Integer>(storageSpace.size()*3/2);
        }
        
        for(String word : storageSpace)
        {
            Integer count = wordCounts.get(word);
            if(count == null) {
              wordCounts.put(word, 1);
            } else {
              wordCounts.put(word, count+1);
            }
        }
        
        SparseVector vec = new SparseVector(currentLength+1, wordCounts.size());//+1 to avoid issues when its length is zero, will be corrected in finalization step anyway
        for(Iterator<Map.Entry<String, Integer>> iter = wordCounts.entrySet().iterator(); iter.hasNext();)
        {
            Map.Entry<String, Integer> entry = iter.next();
            String word = entry.getKey();
            
            Integer indx = wordIndex.get(word);
            if(indx == null)//this word has never been seen before!
            {
                allWords.add(word);
                wordIndex.put(word, currentLength++);
                termDocumentFrequencys.add(1);
                vec.setLength(currentLength);
                vec.set(currentLength-1, entry.getValue());
            }
            else//this word has been seen before
            {
                termDocumentFrequencys.set(indx, termDocumentFrequencys.get(indx)+1);
                vec.set(indx, entry.getValue());
            }
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
        if(!noMoreAdding)
        {
            initialLoad();
            finishAdding();
        }
        
        List<DataPoint> dataPoints= new ArrayList<DataPoint>(vectors.size());
        
        for(SparseVector vec : vectors) {
          dataPoints.add(new DataPoint(vec, new int[0], new CategoricalData[0]));
        }
        
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
        if(!noMoreAdding) {
          throw new RuntimeException("Initial documents have not yet loaded");
        }
        return getTextVectorCreator().newText(text);
    }

    @Override
    public Vec newText(String input, StringBuilder workSpace, List<String> storageSpace)
    {
        if(!noMoreAdding) {
          throw new RuntimeException("Initial documents have not yet loaded");
        }
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
        if(!noMoreAdding) {
          throw new RuntimeException("Initial documents have not yet loaded");
        } else if(tvc == null) {
          tvc = new BasicTextVectorCreator(tokenizer, wordIndex, weighting);
        }
        return tvc;
    }
    
    /**
     * Returns the original token for the given index in the data set
     * @param index the numeric feature index
     * @return the word token associated with the index
     */
    public String getWordForIndex(int index)
    {
        if(index >= 0 && index < allWords.size()) {
          return allWords.get(index);
        } else {
          return null;
        }
    }
    
    /**
     * Return the number of times a token has been seen in the document
     * @param index the numeric feature index 
     * @return the total occurrence count for the feature
     */
    public int getTermFrequency(int index)
    {
        return termDocumentFrequencys.get(index);
    }
    
    /**
     * Creates a new transform factory to remove all features for tokens that 
     * did not occur a certain number of times
     * @param minCount the minimum number of occurrences to be kept as a feature
     * @return a transform factory for removing features that did not occur 
     * often enough
     */
    @SuppressWarnings("unchecked")
    public RemoveAttributeTransformFactory getMinimumOccurrenceDTF(int minCount)
    {
        
        final Set<Integer> numericToRemove = new IntSet();
        for(int i = 0; i < termDocumentFrequencys.size(); i++) {
          if (termDocumentFrequencys.get(i) < minCount) {
            numericToRemove.add(i);
          }
        }
        
        return new RemoveAttributeTransformFactory(Collections.EMPTY_SET, numericToRemove);
    }
}
