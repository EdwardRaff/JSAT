package jsat.text.tokenizer;

import java.util.*;

/**
 * This tokenizer wraps another such that any stop words that would have been 
 * returned by the base tokenizer are removed. The stop list is case sensitive.
 * 
 * @author Edward Raff
 */
public class StopWordTokenizer implements Tokenizer
{    

	private static final long serialVersionUID = 445704970760705567L;
	private Tokenizer base;
    private Set<String> stopWords;

    /**
     * Creates a new Stop Word tokenizer
     * @param base the base tokenizer to use
     * @param stopWords the collection of stop words to remove from 
     * tokenizations. A copy of the collection will be made
     */
    public StopWordTokenizer(Tokenizer base, Collection<String> stopWords)
    {
        this.base = base;
        this.stopWords = new HashSet<String>(stopWords);
    }
    
    /**
     * Creates a new Stop Word tokenizer
     * @param base the base tokenizer to use
     * @param stopWords the array of strings to use as stop words
     */
    public StopWordTokenizer(Tokenizer base, String... stopWords)
    {
        this(base, Arrays.asList(stopWords));
    }

    
    @Override
    public List<String> tokenize(String input)
    {
        List<String> tokens = base.tokenize(input);
        tokens.removeAll(stopWords);
        return tokens;
    }

    @Override
    public void tokenize(String input, StringBuilder workSpace, List<String> storageSpace)
    {
        base.tokenize(input, workSpace, storageSpace);
        storageSpace.removeAll(stopWords);
    }
    
    /**
     * This unmodifiable set contains a very small and simple stop word list for
     * English based on the 100 most common English words and includes all 
     * characters. All tokens the set are lowercase. <br>
     * This stop list is not meant to be authoritative or complete, but only a 
     * reasonable starting point that shouldn't degrade any common tasks. <br>
     * <br>
     * Significant gains can be realized by deriving a stop list better suited 
     * to your individual needs. 
     * 
     */
    public static final Set<String> ENGLISH_STOP_SMALL_BASE = Collections.unmodifiableSet(new HashSet<String>(Arrays.asList(
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "the", "of", "to", "and", "in", "is", "it", "you", "that", 
            "was", "for", "are", "on", "as", "have", "with", "they", "be", "at",
            "this", "from", "or", "had", "by", "but", "some", "what", "there", 
            "we", "can", "out", "other", "were", "all", "your", "when", "use", 
            "word", "said", "an", "each", "which", "do", "their", "if", "will", 
            "way", "about", "many", "them", "would", "thing", "than", "down",
            "too")));
}
