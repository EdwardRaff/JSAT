package jsat.text.tokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 * This tokenizer creates n-grams, which are sequences of tokens combined into 
 * their own larger token. For example, "the dog barked" could be a 3-gram. If 
 * all sub n-grams are also being generated, the returned set would contain the 
 * 1-grams "the", "dog", and "barked", the 2-grams "the dog" and "dog barked", 
 * and the aforementioned 3-gram. For this to work, this tokenizer assumes the 
 * base tokenizer returns tokens in the order they were seen. <br>
 * Note that n-grams can significantly increase the number of unique tokens, and
 * n-grams are inherently rarer than the 1-grams they are generated from. 
 * 
 * @author Edward Raff
 */
public class NGramTokenizer implements Tokenizer
{

	private static final long serialVersionUID = 7551087420391197139L;
	/**
     * The number of n-grams to generate
     */
    private int n;
    /**
     * The base tokenizer
     */
    private Tokenizer base;
    /**
     * whether or not to generate all sub n-grams
     */
    private boolean allSubN;

    /**
     * Creates a new n-gramer 
     * @param n the length of the ngrams. While it should be greater than 1, 1 
     * is still a valid input. 
     * @param base the base tokenizer to create n-grams from
     * @param allSubN {@code true} to generate all sub n-grams, {@code false} to 
     * only return the n-grams specified
     */
    public NGramTokenizer(int n, Tokenizer base, boolean allSubN)
    {
        if(n <= 0)
            throw new IllegalArgumentException("Number of n-grams must be positive, not " + n);
        this.n = n;
        this.base = base;
        this.allSubN = allSubN;
    }

    
    @Override
    public List<String> tokenize(String input)
    {
        List<String> storageSpace = new ArrayList<String>();
        tokenize(input, new StringBuilder(), storageSpace);
        return storageSpace;
    }

    @Override
    public void tokenize(String input, StringBuilder workSpace, List<String> storageSpace)
    {
        base.tokenize(input, workSpace, storageSpace);//the "1-grams"
        int origSize = storageSpace.size();
        if(n == 1)
            return;//nothing more to do

        for (int i = 1; i < origSize; i++)//slide from left to right on the 1-grams
        {
            //generate the n-grams from 2 to n
            for (int gramSize = allSubN ? 2 : n; gramSize <= n; gramSize++)
            {
                workSpace.setLength(0);
                int j = i - (gramSize - 1);
                if(j < 0)//means we are going past what we have, and we would be adding duplicates
                    continue;
                for(; j < i; j++)
                {
                    if (workSpace.length() > 0)
                        workSpace.append(' ');
                    workSpace.append(storageSpace.get(j));
                }
                workSpace.append(' ').append(storageSpace.get(i));
                storageSpace.add(workSpace.toString());
            }
        }
        
        if(!allSubN)//dont generate subs! get rid of those dirty 1-grams
            storageSpace.subList(0, origSize).clear();
    }
    
}
