
package jsat.text.stemming;

import java.io.Serializable;
import java.util.List;

/**
 * Stemmers are algorithms that attempt reduce strings to their common stem or
 * root word. For example, a stemmer might idly reduce "runs" "running" and 
 * "ran" to the single stem word "run". This reduces the feature space size, 
 * and allows multiple words that have the same meaning to be counted together. 
 * <br>
 * Do not expect perfect results from stemming. This class provides the 
 * contract for a stemmer that does not have any word history. 
 * 
 * @author Edward Raff
 */
public abstract class Stemmer implements Serializable
{

	private static final long serialVersionUID = 1889842876393488149L;

	/**
     * Reduce the given input to its stem word
     * @param word the unstemmed input word
     * @return the stemmed version of the word
     */
    abstract public String stem(String word);
    
    /**
     * Replaces each value in the list with the stemmed version of the word
     * @param list the list to apply stemming to
     */
    public void applyTo(List<String> list)
    {
        for(int i = 0; i < list.size(); i++)
            list.set(i, stem(list.get(i)));
    }
    
    /**
     * Replaces each value in the array with the stemmed version of the word
     * @param arr the array to apply stemming to
     */
    public void applyTo(String[] arr)
    {
        for(int i = 0; i < arr.length; i++)
            arr[i] = stem(arr[i]);
    }
}
