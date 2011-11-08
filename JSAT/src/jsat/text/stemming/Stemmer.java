
package jsat.text.stemming;

import java.util.List;

/**
 *
 * @author Edward Raff
 */
public abstract class Stemmer
{
    abstract public String stem(String word);
    
    public void applyTo(List<String> list)
    {
        for(int i = 0; i < list.size(); i++)
            list.set(i, stem(list.get(i)));
    }
    
    public void applyTo(String[] arr)
    {
        for(int i = 0; i < arr.length; i++)
            arr[i] = stem(arr[i]);
    }
}
