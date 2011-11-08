
package jsat.text.stemming;

/**
 * The most naive of stemming possible, this class simply returns whatever string is given to it. 
 * @author Edward Raff
 */
public class VoidStemmer extends Stemmer
{

    public String stem(String word)
    {
        return word;
    }
    
}
