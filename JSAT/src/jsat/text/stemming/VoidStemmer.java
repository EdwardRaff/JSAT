
package jsat.text.stemming;

/**
 * The most naive of stemming possible, this class simply returns whatever string is given to it. 
 * @author Edward Raff
 */
public class VoidStemmer extends Stemmer
{


	private static final long serialVersionUID = -5059926028932641447L;

	public String stem(String word)
    {
        return word;
    }
    
}
