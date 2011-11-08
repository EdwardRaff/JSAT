
package jsat.text.tokenizer;

import java.util.Arrays;
import java.util.List;
import jsat.text.stemming.Stemmer;

/**
 *
 * A simple tokenizer. It converts everything to lower case, and splits on white space. Anything that is not a letter, digit, or space, is removed. 
 * 
 * @author Edward Raff
 */
public class NaiveTokenizer implements Tokenizer
{
    public List<String> tokenize(String input)
    {
        return Arrays.asList(input.toLowerCase().replaceAll("[^a-z0-9\\s]+", "").split("\\s+"));
    }
    
}
