
package jsat.text.stemming.tokenizer;

import java.util.List;
import jsat.text.stemming.Stemmer;

/**
 *
 * @author Edward Raff
 */
public class StemmingTokenizer implements Tokenizer
{
    private Stemmer stemmer;
    private Tokenizer baseTokenizer;

    public StemmingTokenizer(Stemmer stemmer, Tokenizer baseTokenizer)
    {
        this.stemmer = stemmer;
        this.baseTokenizer = baseTokenizer;
    }
    
    public List<String> tokenize(String input)
    {
        List<String> tokens = baseTokenizer.tokenize(input);
        stemmer.applyTo(tokens);
        return tokens;
    }
    
}
