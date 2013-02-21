
package jsat.text.tokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * A simple tokenizer. It converts everything to lower case, and splits on white
 * space. Anything that is not a letter, digit, or space, is treated as white 
 * space. <br>
 * 
 * 
 * @author Edward Raff
 */
public class NaiveTokenizer implements Tokenizer
{   
    private boolean useLowerCase;

    /**
     * Creates a new naive tokenizer that converts words to lower case
     */
    public NaiveTokenizer()
    {
        this(true);
    }
    
    /**
     * Creates a new naive tokenizer
     * 
     * @param useLowerCase {@code true} to convert everything to lower, 
     * {@code false} to leave the case as is
     */
    public NaiveTokenizer(boolean useLowerCase)
    {
        this.useLowerCase = useLowerCase;
    }
    
    @Override
    public List<String> tokenize(String input)
    {
        ArrayList<String> toRet = new ArrayList<String>();
        
        StringBuilder sb = new StringBuilder(input.length()/10);
        for(int i = 0; i < input.length(); i++)
        {
            char c = input.charAt(i);
            if(Character.isLetter(c))
                if (useLowerCase)
                    sb.append(Character.toLowerCase(c));
                else
                    sb.append(c);
            else if (Character.isDigit(c))
                sb.append(c);
            else if(sb.length() > 0)
            {
                toRet.add(sb.toString());
                sb.setLength(0);
            }
        }
        
        return toRet;
    }
    
}
