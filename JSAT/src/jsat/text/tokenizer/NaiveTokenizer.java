
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
    private boolean otherToWhiteSpace = false;

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
        
        tokenize(input, sb, toRet);
        
        return toRet;
    }

    @Override
    public void tokenize(String input, StringBuilder workSpace, List<String> storageSpace)
    {
        for(int i = 0; i < input.length(); i++)
        {
            char c = input.charAt(i);
            if(Character.isLetter(c))
                if (useLowerCase)
                    workSpace.append(Character.toLowerCase(c));
                else
                    workSpace.append(c);
            else if (Character.isDigit(c))
                workSpace.append(c);
            else if(Character.isWhitespace(c) && workSpace.length() > 0)
            {
                storageSpace.add(workSpace.toString());
                workSpace.setLength(0);
            }
        }
        
        if(workSpace.length() > 0)
            storageSpace.add(workSpace.toString());
    }
    
}
