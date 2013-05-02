
package jsat.text.tokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * A simple tokenizer. It converts everything to lower case, and splits on white
 * space. Anything that is not a letter, digit, or space, is treated as white 
 * space. This behavior can be altered slightly <br>
 * 
 * @author Edward Raff
 */
public class NaiveTokenizer implements Tokenizer
{   
    private boolean useLowerCase;
    private boolean otherToWhiteSpace = true;

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

    /**
     * Sets whether or not characters are made to be lower case or not
     * @param useLowerCase 
     */
    public void setUseLowerCase(boolean useLowerCase)
    {
        this.useLowerCase = useLowerCase;
    }

    /**
     * Returns {@code true} if letters are converted to lower case, 
     * {@code false} for case sensitive
     * @return {@code true} if letters are converted to lower case, 
     */
    public boolean isUseLowerCase()
    {
        return useLowerCase;
    }

    /**
     * Sets whether or not all non letter and digit characters are treated as 
     * white space, or ignored completely. If ignored, the tokenizer parses the 
     * string as if all non letter, digit, and whitespace characters did not 
     * exist in the original string.<br>
     * <br>
     * Setting this to {@code false} can result in a lower feature count, 
     * especially for noisy documents. 
     * @param otherToWhiteSpace {@code true} to treat all "other" characters as 
     * white space, {@code false} to ignore them
     */
    public void setOtherToWhiteSpace(boolean otherToWhiteSpace)
    {
        this.otherToWhiteSpace = otherToWhiteSpace;
    }

    /**
     * Returns whether or not all other illegal characters are treated as 
     * whitespace, or ignored completely. 
     * @return {@code true} if all other characters are treated as whitespace
     */
    public boolean isOtherToWhiteSpace()
    {
        return otherToWhiteSpace;
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
            else if((Character.isWhitespace(c) || otherToWhiteSpace) && workSpace.length() > 0)
            {
                storageSpace.add(workSpace.toString());
                workSpace.setLength(0);
            }
        }
        
        if(workSpace.length() > 0)
            storageSpace.add(workSpace.toString());
    }
    
}
