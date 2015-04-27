
package jsat.text.tokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * A simple tokenizer. It converts everything to lower case, and splits on white
 * space. Anything that is not a letter, digit, or space, is treated as white 
 * space. This behavior can be altered slightly, and allows for setting a 
 * minimum and maximum allowed length for tokens. This can be useful when 
 * dealing with noisy documents, and removing small words. <br>
 * 
 * @author Edward Raff
 */
public class NaiveTokenizer implements Tokenizer
{   

	private static final long serialVersionUID = -2112091783442076933L;
	private boolean useLowerCase;
    private boolean otherToWhiteSpace = true;
    private boolean noDigits = false;
    private int minTokenLength = 1;
    private int maxTokenLength = Integer.MAX_VALUE;

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
            else if (!noDigits && Character.isDigit(c))
                workSpace.append(c);
            else if(!otherToWhiteSpace && !Character.isWhitespace(c))
                continue;
            else //end of token
            {
                if(workSpace.length() >= minTokenLength && workSpace.length() <= maxTokenLength)
                    storageSpace.add(workSpace.toString());
                workSpace.setLength(0);
            }
        }
        
        if(workSpace.length() >= minTokenLength && workSpace.length() <= maxTokenLength)
            storageSpace.add(workSpace.toString());
    }

    /**
     * Sets the maximum allowed length for any token. Any token discovered 
     * exceeding the length will not be accepted and skipped over. The default 
     * is unbounded. 
     * 
     * @param maxTokenLength the maximum token length to accept as a valid token
     */
    public void setMaxTokenLength(int maxTokenLength)
    {
        if(maxTokenLength < 1)
            throw new IllegalArgumentException("Max token length must be positive, not " + maxTokenLength);
        if(maxTokenLength <= minTokenLength)
            throw new IllegalArgumentException("Max token length must be larger than the min token length");
        this.maxTokenLength = maxTokenLength;
    }

    /**
     * Returns the maximum allowed token length
     * @return the maximum allowed token length
     */
    public int getMaxTokenLength()
    {
        return maxTokenLength;
    }

    /**
     * Sets the minimum allowed token length. Any token discovered shorter than 
     * the minimum length will not be accepted and skipped over. The default 
     * is 0. 
     * @param minTokenLength the minimum length for a token to be used 
     */
    public void setMinTokenLength(int minTokenLength)
    {
        if(minTokenLength < 0)
            throw new IllegalArgumentException("Minimum token length must be non negative, not " + minTokenLength);
        if(minTokenLength > maxTokenLength)
            throw new IllegalArgumentException("Minimum token length can not exced the maximum token length");
        this.minTokenLength = minTokenLength;
    }

    /**
     * Returns the minimum allowed token length
     * @return the maximum allowed token length
     */
    public int getMinTokenLength()
    {
        return minTokenLength;
    }

    /**
     * Sets whether digits will be accepted in tokens or treated as "other" (not
     * white space and not character). <br>
     * The default it to allow digits.
     * 
     * @param noDigits {@code true} to disallow numeric digits, {@code false} to 
     * allow digits. 
     */
    public void setNoDigits(boolean noDigits)
    {
        this.noDigits = noDigits;
    }

    /**
     * Returns {@code true} if digits are not allowed in tokens, {@code false} 
     * otherwise. 
     * @return {@code true} if digits are not allowed in tokens, {@code false} 
     * otherwise.
     */
    public boolean isNoDigits()
    {
        return noDigits;
    }
    
}
