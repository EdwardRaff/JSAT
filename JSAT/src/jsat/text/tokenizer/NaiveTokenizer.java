
package jsat.text.tokenizer;

import java.util.*;

/**
 *
 * A simple tokenizer. It converts everything to lower case, and splits on white space. Anything that is not a letter, digit, or space, is removed. 
 * 
 * @author Edward Raff
 */
public class NaiveTokenizer implements Tokenizer
{   
    @Override
    public List<String> tokenize(String input)
    {
        ArrayList<String> toRet = new ArrayList<String>();
        
        StringBuilder sb = new StringBuilder(input.length()/10);
        for(int i = 0; i < input.length(); i++)
        {
            char c = input.charAt(i);
            if(Character.isLetter(c))
                sb.append(Character.toLowerCase(c));
            else if(Character.isDigit(c))
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
