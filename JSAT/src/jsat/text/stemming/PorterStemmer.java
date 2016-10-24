
package jsat.text.stemming;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Implements Porter's stemming algorithm http://tartarus.org/~martin/PorterStemmer/def.txt . <br>
 * Implemented for ease of understanding and legibility rather than performance.
 * @author Edward Raff
 */
public class PorterStemmer extends Stemmer
{
    private static final long serialVersionUID = -3809291457988435043L;

    static final Map<String, String> step2_endings = new LinkedHashMap<String, String>();
    static
    {
        step2_endings.put("ational", "ate");
        step2_endings.put("tional", "tion");
        step2_endings.put("enci", "ence");
        step2_endings.put("anci", "ance");
        step2_endings.put("izer", "ize");
        step2_endings.put("abli", "able");
        step2_endings.put("alli", "al");
        step2_endings.put("entli", "ent");
        step2_endings.put("eli", "e");
        step2_endings.put("ousli", "ous");
        step2_endings.put("ization", "ize");
        step2_endings.put("ation", "ate");
        step2_endings.put("ator", "ate");
        step2_endings.put("alsim", "al");
        step2_endings.put("iveness", "ive");
        step2_endings.put("fulness", "ful");
        step2_endings.put("ousness", "ous");
        step2_endings.put("aliti", "al");
        step2_endings.put("iviti", "ive");
        step2_endings.put("biliti", "ble");
    }
    
    static final Map<String, String> step3_endings = new LinkedHashMap<String, String>();
    static
    {
        step3_endings.put("icate", "ic");
        step3_endings.put("ative", "");
        step3_endings.put("alize", "al");
        step3_endings.put("iciti", "ic");
        step3_endings.put("ical", "ic");
        step3_endings.put("ful", "");
        step3_endings.put("ness", "");
    }
    
    static final Map<String, String> step4_endings = new LinkedHashMap<String, String>();
    static
    {
        step4_endings.put("al", "");
        step4_endings.put("ance", "");
        step4_endings.put("ence", "");
        step4_endings.put("er", "");
        step4_endings.put("ic", "");
        step4_endings.put("able", "");
        step4_endings.put("ible", "");
        step4_endings.put("ant", "");
        step4_endings.put("ement", "");
        step4_endings.put("ment", "");
        step4_endings.put("ent", "");
        step4_endings.put("ion", "");
        step4_endings.put("ou", "");
        step4_endings.put("ism", "");
        step4_endings.put("ate", "");
        step4_endings.put("iti", "");
        step4_endings.put("ous", "");
        step4_endings.put("ive", "");
        step4_endings.put("ize", "");
    }
    
    
    @Override
    public String stem(String s)
    {
        String tmp;
        //Step 1a
        if (s.endsWith("sses"))
            s = s.replaceAll("sses$", "ss");
        else if (s.endsWith("ies"))
            s = s.replaceAll("ies$", "i");
        else if(s.endsWith("ss"))
        {
            //Do nothing
        }
        else if(s.endsWith("s"))
            s = s.substring(0, s.length()-1);


        //Step 1b
        boolean step1b_specialCase = false;//If the second or third of the rules in Step 1b is successful
        if (s.endsWith("eed"))
        {
            tmp = s.replaceAll("eed$", "ee");
            if(measure(tmp) > 0)
                s = tmp;
        }
        else if (s.endsWith("ed"))
        {
            tmp = s.replaceAll("ed$", "");
            if(containsVowel(tmp))
            {
                s = tmp;
                step1b_specialCase = true;
            }
        }
        else if (s.endsWith("ing"))
        {
            tmp = s.replaceAll("ing$", "");
            if(containsVowel(tmp))
            {
                s = tmp;
                step1b_specialCase = true;
            }
        }

        if (step1b_specialCase)
        {
            if (s.endsWith("at"))
                s = s.concat("e");
            else if (s.endsWith("bl"))
                s = s.concat("e");
            else if (s.endsWith("iz"))
                s = s.concat("e");
            else if(doubleConstant(s, 'l', 's', 'z'))
                s = s.substring(0, s.length()-1);//remove last letter
            else if(oRule(s) && measure(s) == 1)
                s = s.concat("e");
        }

        //Step 1c
        if(s.endsWith("y") && containsVowel(s.substring(0, s.length()-1)))
            s = s.substring(0, s.length()-1).concat("i");

        //Step 2
        for (Map.Entry<String, String> entry : step2_endings.entrySet())
            if (s.endsWith(entry.getKey()))
            {
                tmp = s.replaceAll(entry.getKey() + "$", entry.getValue());
                if (measure(tmp) > 0)
                {
                    s = tmp;
                    break;
                }
            }

        //Step 3
        for (Map.Entry<String, String> entry : step3_endings.entrySet())
            if (s.endsWith(entry.getKey()))
            {
                tmp = s.replaceAll(entry.getKey() + "$", entry.getValue());
                if (measure(tmp) > 0)
                {
                    s = tmp;
                    break;
                }
            }

        //Step 4
        for (Map.Entry<String, String> entry : step4_endings.entrySet())
            if (s.endsWith(entry.getKey()))
            {
                if(s.endsWith("ion") && !(s.length() >= 4 && (s.charAt(s.length()-4) == 's' || s.charAt(s.length()-4) == 't')))
                    continue;//special case on ion, and they didn't match
                tmp = s.replaceAll(entry.getKey() + "$", entry.getValue());
                if (measure(tmp) > 1)
                {
                    s = tmp;
                    break;
                }
            }
        
        //Step 5a
        if (s.endsWith("e"))
        {
            tmp = s.substring(0, s.length() - 1);
            if(measure(tmp) > 1)
                s = tmp;
            else if(measure(tmp) == 1 && !oRule(tmp))
                s = tmp;       
        }

        //Step 5b
        int lp = s.length()-1;
        if(lp < 1)
            return s;
        if(s.charAt(lp) == s.charAt(lp-1) && s.charAt(lp) == 'l')
        {
            tmp = s.substring(0, s.length() - 1);
            if(measure(tmp) > 1)
                s = tmp;
        }
        
        return s;
    }
    
    
    private static int measure(String s)
    {
        return measure(s, 0, s.length());
    }
    
    private static int measure(String c, int start, int length)
    {
        //[C](VC){m}[V]  
        //Measure == the value of m in the above exprsion
        int pos = start;
        int m = 0;
        //Move past first C, now we are detecing   (VC){m}[V]
        while(!isVowel(c, pos) && pos < (length - start))
            pos++;

        boolean vFollowedByC = false;

        do
        {
            vFollowedByC = false;
            while (isVowel(c, pos)&& pos < (length-start))
                pos++;
            while (!isVowel(c, pos) && pos < (length-start))
            {
                pos++;
                vFollowedByC = true;
            }

            m++;
        }
        while (pos < (length - start) && vFollowedByC);

        if(vFollowedByC)//VC <- endded like that, it counts
            return m;
        else//V <- ended in V, dosnt count
            return m-1;
    }

    private static boolean isVowel(String s, int pos)
    {
        /*
         * A \consonant\ in a word is a letter other than A, E, I, O or U, and other
         * than Y preceded by a consonant.
         */
        if (pos >= s.length())
            return false;

        switch (s.charAt(pos))
        {
            case 'a':
            case 'e':
            case 'i':
            case 'o':
            case 'u':
                return true;
            case 'y':
                if (pos == s.length() - 1)//end of the array
                    return true;
                return isVowel(s, pos + 1);//Y preceded by a constant is a Vowel
            default:
                return false;
        }
    }
    
    /**
     * *o  - the stem ends cvc, where the second c is not W, X or Y (e.g. -WIL, -HOP).
     */
    private static boolean oRule(String s)
    {
        int pos = s.length()-1;
        if(pos < 2)
            return false;
        if (!isVowel(s, pos) && isVowel(s, pos - 1) && !isVowel(s, pos - 2))
        {
            switch (s.charAt(pos))
            {
                case 'w':
                case 'x':
                case 'y':
                    return false;
                default:
                    return true;
            }
        }
        return false;
    }
    
    private static boolean containsVowel(String s)
    {
        for (int i = 0; i < s.length(); i++)
            if (isVowel(s, i))
                return true;
        return false;
    }
    
    private static boolean doubleConstant(String s, char... except)
    {
        if (s.length() <= 1)
            return false;

        char c;
        if ((c = s.charAt(s.length() - 1)) == s.charAt(s.length() - 2))
        {
            for (char e : except)
                if (c == e)
                    return false;
            return true;
        }

        return false;
    }
}
