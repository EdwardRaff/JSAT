
package jsat.text.stemming;

/**
 * Implements Porter's stemming algorithm http://tartarus.org/~martin/PorterStemmer/def.txt . <br>
 * Implemented for ease of understanding and legibility rather than performance.
 * @author Edward Raff
 */
public class PorterStemmer extends Stemmer
{

    public String stem(String s)
    {
        //Step 1a
        if(s.endsWith("sses"))
            s = s.replaceAll("sses$", "ss");
        else if(s.endsWith("ies"))
            s = s.replaceAll("ies$", "s");
        else if(s.endsWith("ss"))
        {
            //Do nothing
        }
        else if(s.endsWith("s"))
            s = s.substring(0, s.length()-1);


        //Step 1b
        boolean step1b_specialCase = false;//If the second or third of the rules in Step 1b is successful
        if (s.endsWith("eed") && measure(s) > 0)
            s = s.replaceAll("eed$", "ee");
        else if (s.endsWith("ed") && measure(s) > 1)//(*v*) ED  ->   null, its eqivalent to (m>1) ED ->
        {
            s = s.replaceAll("ed$", "");
            step1b_specialCase = true;
        }
        else if (s.endsWith("ing") && measure(s) > 1)//(*v*) ING  ->   null, its eqivalent to (m>1) ING ->
        {
            s = s.replaceAll("ing$", "");
            step1b_specialCase = true;
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
        if(measure(s) > 0)
        {
            if (s.endsWith("ational"))
                s = s.replaceAll("ational$", "ate");
            else if(s.endsWith("tional"))
                s = s.replaceAll("tional$", "tion");
            else if(s.endsWith("enci"))
                s = s.replaceAll("enci$", "ence");
            else if(s.endsWith("anci"))
                s = s.replaceAll("anci$", "ance");
            else if(s.endsWith("izer"))
                s = s.replaceAll("izer$", "ize");
            else if(s.endsWith("abli"))
                s = s.replaceAll("abli$", "able");
            else if(s.endsWith("alli"))
                s = s.replaceAll("alli$", "al");
            else if(s.endsWith("entli"))
                s = s.replaceAll("entli$", "ent");
            else if(s.endsWith("eli"))
                s = s.replaceAll("eli$", "e");
            else if(s.endsWith("ousli"))
                s = s.replaceAll("ousli$", "ous");
            else if(s.endsWith("ization"))
                s = s.replaceAll("ization$", "ize");
            else if(s.endsWith("ation"))
                s = s.replaceAll("ation$", "ate");
            else if(s.endsWith("ator"))
                s = s.replaceAll("ator$", "ate");
            else if(s.endsWith("alsim"))
                s = s.replaceAll("alsim$", "al");
            else if(s.endsWith("iveness"))
                s = s.replaceAll("iveness$", "ive");
            else if(s.endsWith("fulness"))
                s = s.replaceAll("fulness$", "ful");
            else if(s.endsWith("ousness"))
                s = s.replaceAll("ousness$", "ous");
            else if(s.endsWith("aliti"))
                s = s.replaceAll("aliti$", "al");
            else if(s.endsWith("iviti"))
                s = s.replaceAll("iviti$", "ive");
            else if(s.endsWith("biliti"))
                s = s.replaceAll("biliti$", "ble");
        }

        //Step 3
        if(measure(s) > 0)
        {
            if(s.endsWith("icate"))
                s = s.replaceAll("icate$", "oc");
            else if(s.endsWith("ative"))
                s = s.replaceAll("ative$", "");
            else if(s.endsWith("alize"))
                s = s.replaceAll("alize$", "al");
            else if(s.endsWith("iciti"))
                s = s.replaceAll("iciti$", "ic");
            else if(s.endsWith("ical"))
                s = s.replaceAll("ical$", "ic");
            else if(s.endsWith("ful"))
                s = s.replaceAll("$", "");
            else if(s.endsWith("ness"))
                s = s.replaceAll("$", "");
        }

        //Step 4
        if(measure(s) > 1)
        {
            if(s.endsWith("al"))
                s = s.replaceAll("al$", "");
            else if(s.endsWith("ance"))
                s = s.replaceAll("ance$", "");
            else if(s.endsWith("ence"))
                s = s.replaceAll("ence$", "");
            else if(s.endsWith("er"))
                s = s.replaceAll("er$", "");
            else if(s.endsWith("ic"))
                s = s.replaceAll("ic$", "");
            else if(s.endsWith("able"))
                s = s.replaceAll("able$", "");
            else if(s.endsWith("ible"))
                s = s.replaceAll("ible$", "");
            else if(s.endsWith("ant"))
                s = s.replaceAll("ant$", "");
            else if(s.endsWith("ement"))
                s = s.replaceAll("ement$", "");
            else if(s.endsWith("ment"))
                s = s.replaceAll("ment$", "");
            else if(s.endsWith("ent"))
                s = s.replaceAll("ent$", "");
            else if(s.endsWith("ion") &&
                    (s.charAt(s.length()-4) == 's' || s.charAt(s.length()-4) == 's'))
                s = s.replaceAll("ion$", "");
            else if(s.endsWith("ou"))
                s = s.replaceAll("ou$", "");
            else if(s.endsWith("ism"))
                s = s.replaceAll("ism$", "");
            else if(s.endsWith("ate"))
                s = s.replaceAll("ate$", "");
            else if(s.endsWith("iti"))
                s = s.replaceAll("iti$", "");
            else if(s.endsWith("ous"))
                s = s.replaceAll("ous$", "");
            else if(s.endsWith("ive"))
                s = s.replaceAll("ive$", "");
            else if(s.endsWith("ize"))
                s = s.replaceAll("ize$", "");
        }
        //Step 5a

        if (s.endsWith("e") && measure(s) > 1)
            s = s.substring(0, s.length() - 1);
        else if(measure(s) == 1 && !oRule(s))
            s = s.substring(0, s.length() - 1);

        //Step 5b
        int lp = s.length()-1;
        if(measure(s) > 1 && s.charAt(lp) == s.charAt(lp-1) && s.charAt(lp) == 'l')
            s = s.substring(0, s.length() - 1);
        
        return s;
    }
    
    
    private static int measure(String s)
    {
        return measure(s.toCharArray(), 0, s.length());
    }
    
    private static int measure(char[] c, int start, int length)
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
    
    private static boolean isVowel(char[] c, int pos)
    {
        /*
         * A \consonant\ in a word is a letter other than A, E, I, O or U, and other
         * than Y preceded by a consonant.
         */
        if(pos >= c.length)
            return false;

        switch (c[pos])
        {
            case 'a':
            case 'e':
            case 'i':
            case 'o':
            case 'u':
                return true;
            case 'y':
                if(pos == c.length-1)//end of the array
                    return true;
                return isVowel(c, pos+1);//Y preceded by a constant is a Vowel
            default:
                return false;
        }
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
