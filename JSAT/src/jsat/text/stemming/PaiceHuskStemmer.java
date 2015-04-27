package jsat.text.stemming;

/**
 * Provides an implementation of the Paice Husk stemmer as described in: <br>
 * Paice, C. D. (1990). <i>Another Stemmer</i>. ACM SIGIR Forum, 4(3), 56–61.
 * @author Edward Raff
 */
public class PaiceHuskStemmer extends Stemmer
{
    

	private static final long serialVersionUID = -5949389288166850651L;

	static private class Rule
    {
        /**
         * The ending to try and match
         */
        public final String ending;
        
        /**
         * How many characters from the end of the string to remove
         */
        public final int toRemove;
        
        /**
         * The string to append the ending with
         */
        public final String newEnding;
        
        /**
         * Indicates that this rule may only be applied if the word has not been
         * modified before
         */
        public final boolean virgin;
        /**
         * Indicates that the stemming process should exit if this rule is applied
         */
        public final boolean terminal;

        public Rule(String ending, int toRemove, String newEnding, boolean virgin, boolean terminal)
        {
            this.ending = ending;
            this.toRemove = toRemove;
            this.newEnding = newEnding;
            this.virgin = virgin;
            this.terminal = terminal;
        }

        
        /**
         * If valid, returns the modified input string based on this rule. If no
         * modification was done, the exact same string that was passed in is 
         * returned. An object comparison can then be done to check if the 
         * string was modified. <br>
         * Appropriate handling of virgin and terminal flags is up to the user
         * @param input the unstemmed input
         * @return the stemmed output
         */
        public String apply(String input)
        {
            if(input.endsWith(ending))
            {
                if(isVowel(input.charAt(0)))
                {
                    //Stats with a vowel, stemmed result must be at least 2 chars long
                    if(input.length()-toRemove+newEnding.length() < 2)
                        return input;
                }
                else//Starts with a consonant, 3 lets must remain
                {
                    if(input.length()-toRemove+newEnding.length() < 3)
                        return input;//Not long enought
                    //Result must also contain at least one vowel
                    boolean noVowels = true;
                    for(int i = 0; i < input.length()-toRemove && noVowels; i++)
                        if(isVowel(input.charAt(i)) || input.charAt(i)== 'y')
                            noVowels = false;
                    for(int i = 0; i < newEnding.length() && noVowels; i++)
                        if(isVowel(newEnding.charAt(i)) || newEnding.charAt(i)== 'y')
                            noVowels = false;
                    if(noVowels)
                        return input;//No vowels left, stemmin is not valid to aply
                }
                //We made it, we can apply the stem and return a new string
                if(toRemove == 0)//Proctected word, return a new string explicitly to be super sure
                    return new String(input);
                return input.substring(0, input.length()-toRemove) + newEnding;
            }
            return input;
        }
    }
    
    /*
     * Oreded alphabetically by ending, meaning the rules should be attempted in
     * the order they are presented in the array
     */ 
    
    private static final Rule[] ARules = new Rule[]
    {
        new Rule("ia", 2, "", true, true), //ai*2.
        new Rule("a", 1, "", true, true), //a*1.
    };
    
    private static final Rule[] BRules = new Rule[]
    {
        new Rule("bb", 1, "", false, true), //bb1.
    };
    
    private static final Rule[] CRules = new Rule[]
    {
        new Rule("ytic", 3, "s", false, true),//city3s.
        new Rule("ic", 2, "", false, false), //ci2>
        new Rule("nc", 1, "t", false, false), //cn1t>
    };
    
    private static final Rule[] DRules = new Rule[]
    {
        new Rule("dd", 1, "", false, true), //dd1.
        new Rule("ied", 3, "y", false, false),//dei3y>
        new Rule("ceed", 2, "ss", false, true), //deec2ss.
        new Rule("eed", 1, "", false, true), //dee1.
        new Rule("ed", 2, "", false, false), //de2>
        new Rule("hood", 4, "", false, false), //dooh4>
    };
    
    private static final Rule[] ERules = new Rule[]
    {
        new Rule("e", 1, "", false, false), //e1>
    };
    
    private static final Rule[] FRules = new Rule[]
    {
        new Rule("lief", 1, "v", false, true), //feil1v.
        new Rule("if", 2, "", false, true), //fi2>
    };
    
    private static final Rule[] GRules = new Rule[]
    {
        new Rule("ing", 3, "", false, false), //gni3>
        new Rule("iag", 3, "y", false, true), //gai3y.
        new Rule("ag", 2, "", false, false), //ga2>
        new Rule("gg", 1, "", false, true), //gg1.
    };
    
    private static final Rule[] HRules = new Rule[]
    {
        new Rule("th", 2, "", true, true), //ht*2.
        new Rule("guish", 5, "ct", false, true), //hsiug5ct.
        new Rule("ish", 3, "", false, false), //hsi3>
    };
    
    private static final Rule[] IRules = new Rule[]
    {
        new Rule("i", 1, "", true, true), //i*1.
        new Rule("i", 1, "y", false, false), //i1y>
    };
    
    private static final Rule[] JRules = new Rule[]
    {
        new Rule("ij", 1, "d", false, true), //ji1d.
        new Rule("fuj", 1, "S", false, true), //juf1s.
        new Rule("uj", 1, "d", false, true), //ju1d.
        new Rule("oj", 1, "d", false, true), //jo1d.
        new Rule("hej", 1, "r", false, true), //jeh1r.
        new Rule("verj", 1, "t", false, true), //jrev1t.
        new Rule("misj", 2, "t", false, true), //jsim2t.
        new Rule("nj", 1, "d", false, true), //jn1d.
        new Rule("j", 1, "s", false, true), //j1s.
    };
    
    private static final Rule[] LRules = new Rule[]
    {
        new Rule("ifiabl", 6, "", false, true), //lbaifi6.
        new Rule("iabl", 4, "y", false, true), //lbai4y.
        new Rule("abl", 3, "", false, false), //lba3>
        new Rule("ibl", 3, "", false, true), //lbi3.
        new Rule("bil", 2, "l", false, false), //lib2l>
        new Rule("cl", 1, "", false, true), //lc1.
        new Rule("iful", 4, "y", false, true), //lufi4y.
        new Rule("ful", 3, "", false, false), //luf3>
        new Rule("uf", 2, "", false, true), //lu2.
        new Rule("ial", 3, "", false, false), //lai3>
        new Rule("ual", 3, "", false, false), //lau3>
        new Rule("al", 2, "", false, false), //la2>
        new Rule("ll", 1, "", false, true), //ll1.
    };
    
    private static final Rule[] MRules = new Rule[]
    {
        new Rule("ium", 3, "", false, true), //mui3.
        new Rule("mu", 2, "", true, true), //mu*2.
        new Rule("ism", 3, "", false, false), //msi3>
        new Rule("mm", 1, "", false, true), //mm1.
    };
    
    private static final Rule[] NRules = new Rule[]
    {
        new Rule("sion", 4, "j", false, false), //nois4j>
        new Rule("xion", 4, "ct", false, true), //noix4ct.
        new Rule("ion", 3, "", false, false), //noi3>
        new Rule("ian", 3, "", false, false), //nai3>
        new Rule("an", 2, "", false, false), //na2>
        new Rule("een", 0, "", false, true), //nee0.
        new Rule("en", 2, "", false, false), //ne2>
        new Rule("nn", 1, "", false, true), //nn1.
    };
    
    private static final Rule[] PRules = new Rule[]
    {
        new Rule("ship", 4, "", false, false), //pihs4>
        new Rule("pp", 1, "", false, true), //pp1.
    };
    
    private static final Rule[] RRules = new Rule[]
    {
        new Rule("er", 2, "", false, false), //re2>
        new Rule("ear", 0, "", false, true), //rea0.
        new Rule("ar", 2, "", false, true), //ra2.
        new Rule("or", 2, "", false, false), //ro2>
        new Rule("ur", 2, "", false, false), //ru2>
        new Rule("rr", 1, "", false, true), //rr1.
        new Rule("tr", 1, "", false, false), //rt1>
        new Rule("ier", 3, "y", false, false), //rei3y>
    };
    
    private static final Rule[] SRules = new Rule[]
    {
        new Rule("ies", 3, "y", false, false), //sei3y>
        new Rule("sis", 2, "", false, true), //sis2.
        new Rule("ness", 4, "", false, false), //ssen4>
        new Rule("ss", 0, "", false, true), //ss0.
        new Rule("ous", 3, "", false, false), //suo3>
        new Rule("us", 2, "", true, true), //su*2.
        new Rule("s", 1, "", true, false), //s*1>
        new Rule("s", 0, "", false, true), //s0.
    };
    
    private static final Rule[] TRules = new Rule[]
    {
        new Rule("plicat", 4, "y", false, true), //tacilp4y.
        new Rule("at", 2, "", false, false), //ta2>
        new Rule("ment", 4, "", false, false), //tnem4>
        new Rule("ent", 3, "", false, false), //tne3>
        new Rule("ant", 3, "", false, false), //tna3>
        new Rule("ript", 2, "b", false, true), //tpir2b.
        new Rule("orpt", 2, "b", false, true), //tpro2b.
        new Rule("duct", 1, "", false, true), //tcud1.
        new Rule("sumpt", 2, "", false, true), //tpmus2.
        new Rule("cept", 2, "iv", false, true), //tpec2iv.
        new Rule("olut", 2, "v", false, true), //tulo2v.
        new Rule("sist", 0, "", false, true), //tsis0.
        new Rule("ist", 3, "", false, false), //tsi3>
        new Rule("tt", 1, "", false, true), //tt1.
    };
    
    private static final Rule[] URules = new Rule[]
    {
        new Rule("iqu", 3, "", false, true), //uqi3.
        new Rule("ogu", 1, "", false, true), //ugo1.
    };
    
    private static final Rule[] VRules = new Rule[]
    {
        new Rule("siv", 3, "j", false, false), //vis3j>
        new Rule("iev", 0, "", false, true), //vie0.
        new Rule("iv", 2, "", false, false), //vi2>
    };
    
    private static final Rule[] YRules = new Rule[]
    {
        new Rule("bly", 1, "", false, false), //ylb1>
        new Rule("ily", 3, "y", false, false), //yli3y>
        new Rule("ply", 0, "", false, true), //ylp0.
        new Rule("ly", 2, "", false, false), //yl2>
        new Rule("ogy", 1, "", false, true), //ygo1.
        new Rule("phy", 1, "", false, true), //yhp1.
        new Rule("omy", 1, "", false, true), //ymo1.
        new Rule("opy", 1, "", false, true), //ypo1.
        new Rule("ity", 3, "", false, false), //yti3>
        new Rule("ety", 3, "", false, false), //yte3>
        new Rule("lty", 2, "", false, true), //ytl2.
        new Rule("istry", 5, "", false, true), //yrtsi5.
        new Rule("ary", 3, "", false, false), //yra3>
        new Rule("ory", 3, "", false, false), //yro3>
        new Rule("ify", 3, "", false, true), //yfi3.
        new Rule("ncy", 2, "t", false, false), //ycn2t>
        new Rule("acy", 3, "", false, false), //yca3>
    };
    
    private static final Rule[] ZRules = new Rule[]
    {
        new Rule("iz", 2, "", false, false), //zi2>
        new Rule("yz", 1, "s", false, true), //zy1s.
    };
    
    private static final Rule[] NoRules = new Rule[0];
    
    /**
     * The rules for the 
     */
    private static final Rule[][] rules = new Rule[][]
    {
        ARules, BRules, CRules, DRules, ERules, 
        FRules, GRules, HRules, IRules, JRules, 
        NoRules, LRules, MRules, NRules, NoRules,
        PRules, NoRules, RRules, SRules, TRules, 
        URules, VRules, NoRules, NoRules, YRules, 
        ZRules    
    };
    
    
    private static boolean isVowel(char letter)
    {
        return letter == 'a' || letter == 'e' || letter == 'i' || letter == 'o' || letter == 'u';
    }

    @Override
    public String stem(String word)
    {
        boolean virginRound = true;
        boolean stop;
        
        
        int charOffset = "a".charAt(0);
        do
        {
            stop = true;
            
            int ruleIndex = word.charAt(word.length()-1)-charOffset;
            if(ruleIndex < 0 || ruleIndex > rules.length)
                continue;
            for(Rule rule : rules[ruleIndex])
            {
                if(rule.virgin && !virginRound)
                    continue;
                String test = rule.apply(word);
                if(test != word)//Rule was applied, is it acceptable?
                {
                    word = test;
                    stop = false;
                    if(rule.terminal)
                        return word;
                    else
                        break;
                }
            }
            
            virginRound = false;
        }
        while(!stop);
        
        
        return word;
    }
    
}
