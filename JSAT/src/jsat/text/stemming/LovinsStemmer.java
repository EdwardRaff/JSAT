
package jsat.text.stemming;

import java.util.HashMap;

/**
 * Implements Lovins' stemming algorithm described here 
 * http://snowball.tartarus.org/algorithms/lovins/stemmer.html
 * 
 * @author Edward Raff
 */
public class LovinsStemmer extends Stemmer
{
    

	private static final long serialVersionUID = -3229865664217642197L;

	//There are 11 ending hash maps, each postfixed with the number of characters
    private static final HashMap<String, String> ending11 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = 4064350307133685150L;

	{
        put("alistically", "B"); put("arizability", "A"); put("izationally", "B");
    }};
    
    private static final HashMap<String, String> ending10 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -5247798032923242997L;

	{
        put("antialness", "A"); put("arisations", "A"); put("arizations", "A");
        put("entialness", "A");
    }};
    
    private static final HashMap<String, String> ending9 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -9153017770847287495L;

	{
        put("allically", "C"); put("antaneous", "A"); put("antiality", "A");
        put("arisation", "A"); put("arization", "A"); put("ationally", "B");
        put("ativeness", "A"); put("eableness", "E"); put("entations", "A");
        put("entiality", "A"); put("entialize", "A"); put("entiation", "A");
        put("ionalness", "A"); put("istically", "A"); put("itousness", "A");
        put("izability", "A"); put("izational", "A");
    }};
    
    private static final HashMap<String, String> ending8 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = 3671522347706544570L;

	{
        put("ableness", "A"); put("arizable", "A"); put("entation", "A");
        put("entially", "A"); put("eousness", "A"); put("ibleness", "A");
        put("icalness", "A"); put("ionalism", "A"); put("ionality", "A");
        put("ionalize", "A"); put("iousness", "A"); put("izations", "A"); 
        put("lessness", "A");
    }};
    
    private static final HashMap<String, String> ending7 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -4697823317524161452L;

	{
        put("ability", "A"); put("aically", "A"); put("alistic", "B");
        put("alities", "A"); put("ariness", "E"); put("aristic", "A");
        put("arizing", "A"); put("ateness", "A"); put("atingly", "A");
        put("ational", "B"); put("atively", "A"); put("ativism", "A");
        put("elihood", "E"); put("encible", "A"); put("entally", "A");
        put("entials", "A"); put("entiate", "A"); put("entness", "A");
        put("fulness", "A"); put("ibility", "A"); put("icalism", "A");
        put("icalist", "A"); put("icality", "A"); put("icalize", "A");
        put("ication", "G"); put("icianry", "A"); put("ination", "A");
        put("ingness", "A"); put("ionally", "A"); put("isation", "A");
        put("ishness", "A"); put("istical", "A"); put("iteness", "A");
        put("iveness", "A"); put("ivistic", "A"); put("ivities", "A");
        put("ization", "F"); put("izement", "A"); put("oidally", "A");
        put("ousness", "A");
    }};
    
    private static final HashMap<String, String> ending6 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -7030401064572348271L;

	{
        put("aceous", "A"); put("acious", "B"); put("action", "G");
        put("alness", "A"); put("ancial", "A"); put("ancies", "A");
        put("ancing", "B"); put("ariser", "A"); put("arized", "A");
        put("arizer", "A"); put("atable", "A"); put("ations", "B");
        put("atives", "A"); put("eature", "Z"); put("efully", "A");
        put("encies", "A"); put("encing", "A"); put("ential", "A");
        put("enting", "C"); put("entist", "A"); put("eously", "A");
        put("ialist", "A"); put("iality", "A"); put("ialize", "A");
        put("ically", "A"); put("icance", "A"); put("icians", "A");
        put("icists", "A"); put("ifully", "A"); put("ionals", "A");
        put("ionate", "D"); put("ioning", "A"); put("ionist", "A");
        put("iously", "A"); put("istics", "A"); put("izable", "E");
        put("lessly", "A"); put("nesses", "A"); put("oidism", "A");
    }};
    
    private static final HashMap<String, String> ending5 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -5282435864116373834L;

	{
        put("acies", "A"); put("acity", "A"); put("aging", "B");
        put("aical", "A"); put("alism", "B"); put("ality", "A");
        put("alize", "A"); put("allic", "b"); put("anced", "B");
        put("ances", "B"); put("antic", "C"); put("arial", "A");
        put("aries", "A"); put("arily", "A"); put("arity", "B");
        put("arize", "A"); put("aroid", "A"); put("ately", "A");
        put("ating", "I"); put("ation", "B"); put("ative", "A");
        put("ators", "A"); put("atory", "A"); put("ature", "E");
        put("early", "Y"); put("ehood", "A"); put("eless", "A");
        put("ement", "A"); put("enced", "A"); put("ences", "A");
        put("eness", "E"); put("ening", "E"); put("ental", "A");
        put("ented", "C"); put("ently", "A"); put("fully", "A");
        put("ially", "A"); put("icant", "A"); put("ician", "A");
        put("icide", "A"); put("icism", "A"); put("icist", "A");
        put("icity", "A"); put("idine", "I"); put("iedly", "A");
        put("ihood", "A"); put("inate", "A"); put("iness", "A");
        put("ingly", "B"); put("inism", "J"); put("inity", "c");
        put("ional", "A"); put("ioned", "A"); put("ished", "A");
        put("istic", "A"); put("ities", "A"); put("itous", "A");
        put("ively", "A"); put("ivity", "A"); put("izers", "F");
        put("izing", "F"); put("oidal", "A"); put("oides", "A");
        put("otide", "A"); put("ously", "A");
    }};
    
    private static final HashMap<String, String> ending4 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -7293777277850278026L;

	{
        put("able", "A"); put("ably", "A"); put("ages", "B");
        put("ally", "B"); put("ance", "B"); put("ancy", "B");
        put("ants", "B"); put("aric", "A"); put("arly", "K");
        put("ated", "I"); put("ates", "A"); put("atic", "B");
        put("ator", "A"); put("ealy", "Y"); put("edly", "E");
        put("eful", "A"); put("eity", "A"); put("ence", "A");
        put("ency", "A"); put("ened", "E"); put("enly", "E");
        put("eous", "A"); put("hood", "A"); put("ials", "A");
        put("ians", "A"); put("ible", "A"); put("ibly", "A");
        put("ical", "A"); put("ides", "L"); put("iers", "A");
        put("iful", "A"); put("ines", "M"); put("ings", "N");
        put("ions", "B"); put("ious", "A"); put("isms", "B");
        put("ists", "A"); put("itic", "H"); put("ized", "F");
        put("izer", "F"); put("less", "A"); put("lily", "A");
        put("ness", "A"); put("ogen", "A"); put("ward", "A");
        put("wise", "A"); put("ying", "B"); put("yish", "A");
    }};
    
    private static final HashMap<String, String> ending3 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -5629841014950478203L;

	{
        put("acy", "A"); put("age", "B"); put("aic", "A");
        put("als", "b"); put("ant", "B"); put("ars", "O");
        put("ary", "F"); put("ata", "A"); put("ate", "A");
        put("eal", "Y"); put("ear", "Y"); put("ely", "E");
        put("ene", "E"); put("ent", "C"); put("ery", "E");
        put("ese", "A"); put("ful", "A"); put("ial", "A");
        put("ian", "A"); put("ics", "A"); put("ide", "L");
        put("ied", "A"); put("ier", "A"); put("ies", "P");
        put("ily", "A"); put("ine", "M"); put("ing", "N");
        put("ion", "Q"); put("ish", "C"); put("ism", "B");
        put("ist", "A"); put("ite", "a"); put("ity", "A");
        put("ium", "A"); put("ive", "A"); put("ize", "F");
        put("oid", "A"); put("one", "R"); put("ous", "A");
    }};
    
    private static final HashMap<String, String> ending2 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -8894812965945848256L;

	{
        put("ae", "A"); put("al", "b"); put("ar", "X");
        put("as", "B"); put("ed", "E"); put("en", "F");
        put("es", "E"); put("ia", "A"); put("ic", "A");
        put("is", "A"); put("ly", "B"); put("on", "S");
        put("or", "T"); put("um", "U"); put("us", "V");
        put("yl", "R"); put("s\'", "A"); put("\'s", "A");
    }};
    
    private static final HashMap<String, String> ending1 = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -7536643426902207427L;

	{
        put("a", "A"); put("e", "A"); put("i", "A");
        put("o", "A"); put("s", "W"); put("y", "B");	
    }};
    
    private static final HashMap<String, String> endings = new HashMap<String, String>()
    {/**
		 * 
		 */
		private static final long serialVersionUID = -8057392854617089310L;

	{
        putAll(ending11); putAll(ending10); putAll(ending9); putAll(ending8); putAll(ending7); putAll(ending6);
        putAll(ending5); putAll(ending4); putAll(ending3); putAll(ending2); putAll(ending1);
    }};
    
    
    
    private static String removeEnding(String word)
    {
        //The stem must contain at least 2 characters, so word-2 is the min
        for(int i = Math.min(11, word.length()-2); i > 0; i--)
        {
            String ending = word.substring(word.length()-i);
            String condition = endings.get(ending);
            if(condition == null)
                continue;
            
            String stem = word.substring(0, word.length()-i);
            switch(condition.charAt(0))
            {
                case 'A'://No restrictions on stem
                    return stem;
                case 'B': //Minimum stem length = 3
                    if(stem.length() >= 3)
                        return stem;
                    break;
                case 'C': //Minimum stem length = 4
                    if(stem.length() >= 4)
                        return stem;
                    break;
                case 'D'://Minimum stem length = 5
                    if(stem.length() >= 5)
                        return stem;
                    break;
                case 'E'://Do not remove ending after e
                    if(stem.endsWith("e"))
                        break;
                    return stem;
                case 'F'://Minimum stem length = 3 and do not remove ending after e
                    if(stem.endsWith("e") || stem.length() < 3)
                        break;
                    return stem;
                case 'G'://Minimum stem length = 3 and remove ending only after f
                    if(stem.endsWith("f") && stem.length() >= 3)
                        return stem;
                    break;
                case 'H'://Remove ending only after t or ll
                    if(stem.endsWith("t") || stem.endsWith("ll"))
                        return stem;
                    break;
                case 'I'://Do not remove ending after o or e
                    if(stem.endsWith("o") || stem.endsWith("e"))
                        break;
                    return stem;
                case 'J': //Do not remove ending after a or e
                    if(stem.endsWith("a") ||  stem.endsWith("e"))
                        break;
                    return stem;
                case 'K'://Minimum stem length = 3 and remove ending only after l, i or u*e
                    if(stem.length() >= 3 && stem.matches(".*(i|u.e|l)$"))
                        return stem;
                    break;
                case 'L'://Do not remove ending after u, x or s, unless s follows o
                    if(stem.endsWith("os"))
                        return stem;
                    else if(stem.matches(".*(u|x|s)$"))
                        break;
                    return stem;
                case 'M'://Do not remove ending after a, c, e or m
                    if(stem.endsWith("a") || stem.endsWith("c") || stem.endsWith("e") || stem.endsWith("m"))
                        break;
                    else
                        return stem;
                case 'N'://Minimum stem length = 4 after s**, elsewhere = 3
                    if (stem.matches(".*s..$"))
                        if (stem.length() >= 4)
                            return stem;
                        else
                            break;
                    else if (stem.length() >= 3)
                        return stem;
                    break;
                case 'O'://Remove ending only after l or i
                    if(stem.endsWith("l") || stem.endsWith("i"))
                        return stem;
                    break;
                case 'P'://Do not remove ending after c
                    if(stem.endsWith("e"))
                        break;
                    return stem;
                case 'Q'://Minimum stem length = 3 and do not remove ending after l or n
                    if(stem.length() < 3 || stem.endsWith("l") || stem.endsWith("n"))
                        break;
                    return stem;
                case 'R'://Remove ending only after n or r
                    if(stem.endsWith("n") || stem.endsWith("r"))
                        return stem;
                    break;
                case 'S'://Remove ending only after dr or t, unless t follows t
                    if(stem.endsWith("dr") || (stem.endsWith("t") && !stem.endsWith("tt")))
                        return stem;
                    break;
                case 'T'://Remove ending only after s or t, unless t follows o
                    if(stem.endsWith("s") || (stem.endsWith("t") && !stem.endsWith("ot"))) 
                        return stem;
                    break;
                case 'U'://Remove ending only after l, m, n or r
                    if(stem.endsWith("l") || stem.endsWith("m") || stem.endsWith("n") || stem.endsWith("r"))
                        return stem;
                    break;
                case 'V'://Remove ending only after c
                    if(stem.endsWith("c"))
                        return stem;
                    break;
                case 'W'://Do not remove ending after s or u
                    if(stem.endsWith("s") || stem.endsWith("u"))
                        break;
                    return stem;
                case 'X'://Remove ending only after l, i or u*e
                    if(stem.matches(".*(l|i|u.e)$"))
                        return stem;
                    break;
                case 'Y'://Remove ending only after in
                    if(stem.endsWith("in"))
                        return stem;
                    break;
                case 'Z'://Do not remove ending after f
                    if(stem.endsWith("f"))
                        break;
                    return stem;
                case 'a'://AA: Remove ending only after d, f, ph, th, l, er, or, es or t
                    if(stem.matches(".*(d|f|ph|th|l|er|or|es|t)$"))
                        return stem;
                    break;
                case 'b'://BB: Minimum stem length = 3 and do not remove ending after met or ryst
                    if(stem.length() < 3 || stem.endsWith("met") || stem.endsWith("ryst"))
                        break;
                    return stem;
                case 'c'://CC: Remove ending only after l
                    if(stem.endsWith("l"))
                        return stem;
                    break;
            }
            
            
        }
        
        return word;
    }
    
    private static String fixStem(String stem)
    {
        //Rule 1 remove one of double b, d, g, l, m, n, p, r, s, t
        char lastChar = stem.charAt(stem.length()-1);
        stem = stem.replaceFirst("(dd|bb|gg|ll|mm|nn|pp|rr|ss|tt)$", "" + lastChar);
        //Rule 2
        stem = stem.replaceFirst("iev$", "ief");
        //Rule 3
        stem = stem.replaceFirst("uct$", "uc");
        //Rule 4
        stem = stem.replaceFirst("umpt$", "um");
        //Rule 5
        stem = stem.replaceFirst("rpt$", "rb");
        //Rule 6
        stem = stem.replaceFirst("urs$", "ur");
        //Rule 7
        stem = stem.replaceFirst("istr$", "ister");
        //Rule 7a
        stem = stem.replaceFirst("metr$", "meter");
        //Rule 8
        stem = stem.replaceFirst("olv$", "olut");
        //Rule 9
        if(stem.endsWith("ul") && !stem.endsWith("aoiul"))
            stem = stem.replaceFirst("[^aoi]ul$", "l");
        //Rule 10
        stem = stem.replaceFirst("bex$", "bic");
        //Rule 11
        stem = stem.replaceFirst("dex$", "dic");
        //Rule 12
        stem = stem.replaceFirst("pex$", "pic");
        //Rule 13
        stem = stem.replaceFirst("tex$", "tic");
        //Rule 14
        stem = stem.replaceFirst("ax$", "ac");
        //Rule 15
        stem = stem.replaceFirst("ex$", "ec");
        //Rule 16
        stem = stem.replaceFirst("ix$", "ic");
        //Rule 17
        stem = stem.replaceFirst("lux$", "luc");
        //Rule 18
        stem = stem.replaceFirst("uad$", "uas");
        //Rule 19
        stem = stem.replaceFirst("vad$", "vas");
        //Rule 20
        stem = stem.replaceFirst("cid$", "cis");
        //Rule 21
        stem = stem.replaceFirst("lid$", "lis");
        //Rule 22
        stem = stem.replaceFirst("erid$", "eris");
        //Rule 23
        stem = stem.replaceFirst("pand$", "pans");
        //Rule 24
        if(stem.endsWith("end") && !stem.endsWith("send"))
            stem = stem.replaceFirst("[^s]end$", "ens");
        //Rule 25
        stem = stem.replaceFirst("ond$", "ons");
        //Rule 26
        stem = stem.replaceFirst("lud$", "lus");
        //Rule 27
        stem = stem.replaceFirst("rud$", "rus");
        //Rule 28
        stem = stem.replaceFirst("[^pt]her$", "hes");
        //Rule 29
        stem = stem.replaceFirst("mit$", "mis");
        //Rule 30
        if(stem.endsWith("ent") && !stem.endsWith("ment"))
            stem = stem.replaceFirst("[^m]ent$", "ens");
        //Rule 31
        stem = stem.replaceFirst("ert$", "ers");
        //Rule 32
        if(stem.endsWith("et") && !stem.endsWith("net"))
            stem = stem.replaceFirst("et$", "es");
        //Rule 33
        stem = stem.replaceFirst("yt$", "ys");
        //Rule 34
        stem = stem.replaceFirst("yz$", "ys");

        return stem;
    }
    public String stem(String word)
    {
        return fixStem(removeEnding(word));
    }
    
}
