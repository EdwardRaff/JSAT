
package jsat.text.stemming;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class PaiceHuskStemmerTest
{
    
    public static String[] original;
    public static String[] expected;
    
    public PaiceHuskStemmerTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        original = new String[]
        {
            "am",
            "is",
            "are",
            "be",
            "being",
            "been",
            "have",
            "has",
            "having",
            "had",
            "do",
            "does",
            "doing",
            "did",
            "hello",
            "ear",
            "owel",
            "owed",
            "police",
            "policy",
            "dog",
            "cat",
            "sinner",
            "sinners",
            "discrimination",
            "counties",
            "county",
            "country",
            "countries",
            "fighters",
            "civilization",
            "civilizations",
            "currencies",
            "constructed",
            "constructing",
            "stemming",
            "stemmer",
            "connection",
            "connections",
            "connective",
            "connected",
            "connecting",
            "fortification",
            "electricity",
            "fantastically",
            "contemplative",
            "conspirator",
            "relativity",
            "instinctively",
            "incapability",
            "charitably",
            "famously",
        };
        
        expected = new String[]
        {
            "am",
            "is",
            "ar",
            "be",
            "being",
            "been",
            "hav",
            "has",
            "hav",
            "had",
            "do",
            "doe",
            "doing",
            "did",
            "hello",
            "ear",
            "owel",
            "ow",
            "pol",
            "policy",
            "dog",
            "cat",
            "sin",
            "sin",
            "discrimin",
            "county",
            "county",
            "country",
            "country",
            "fight",
            "civil",
            "civil",
            "cur",
            "construct",
            "construct",
            "stem",
            "stem",
            "connect",
            "connect",
            "connect",
            "connect",
            "connect",
            "fort",
            "elect",
            "fantast",
            "contempl",
            "conspir",
            "rel",
            "instinct",
            "incap",
            "charit",
            "fam",
        };
    }
    
    @AfterClass
    public static void tearDownClass()
    {
    }
    
    @Before
    public void setUp()
    {
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of stem method, of class PaiceHuskStemmer.
     */
    @Test
    public void testStem()
    {
        System.out.println("stem");
        
        PaiceHuskStemmer stemmer = new PaiceHuskStemmer();
       
        for (int i = 0; i < original.length; i++)
            assertEquals("Stemming results incorrect for \"" + original[i] + "\"", expected[i], stemmer.stem(original[i]));
    }
}
