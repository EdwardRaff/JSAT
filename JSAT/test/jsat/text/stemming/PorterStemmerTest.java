/*
 * Copyright (C) 2015 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.text.stemming;

import java.util.LinkedHashMap;
import java.util.Map;
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
public class PorterStemmerTest
{
    private static final Map<String, String> testCases = new LinkedHashMap<String, String>();
    
    public PorterStemmerTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
        testCases.put("a", "a");
        testCases.put("caresses", "caress");
        testCases.put("ponies", "poni");
        testCases.put("ties", "ti");
        testCases.put("carress", "carress");
        testCases.put("cats", "cat");
        testCases.put("feed", "feed");
        testCases.put("agreed", "agre");
        testCases.put("plastered", "plaster");
        testCases.put("bled", "bled");
        testCases.put("motoring", "motor");
        testCases.put("sing", "sing");
        testCases.put("conflated", "conflat");
        testCases.put("troubling", "troubl");
        testCases.put("sized", "size");
        testCases.put("falling", "fall");
        testCases.put("happy", "happi");
        testCases.put("sky", "sky");
        testCases.put("relational", "relat");
        testCases.put("conditional", "condit");
        testCases.put("vileli", "vile");
        testCases.put("analogousli", "analog");
        testCases.put("vietnamization", "vietnam");
        testCases.put("predication", "predic");
        testCases.put("operator", "oper");
        testCases.put("feudalism", "feudal");
        testCases.put("decisiveness", "decis");
        testCases.put("hopefulness", "hope");
        testCases.put("callousness", "callous");
        testCases.put("formaliti", "formal");
        testCases.put("sensitiviti", "sensit");
        testCases.put("sensibiliti", "sensibl");
        testCases.put("triplicate", "triplic");
        testCases.put("formative", "form");
        testCases.put("formalize", "formal");
        testCases.put("electriciti", "electr");
        testCases.put("electrical", "electr");
        testCases.put("hopeful", "hope");
        testCases.put("goodness", "good");
        testCases.put("revival", "reviv");
        testCases.put("allowance", "allow");
        testCases.put("inference", "infer");
        testCases.put("airliner", "airlin");
        testCases.put("gyroscopic", "gyroscop");
        testCases.put("adjustable", "adjust");
        testCases.put("defensible", "defens");
        testCases.put("irritant", "irrit");
        testCases.put("replacement", "replac");
        testCases.put("adjustment", "adjust");
        testCases.put("dependent", "depend");
        testCases.put("adoption", "adopt");
        testCases.put("homologou", "homolog");
        testCases.put("communism", "commun");
        testCases.put("activate", "activ");
        testCases.put("angulariti", "angular");
        testCases.put("homologous", "homolog");
        testCases.put("effective", "effect");
        testCases.put("bowdlerize", "bowdler");
        testCases.put("probate", "probat");
        testCases.put("rate", "rate");
        testCases.put("cease", "ceas");
        testCases.put("controll", "control");
        testCases.put("roll", "roll");
        testCases.put("ions", "ion");
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
     * Test of stem method, of class PorterStemmer.
     */
    @Test
    public void testStem()
    {
        System.out.println("stem");
        PorterStemmer instance = new PorterStemmer();
        for(Map.Entry<String, String> entry : testCases.entrySet())
            assertEquals("Looking for '" + entry.getValue() + "' from '" + entry.getKey() + "'", entry.getValue(), instance.stem(entry.getKey()));
    }
    
}
