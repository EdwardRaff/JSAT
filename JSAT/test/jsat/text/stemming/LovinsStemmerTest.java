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
public class LovinsStemmerTest
{
    
    public LovinsStemmerTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
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
     * Test of stem method, of class LovinsStemmer.
     */
    @Test
    public void testStem()
    {
        System.out.println("stem");
        String[] origSent = ("such an analysis can reveal features that are not easily visible "
                + "from the variations in the individual genes and can lead to a picture of "
                + "expression that is more biologically transparent and accessible to "
                + "interpretation").split(" ");
        LovinsStemmer instance = new LovinsStemmer();
        String[] expResult = ("such an analys can reve featur that ar not eas vis from th "
                + "vari in th individu gen and can lead to a pictur of expres that is mor "
                + "biolog transpar and acces to interpres").split(" ");
        
        for(int i = 0; i < origSent.length; i++)
            assertEquals(expResult[i], instance.stem(origSent[i]));
    }
    
}
