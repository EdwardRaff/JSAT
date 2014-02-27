
package jsat.driftdetectors;

import java.util.List;
import java.util.Random;
import jsat.utils.random.XORWOW;
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
public class DDMTest
{
    
    public DDMTest()
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
     * Test of addSample method, of class DDM.
     */
    @Test
    public void testAddSample_double_GenericType()
    {
        System.out.println("addSample");
        Random rand = new XORWOW(123);
        DDM<Integer> ddm = new DDM<Integer>();
        //Start should not ever observe a false positive
        for(int i = 0; i < 50; i++)
            if(ddm.addSample(rand.nextDouble() < 0.8, i))
                assertFalse(ddm.isDrifting());
        assertEquals(0.8, ddm.getSuccessRate(), 0.15);
        
        //Increase ina ccuracy, still shouldn't trigger
        for(int i = 0; i < 50; i++)
            if(ddm.addSample(rand.nextDouble() < 0.9, i))
                assertFalse(ddm.isDrifting());
        
        boolean seenWarning = false;
        boolean seenDrift = false;
        //Now we should see an error
        for(int i = 1; i <= 100; i++)
            if(ddm.addSample(rand.nextDouble() < 0.7, -i))//negative to diferentiate from before
            {
                if(ddm.isWarning())
                {
                    seenWarning = true;
                }
                else if(ddm.isDrifting())
                {
                    assertTrue(i < 40);//got to detect it fast enought
                    assertFalse(seenDrift);
                    assertTrue(seenWarning);
                    List<Integer> drifted = ddm.getDriftedHistory();
                    //make sure there is some history
                    assertTrue(drifted.size() > 5);
                    assertTrue(ddm.getDriftAge() > 5);
                    assertTrue(ddm.getDriftAge() == drifted.size());
                    for(int j = 0; j < drifted.size(); j++)
                    {
                        assertTrue(drifted.get(j) < 0);
                        if(j < drifted.size()-1)
                            assertTrue(drifted.get(j) < drifted.get(j+1));
                    }
                    ddm.driftHandled();
                    seenDrift = true;
                }
            }
        
        assertEquals(0.7, ddm.getSuccessRate(), 0.15);
    }

}
