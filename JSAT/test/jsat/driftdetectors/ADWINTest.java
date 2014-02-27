package jsat.driftdetectors;

import java.util.List;
import java.util.Random;
import jsat.distributions.Normal;
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
public class ADWINTest
{
    
    public ADWINTest()
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

    @Test
    public void testAddSample_double_GenericType()
    {
        System.out.println("addSample");
        
        Normal normal_0_1 = new Normal(0, 1);
        Normal normal_1_1 = new Normal(1, 1);
        Normal normal_2_1 = new Normal(2, 1);
        
        Random rand = new XORWOW(123);
        ADWIN<Integer> adwin = new ADWIN<Integer>(0.01, 100);
        //Start should not ever observe a false positive
        for(int i = 0; i < 400; i++)
            if(adwin.addSample(normal_0_1.invCdf(rand.nextDouble()), i))
                assertFalse(adwin.isDrifting());
        assertEquals(normal_0_1.mean(), adwin.getMean(), 0.25);//lots of samples, should be close
        assertEquals(400, adwin.getWidnowLength());
        
        //drift up
        {
            boolean drifted = false;
            for(int i = 0; i < 400; i++)
                if(adwin.addSample(normal_1_1.invCdf(rand.nextDouble()), i))
                {
                    assertTrue(adwin.getDriftAge() < i+30);
                    assertFalse(drifted);
                    assertEquals(normal_0_1.mean(), adwin.getOldMean(), 0.25);
                    assertEquals(normal_1_1.mean(), adwin.getNewMean(), 1.5);//few samples, lose bound
                    
                    List<Integer> driftedHistory = adwin.getDriftedHistory();
                    assertEquals(Math.min(adwin.getMaxHistory(), adwin.getDriftAge()), driftedHistory.size());
                    for(int j = 1; j <driftedHistory.size(); j++)
                    {
                        assertTrue(driftedHistory.get(j-1) > driftedHistory.get(j));
                        if(driftedHistory.get(j) == 0)
                            break;
                    }
                    
                    adwin.driftHandled(true);
                    drifted = true;
                }
            assertTrue(drifted);
            assertEquals(normal_1_1.mean(), adwin.getMean(), 0.35);
        }
         
        adwin = adwin.clone();
        //drift and drop NEW values this time, ie: stop ourselves from going back to where we were
        {
            boolean drifted = false;
            for(int i = 0; i < 400 && !drifted; i++)
                if(adwin.addSample(normal_0_1.invCdf(rand.nextDouble()), i))
                {
                    assertTrue(adwin.getDriftAge() < i+30);
                    assertFalse(drifted);
                    
                    assertEquals(normal_1_1.mean(), adwin.getOldMean(), 0.35);
                    assertEquals(normal_0_1.mean(), adwin.getNewMean(), 1.5);//few samples, lose bound
                    
                    List<Integer> driftedHistory = adwin.getDriftedHistory();
                    assertEquals(Math.min(adwin.getMaxHistory(), adwin.getDriftAge()), driftedHistory.size());
                    for(int j = 1; j <driftedHistory.size(); j++)
                    {
                        assertTrue(driftedHistory.get(j-1) > driftedHistory.get(j));
                        if(driftedHistory.get(j) == 0)
                            break;
                    }
                    
                    adwin.driftHandled(false);
                    drifted = true;
                }
            assertEquals(normal_1_1.mean(), adwin.getMean(), 1.5);//few samples, lose bound
        }
        
        //drift up again
        {
            boolean drifted = false;
            for(int i = 0; i < 400; i++)
                if(adwin.addSample(normal_2_1.invCdf(rand.nextDouble()), i))
                {
                    assertTrue(adwin.getDriftAge() < i+30);
                    assertFalse(drifted);
                    assertEquals(normal_1_1.mean(), adwin.getOldMean(), 0.35);
                    assertEquals(normal_2_1.mean(), adwin.getNewMean(), 1.5);//few samples, lose bound
                    
                    List<Integer> driftedHistory = adwin.getDriftedHistory();
                    assertEquals(Math.min(adwin.getMaxHistory(), adwin.getDriftAge()), driftedHistory.size());
                    for(int j = 1; j <driftedHistory.size(); j++)
                    {
                        assertTrue(driftedHistory.get(j-1) > driftedHistory.get(j));
                        if(driftedHistory.get(j) == 0)
                            break;
                    }
                    
                    adwin.driftHandled();
                    drifted = true;
                }
            assertTrue(drifted);
            assertEquals(normal_2_1.mean(), adwin.getMean(), 0.35);
        }
    }
}
