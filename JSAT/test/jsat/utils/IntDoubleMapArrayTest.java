
package jsat.utils;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * 
 *
 * @author Edward Raff
 */
public class IntDoubleMapArrayTest
{
    private static final int TEST_SIZE = 2000;
    private static final int MAX_INDEX = 10000;
    Random rand;
    
    public IntDoubleMapArrayTest()
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
        rand = RandomUtil.getRandom();
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of put method, of class IntDoubleMap.
     */
    @Test
    public void testPut_Integer_Double()
    {
        System.out.println("put");
        Integer key = null;
        Double value = null;
        
        Map<Integer, Double> truthMap = new HashMap<Integer, Double>();
        IntDoubleMapArray idMap = new IntDoubleMapArray();
        
        for(int i = 0; i < TEST_SIZE; i++)
        {
            key = rand.nextInt(MAX_INDEX);
            value = Double.valueOf(rand.nextInt(1000));
            
            Double prevTruth = truthMap.put(key, value);
            Double prev = idMap.put(key, value);
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), idMap.size());
        }
        
        assertEntriesAreEqual(truthMap, idMap);
        
        //will call the iterator remove on everythin
        removeEvenByIterator(idMap.entrySet().iterator());
        removeEvenByIterator(truthMap.entrySet().iterator());
        
        assertEntriesAreEqual(truthMap, idMap);
        
        for(Entry<Integer, Double> entry : idMap.entrySet())
            entry.setValue(1.0);
        for(Entry<Integer, Double> entry : truthMap.entrySet())
            entry.setValue(1.0);
        
        assertEntriesAreEqual(truthMap, idMap);
        
        
        ///again, random keys - and make them colide
        
        truthMap = new HashMap<Integer, Double>();
        idMap = new IntDoubleMapArray();
        
        for(int i = 0; i < TEST_SIZE; i++)
        {
            key = Integer.valueOf(rand.nextInt(50000));
            value = Double.valueOf(rand.nextInt(1000));
            
            Double prevTruth = truthMap.put(key, value);
            Double prev = idMap.put(key, value);
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), idMap.size());
        }
        
        assertEntriesAreEqual(truthMap, idMap);
        
        //will call the iterator remove on everythin
        removeEvenByIterator(idMap.entrySet().iterator());
        removeEvenByIterator(truthMap.entrySet().iterator());
        
        assertEntriesAreEqual(truthMap, idMap);
        
        for(Entry<Integer, Double> entry : idMap.entrySet())
            entry.setValue(1.0);
        for(Entry<Integer, Double> entry : truthMap.entrySet())
            entry.setValue(1.0);
        
        assertEntriesAreEqual(truthMap, idMap);
    }

    private void removeEvenByIterator(Iterator<Entry<Integer, Double>> iterator)
    {
        while(iterator.hasNext())
        {
            Entry<Integer, Double> entry = iterator.next();
            if(entry.getKey() % 2 == 0)
                iterator.remove();
        }
    }

    private void assertEntriesAreEqual(Map<Integer, Double> truthMap, IntDoubleMapArray idMap)
    {
        assertEquals(truthMap.size(), idMap.size());
        
        Map<Integer, Double> copy = new HashMap<Integer, Double>();
        
        for(Entry<Integer, Double> entry : truthMap.entrySet())
            assertEquals(entry.getValue(), idMap.get(entry.getKey()));
        
        int observed = 0;
        for(Entry<Integer, Double> entry : idMap.entrySet())
        {
            copy.put(entry.getKey(), entry.getValue());
            observed++;
            assertTrue(truthMap.containsKey(entry.getKey()));
            assertEquals(truthMap.get(entry.getKey()), entry.getValue());
        }
        assertEquals(truthMap.size(), observed);
        
        //make sure we put every value into the copy!
        for(Entry<Integer, Double> entry : truthMap.entrySet())
            assertEquals(truthMap.get(entry.getKey()), copy.get(entry.getKey()));
    }

    /**
     * Test of put method, of class IntDoubleMap.
     */
    @Test
    public void testPut_int_double()
    {
        System.out.println("put");
        int key;
        double value;
        
        Map<Integer, Double> truthMap = new HashMap<Integer, Double>();
        IntDoubleMapArray idMap = new IntDoubleMapArray();
        
        for(int i = 0; i < TEST_SIZE; i++)
        {
            key = rand.nextInt(MAX_INDEX);
            value = Double.valueOf(rand.nextInt(1000));
            
            Double prevTruth = truthMap.put(key, value);
            Double prev = idMap.put(key, value);
            if(prev.isNaN())
                prev = null;
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), idMap.size());
        }
        
        assertEntriesAreEqual(truthMap, idMap);
        
        //will call the iterator remove on everythin
        removeEvenByIterator(idMap.entrySet().iterator());
        removeEvenByIterator(truthMap.entrySet().iterator());
        
        assertEntriesAreEqual(truthMap, idMap);
        
        for(Entry<Integer, Double> entry : idMap.entrySet())
            entry.setValue(1.0);
        for(Entry<Integer, Double> entry : truthMap.entrySet())
            entry.setValue(1.0);
        
        assertEntriesAreEqual(truthMap, idMap);
        
        
        ///again, random keys - and make them colide
        
        truthMap = new HashMap<Integer, Double>();
        idMap = new IntDoubleMapArray();
        
        for(int i = 0; i < TEST_SIZE; i++)
        {
            key = Integer.valueOf(rand.nextInt(50000));
            value = Double.valueOf(rand.nextInt(1000));
            
            Double prevTruth = truthMap.put(key, value);
            Double prev = idMap.put(key, value);
            if(prev.isNaN())
                prev = null;
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), idMap.size());
        }
        
        assertEntriesAreEqual(truthMap, idMap);
        
        //will call the iterator remove on everythin
        removeEvenByIterator(idMap.entrySet().iterator());
        removeEvenByIterator(truthMap.entrySet().iterator());
        
        assertEntriesAreEqual(truthMap, idMap);
        
        for(Entry<Integer, Double> entry : idMap.entrySet())
            entry.setValue(1.0);
        for(Entry<Integer, Double> entry : truthMap.entrySet())
            entry.setValue(1.0);
        
        assertEntriesAreEqual(truthMap, idMap);
    }

    /**
     * Test of increment method, of class IntDoubleMap.
     */
    @Test
    public void testIncrement()
    {
        System.out.println("increment");
        Integer key = null;
        Double value = null;
        
        Map<Integer, Double> truthMap = new HashMap<Integer, Double>();
        IntDoubleMapArray idMap = new IntDoubleMapArray();
        
        int MAX = TEST_SIZE/2;
        int times =0;
        for(int i = 0; i < MAX; i++)
        {
            key = Integer.valueOf(rand.nextInt(MAX));
            value = Double.valueOf(rand.nextInt(1000));
            if(truthMap.containsKey(key))
                times++;
            Double prevTruth = truthMap.put(key, value);
            Double prev = idMap.put(key, value);
            
            if(prev == null && prevTruth != null)
                System.out.println(idMap.put(key, value));
            assertEquals(prevTruth, prev);
            if(idMap.size() != truthMap.size())
            {
                System.out.println();
            }
            assertEquals(truthMap.size(), idMap.size());
        }
        
        assertEntriesAreEqual(truthMap, idMap);
        
        for(Entry<Integer, Double> entry : truthMap.entrySet())
        {
            double delta = Double.valueOf(rand.nextInt(100));
            double trueNewValue =entry.getValue()+delta;
            entry.setValue(trueNewValue);
            double newValue = idMap.increment(entry.getKey(), delta);
            assertEquals(trueNewValue, newValue, 0.0);
        }
        
        for(int i = MAX; i < MAX*2; i++)
        {
            key = Integer.valueOf(i);//force it to be new
            value = Double.valueOf(rand.nextInt(1000));
            
            truthMap.put(key, value);
            double ldNew =idMap.increment(key, value);
            assertEquals(value.doubleValue(), ldNew, 0.0);
        }
        
        assertEntriesAreEqual(truthMap, idMap);
    }

    /**
     * Test of remove method, of class IntDoubleMap.
     */
    @Test
    public void testRemove_Object()
    {
        System.out.println("remove");
        Integer key = null;
        Double value = null;
        
        Map<Integer, Double> truthMap = new HashMap<Integer, Double>();
        IntDoubleMapArray ldMap = new IntDoubleMapArray();
        
        int MAX = TEST_SIZE/2;
        for(int i = 0; i < MAX; i++)
        {
            key = Integer.valueOf(rand.nextInt(MAX));
            value = Double.valueOf(rand.nextInt(1000));
            
            Double prevTruth = truthMap.put(key, value);
            Double prev = ldMap.put(key, value);
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), ldMap.size());
        }
        
        assertEntriesAreEqual(truthMap, ldMap);
        
        
        for(int i = 0; i < MAX/4; i++)
        {
            key = Integer.valueOf(rand.nextInt(MAX));
            
            Double prevTruth = truthMap.remove(key);
            Double prev = ldMap.remove(key);
            if(prevTruth == null && prev != null)
                prev = ldMap.remove(key);
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), ldMap.size());
        }
        
        
        assertEntriesAreEqual(truthMap, ldMap);
    }

    /**
     * Test of remove method, of class IntDoubleMap.
     */
    @Test
    public void testRemove_int()
    {
        System.out.println("remove");
        Integer key = null;
        Double value = null;
        
        Map<Integer, Double> truthMap = new HashMap<Integer, Double>();
        IntDoubleMapArray idMap = new IntDoubleMapArray();
        
        int MAX = TEST_SIZE/2;
        for(int i = 0; i < MAX; i++)
        {
            key = Integer.valueOf(rand.nextInt(MAX));
            value = Double.valueOf(rand.nextInt(1000));
            
            Double prevTruth = truthMap.put(key, value);
            Double prev = idMap.put(key, value);
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), idMap.size());
        }
        
        assertEntriesAreEqual(truthMap, idMap);
        
        
        for(int i = 0; i < MAX/4; i++)
        {
            key = Integer.valueOf(rand.nextInt(MAX));
            
            Double prevTruth = truthMap.remove(key);
            Double prev = idMap.remove(key.intValue());
            if(prev.isNaN())
                prev = null;
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), idMap.size());
        }
        
        
        assertEntriesAreEqual(truthMap, idMap);
    }

    /**
     * Test of containsKey method, of class IntDoubleMap.
     */
    @Test
    public void testContainsKey_Object()
    {
        System.out.println("containsKey");
        Integer key = null;
        Double value = null;
        
        Map<Integer, Double> truthMap = new HashMap<Integer, Double>();
        IntDoubleMapArray idMap = new IntDoubleMapArray();
        
        int MAX = TEST_SIZE/2;
        for(int i = 0; i < MAX; i++)
        {
            key = Integer.valueOf(rand.nextInt(MAX));
            value = Double.valueOf(rand.nextInt(1000));
            
            Double prevTruth = truthMap.put(key, value);
            Double prev = idMap.put(key, value);
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), idMap.size());
        }
        
        assertEntriesAreEqual(truthMap, idMap);
        
        for(Integer keyInSet : truthMap.keySet())
            assertTrue(idMap.containsKey(keyInSet));
        
        for(long i = MAX+1; i < MAX*2; i++)
            assertFalse(idMap.containsKey(Long.valueOf(i)));
    }

    /**
     * Test of containsKey method, of class IntDoubleMap.
     */
    @Test
    public void testContainsKey_int()
    {
        System.out.println("containsKey");
        Integer key = null;
        Double value = null;
        
        Map<Integer, Double> truthMap = new HashMap<Integer, Double>();
        IntDoubleMapArray idMap = new IntDoubleMapArray();
        
        int MAX = TEST_SIZE/2;
        for(int i = 0; i < MAX; i++)
        {
            key = Integer.valueOf(rand.nextInt(MAX));
            value = Double.valueOf(rand.nextInt(1000));
            
            Double prevTruth = truthMap.put(key, value);
            Double prev = idMap.put(key, value);
            assertEquals(prevTruth, prev);
            assertEquals(truthMap.size(), idMap.size());
        }
        
        assertEntriesAreEqual(truthMap, idMap);
        
        for(Integer keyInSet : truthMap.keySet())
            assertTrue(idMap.containsKey(keyInSet.intValue()));
        
        for(long i = MAX+1; i < MAX*2; i++)
            assertFalse(idMap.containsKey(i));
    }

}
