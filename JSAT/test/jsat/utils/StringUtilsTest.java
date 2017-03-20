package jsat.utils;

import java.util.Random;
import jsat.utils.random.RandomUtil;
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
public class StringUtilsTest
{
    
    public StringUtilsTest()
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
     * Test of parseInt method, of class StringUtils.
     */
    @Test
    public void testParseInt()
    {
        System.out.println("parseInt");
        Random rand = RandomUtil.getRandom();
        for(int radix = Character.MIN_RADIX; radix <= Character.MAX_RADIX; radix++)
        {
            for(int trials = 0; trials < 1000; trials++)
            {
                String preFix = "";
                String postFix = "";

                int prefixSize = rand.nextInt(3);
                int postFixSize = rand.nextInt(3);
                for(int i =0 ; i < prefixSize; i++)
                    preFix += Character.toString((char) rand.nextInt(128));
                for(int i =0 ; i < postFixSize; i++)
                    postFix += Character.toString((char) rand.nextInt(128));


                //first test a large int
                int truth = rand.nextInt();
                String test = preFix + Integer.toString(truth, radix) + postFix;

                assertEquals(truth, StringUtils.parseInt(test, prefixSize, test.length()-postFixSize, radix));
                
                //now a small one
                
                //first test a large int
                truth = rand.nextInt(10)*(rand.nextInt(1)*2-1);
                test = preFix + Integer.toString(truth, radix) + postFix;

                assertEquals(truth, StringUtils.parseInt(test, prefixSize, test.length()-postFixSize, radix));
            }
        }
    }
    
    @Test
    public void testParseDouble()
    {
        System.out.println("parseDouble");
        Random rand = new Random(42);
        
        String[] signOps = new String[]{"+", "-", ""};
        String[] Es = new String[]{"e", "E"};
        String[] zeros = new String[]{"","", "", "0", "00", "000", "0000"};
        
        double truth, attempt;
        String toTest;

        for (int trials = 0; trials < 10000; trials++)
        {
            String preFix = "";
            String postFix = "";

            int prefixSize = 0;//rand.nextInt(3);
            int postFixSize = 0;//rand.nextInt(3);
            for (int i = 0; i < prefixSize; i++)
                preFix += Character.toString((char) rand.nextInt(128));
            for (int i = 0; i < postFixSize; i++)
                postFix += Character.toString((char) rand.nextInt(128));

            //easy cases that should all be exact
            
            //[sign][val]
            toTest = signOps[rand.nextInt(3)] + zeros[rand.nextInt(zeros.length)] + rand.nextInt((int) Math.round(Math.pow(10, rand.nextInt(6)))) + "" + rand.nextInt((int) Math.round(Math.pow(10, rand.nextInt(6))));
            truth = Double.parseDouble(toTest);
            attempt = StringUtils.parseDouble(toTest, 0, toTest.length());
            assertRelativeEquals(truth, attempt);
            
            //[sign][val].[val]
            toTest = signOps[rand.nextInt(3)] + zeros[rand.nextInt(zeros.length)] + rand.nextInt((int) Math.round(Math.pow(10, rand.nextInt(6)))) + "." + zeros[rand.nextInt(zeros.length)] + rand.nextInt((int) Math.round(Math.pow(10, rand.nextInt(6))));
            truth = Double.parseDouble(toTest);
            attempt = StringUtils.parseDouble(toTest, 0, toTest.length());
            assertRelativeEquals(truth, attempt);
            
            //[sign][val][eE][sign][val]
            toTest = signOps[rand.nextInt(3)] + zeros[rand.nextInt(zeros.length)] + rand.nextInt((int) Math.round(Math.pow(10, rand.nextInt(6)))) + Es[rand.nextInt(2)] + signOps[rand.nextInt(3)] + zeros[rand.nextInt(zeros.length)] + rand.nextInt((int) Math.round(Math.pow(10, rand.nextInt(2))));
            truth = Double.parseDouble(toTest);
            attempt = StringUtils.parseDouble(toTest, 0, toTest.length());
            assertRelativeEquals(truth, attempt);
            
            //harder cases, may have nonsense values (over flow to Inf/NegInf, underflows to 0)
            
            //[sign][val]
            //XXX This code generates a random signed integer and then computes the absolute value of that random integer. If the number returned by the random number generator is Integer.MIN_VALUE, then the result will be negative as well (since Math.abs(Integer.MIN_VALUE) == Integer.MIN_VALUE). (Same problem arised for long values as well).
            toTest = signOps[rand.nextInt(3)].replace("-", "") + zeros[rand.nextInt(zeros.length)] + Math.max(Math.abs(rand.nextLong()), 0);
            truth = Double.parseDouble(toTest);
            attempt = StringUtils.parseDouble(toTest, 0, toTest.length());
            assertRelativeEquals(truth, attempt);
            
            //[sign][val].[val]
            toTest = signOps[rand.nextInt(3)] + rand.nextInt(Integer.MAX_VALUE) + zeros[rand.nextInt(zeros.length)] + "." + zeros[rand.nextInt(zeros.length)] + rand.nextInt(Integer.MAX_VALUE) ;
            truth = Double.parseDouble(toTest);
            attempt = StringUtils.parseDouble(toTest, 0, toTest.length());
            assertRelativeEquals(truth, attempt);
            
            //[sign][val][eE][sign][val]
            toTest = signOps[rand.nextInt(3)] + zeros[rand.nextInt(zeros.length)] + rand.nextInt((int) Math.round(Math.pow(10, rand.nextInt(8)))) + Es[rand.nextInt(2)] + signOps[rand.nextInt(3)] + zeros[rand.nextInt(zeros.length)] + rand.nextInt(450);
            truth = Double.parseDouble(toTest);
            attempt = StringUtils.parseDouble(toTest, 0, toTest.length());
            assertRelativeEquals(truth, attempt);
        }

    }

    protected void assertRelativeEquals(double truth, double attempt)
    {
        String message = "Expteced " + truth + " but was " + attempt;
        if (Double.isNaN(truth))
            assertTrue(message, Double.isNaN(attempt));
        else if (Double.isInfinite(truth))
        {
            assertTrue(message, Double.isInfinite(attempt));
            assertTrue(Double.doubleToRawLongBits(truth) == Double.doubleToRawLongBits(attempt));//get the signs right
        }
        else
        {
            double relDiff = Math.abs(truth - attempt) / (Math.max(Math.abs(Math.max(truth, attempt)), 1e-14));
            assertEquals(message, 0, relDiff, 1e-14);
        }
    }
    
}
