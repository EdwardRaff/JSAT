/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.distributions;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class RayleighTest
{
    double[] range = new double[]
    {
        -3., -2.75, -2.5, -2.25, -2., -1.75, 
        -1.5, -1.25, -1., -0.75, -0.5, -0.25,
        0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 
        1.75, 2., 2.25, 2.5, 2.75, 3.
    };
    
    public RayleighTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
    }
    
    @Before
    public void setUp()
    {
    }

    /**
     * Test of setScale method, of class Rayleigh.
     */
    @Test
    public void testSetScale()
    {
        System.out.println("setSig");
        Rayleigh instance = new Rayleigh(0.5);
        instance.setScale(1);
        instance.setScale(0.5);
        
        try
        {
            instance.setScale(0);
            fail("Invalid value, should err");
        }
        catch(Exception ex)
        {
            
        }
        
        try
        {
            instance.setScale(Double.POSITIVE_INFINITY);
            fail("Invalid value, should err");
        }
        catch(Exception ex)
        {
            
        }
    }


    /**
     * Test of pdf method, of class Rayleigh.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        ContinuousDistribution instance = null;
        
        double[] param0p5 = new double[]{0,0.8824969025845955,1.2130613194252668,0.9739574020750492,0.5413411329464508,0.2196846681170371,0.06665397922945383,0.015312437827280196,0.002683701023220095,0.0003605876765365596,0.00003726653172078671,2.9695363536993156e-6,1.8275975693655156e-7,8.699061918680616e-9,3.2056287839037744e-10,9.152905016407986e-12,2.026266487855068e-13,3.4798519123791473e-15,4.638162796478966e-17,4.799711183758816e-19,3.857499695927835e-21,2.4084408226737518e-23,1.1684402949294009e-25,4.40560158952444e-28,1.2912446784050733e-30};
        double[] param2 = new double[]{0,0.06201362114126522,0.12115415430954302,0.1747692173174114,0.22062422564614886,0.2570554882495827,0.2830648507458777,0.2983490786457773,0.3032653298563167,0.29874149495738167,0.28614585110725893,0.26713371266475033,0.2434893505187623,0.21697961612140398,0.18923202097615138,0.16164527240039328,0.1353352832366127,0.11111518886956405,0.08950444730800615,0.07075994102985847,0.054921167029259275,0.041861916287831195,0.03134199871496697,0.023054207083316413,0.016663494807363458};
        double[] param12 = new double[]{0,0.001735734391765489,0.0034692094482488734,0.005198170734934769,0.006920373603960896,0.0086335880503019,0.010335603523544203,0.012024233680718716,0.013697321065887724,0.015352741702467192,0.01698840958460465,0.018602281054324005,0.020192359051590503,0.0217566972249391,0.02329340389084611,0.02480064583060355,0.026276651914076816,0.027719716540384763,0.029128202886235237,0.03050054595337441,0.03183525540736211,0.03313091820066401,0.03438620097385169,0.0355998522295194,0.03677070427435814};
        
        instance = new Rayleigh(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(param0p5[i], instance.pdf(range[i]+3), 1e-10);
        instance = new Rayleigh(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.pdf(range[i]+3), 1e-10);
        instance = new Rayleigh(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.pdf(range[i]+3), 1e-10);
    }

    /**
     * Test of cdf method, of class Rayleigh.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        ContinuousDistribution instance = null;
        
        double[] param0p5 = new double[]{0,0.11750309741540454,0.3934693402873666,0.6753475326416503,0.8646647167633873,0.9560630663765926,0.9888910034617577,0.9978125088818172,0.9996645373720975,0.999959934702607,0.999996273346828,0.9999997300421497,0.9999999847700203,0.9999999993308414,0.9999999999771026,0.9999999999993898,0.9999999999999873,0.9999999999999998,1.,1.,1.,1.,1.,1.,1.};
        double[] param2 = new double[]{0,0.007782061739756485,0.03076676552365587,0.06789750764047242,0.11750309741540454,0.17742243760133536,0.24516039801099265,0.31805924880965186,0.3934693402873666,0.4689040089646548,0.5421666382283857,0.6114418724876358,0.6753475326416503,0.7329481647736567,0.7837348331701127,0.8275783761062472,0.8646647167633873,0.8954209987109986,0.9204404912817723,0.9404126812380139,0.9560630663765926,0.9681052066378429,0.9772058191163877,0.9839622907246495,0.9888910034617577};
        double[] param12 = new double[]{0,0.00021699034307820497,0.000867678904324376,0.0019512188925244756,0.003466201029630911,0.005410656605221109,0.007782061739756485,0.01057734284371692,0.013792883256083743,0.017424531042099622,0.02146760792677216,0.025916919337215627,0.03076676552365587,0.03601095372577512,0.041642811348045816,0.0476552001048236,0.05404053109323459,0.06079078074931621,0.06789750764047242,0.07535187004507038,0.08314464426797108,0.09126624363892988,0.09970673813915565,0.10845587459986206,0.11750309741540454};
        
        instance = new Rayleigh(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(param0p5[i], instance.cdf(range[i]+3), 1e-10);
        instance = new Rayleigh(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.cdf(range[i]+3), 1e-10);
        instance = new Rayleigh(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.cdf(range[i]+3), 1e-10);
    }

    /**
     * Test of invCdf method, of class Rayleigh.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        ContinuousDistribution instance = null;

        double[] param0p5 = new double[]
        {0.06415021097594048,0.1587936611248885,0.2173601198712892,0.2651244524946627,0.3073276166341036,0.34617398318289344,0.38285868033545123,0.41813499523464237,0.45253281099872633,0.48646053511406523,0.5202601117191001,0.5542410588845034,0.5887050112577373,0.6239668877424936,0.6603769423628185,0.6983479248355227,0.7383931805037708,0.7811856713602556,0.8276574896199848,0.8791827189816046,0.9379495207658656,1.0078274461666232,1.0968422338547341,1.2272452027824763,1.549842095946111};
        double[] param2 = new double[]
        {0.2566008439037619,0.635174644499554,0.8694404794851568,1.0604978099786508,1.2293104665364143,1.3846959327315738,1.531434721341805,1.6725399809385695,1.8101312439949053,1.945842140456261,2.0810404468764006,2.2169642355380135,2.3548200450309493,2.4958675509699746,2.641507769451274,2.7933916993420906,2.953572722015083,3.1247426854410225,3.310629958479939,3.5167308759264184,3.7517980830634623,4.031309784666493,4.3873689354189365,4.908980811129905,6.199368383784444};
        double[] param12 = new double[]
        {1.5396050634225715,3.811047866997324,5.2166428769109405,6.362986859871905,7.375862799218486,8.308175596389443,9.18860832805083,10.035239885631416,10.860787463969432,11.675052842737566,12.486242681258403,13.301785413228082,14.128920270185695,14.975205305819848,15.849046616707644,16.760350196052542,17.721436332090498,18.748456112646135,19.863779750879633,21.10038525555851,22.510788498380773,24.187858707998956,26.324213612513617,29.453884866779433,37.19621030270666};

        instance = new Rayleigh(0.5);
        for(int i = 0; i < range.length-2; i++)//-2 b/c it enters a numerically unstable range that isnt fair
            assertEquals(param0p5[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Rayleigh(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Rayleigh(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
    }

    @Test
    public void testMin()
    {
        System.out.println("min");
        ContinuousDistribution dist = new Rayleigh(0.5);
        assertTrue(0 == dist.min());
    }

    /**
     * Test of max method, of class Rayleigh.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        ContinuousDistribution dist = new Rayleigh(0.5);
        assertTrue(Double.POSITIVE_INFINITY == dist.max());
    }
    
    /**
     * Test of mean method, of class ChiSquared.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        ContinuousDistribution dist = new Rayleigh(0.5);
        assertEquals(0.6266570686577501, dist.mean(), 1e-10);
        dist = new Rayleigh(2);
        assertEquals(2.5066282746310002, dist.mean(), 1e-10);
        dist = new Rayleigh(12);
        assertEquals(15.039769647786002, dist.mean(), 1e-10);
    }

    /**
     * Test of median method, of class ChiSquared.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        ContinuousDistribution dist = new Rayleigh(0.5);
        assertEquals(0.5887050112577373, dist.median(), 1e-10);
        dist = new Rayleigh(2);
        assertEquals(2.3548200450309493, dist.median(), 1e-10);
        dist = new Rayleigh(12);
        assertEquals(14.128920270185695, dist.median(), 1e-10);
    }

    /**
     * Test of mode method, of class ChiSquared.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        ContinuousDistribution dist = new Rayleigh(0.5);
        assertEquals(0.5, dist.mode(), 1e-10);
        dist = new Rayleigh(2);
        assertEquals(2, dist.mode(), 1e-10);
        dist = new Rayleigh(12);
        assertEquals(12, dist.mode(), 1e-10);
    }

    /**
     * Test of variance method, of class ChiSquared.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        ContinuousDistribution dist = new Rayleigh(0.5);
        assertEquals(0.10730091830127586, dist.variance(), 1e-10);
        dist = new Rayleigh(2);
        assertEquals(1.7168146928204138, dist.variance(), 1e-10);
        dist = new Rayleigh(12);
        assertEquals(61.805328941534896, dist.variance(), 1e-10);
    }

    /**
     * Test of skewness method, of class ChiSquared.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        ContinuousDistribution dist = new Rayleigh(0.5);
        assertEquals(0.6311106578189364, dist.skewness(), 1e-10);
        dist = new Rayleigh(2);
        assertEquals(0.6311106578189364, dist.skewness(), 1e-10);
        dist = new Rayleigh(12);
        assertEquals(0.6311106578189364, dist.skewness(), 1e-10);
    }
    
	@Test
	public void testEquals(){
		System.out.println("equals");
		ContinuousDistribution d1 = new Rayleigh(0.5);
		ContinuousDistribution d2 = new Rayleigh(0.6);
		ContinuousDistribution d4 = new Rayleigh(0.5);
		Integer i = new Integer(1);
		assertFalse(d1.equals(d2));
		assertFalse(d1.equals(i));
		assertFalse(d1.equals(null));
    	assertEquals(d1, d1);
		assertEquals(d1, d4);
    	assertEquals(d1, d1.clone());
	}
	
	@Test
	public void testHashCode(){
		System.out.println("hashCode");
		ContinuousDistribution d1 = new Rayleigh(0.5);
		ContinuousDistribution d2 = new Rayleigh(0.6);
		ContinuousDistribution d4 = new Rayleigh(0.5);
		assertEquals(d1.hashCode(), d4.hashCode());
		assertFalse(d1.hashCode()==d2.hashCode());
	}
}
