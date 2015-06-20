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
public class ExponentialTest
{
    double[] range = new double[]
    {
        -3., -2.75, -2.5, -2.25, -2., -1.75, 
        -1.5, -1.25, -1., -0.75, -0.5, -0.25,
        0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 
        1.75, 2., 2.25, 2.5, 2.75, 3.
    };
    
    public ExponentialTest()
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
     * Test of logPdf method, of class Exponential.
     */
    @Test
    public void testLogPdf()
    {
        System.out.println("logPdf");
        ContinuousDistribution instance = null;

        double[] param0p5 = new double[]{0,0,-0.6931471805599453,-0.8181471805599453,-0.9431471805599453,-1.0681471805599452,-1.1931471805599454,-1.3181471805599452,-1.4431471805599454,-1.5681471805599454,-1.6931471805599452,-1.8181471805599452,-1.9431471805599454,-2.0681471805599454,-2.1931471805599454,-2.3181471805599454,-2.4431471805599454,-2.5681471805599454,-2.6931471805599454,-2.8181471805599454,-2.9431471805599454,-3.0681471805599454,-3.1931471805599454,-3.3181471805599454,-3.4431471805599454};
        double[] param2 = new double[]{0,0,0.6931471805599453,0.19314718055994531,-0.30685281944005466,-0.8068528194400547,-1.3068528194400546,-1.8068528194400546,-2.3068528194400546,-2.8068528194400546,-3.3068528194400546,-3.8068528194400546,-4.306852819440055,-4.806852819440055,-5.306852819440055,-5.806852819440055,-6.306852819440055,-6.806852819440055,-7.306852819440055,-7.806852819440055,-8.306852819440055,-8.806852819440055,-9.306852819440055,-9.806852819440055,-10.306852819440055};
        double[] param12 = new double[]{0,0,2.4849066497880004,-0.5150933502119998,-3.5150933502119996,-6.515093350211999,-9.515093350212,-12.515093350212,-15.515093350212,-18.515093350212,-21.515093350212,-24.515093350212,-27.515093350212,-30.515093350212,-33.515093350212,-36.515093350212,-39.515093350212,-42.515093350212,-45.515093350212,-48.515093350212,-51.515093350212,-54.515093350212,-57.515093350212,-60.515093350212,-63.515093350212};
        
        instance = new Exponential(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(param0p5[i], instance.logPdf(range[i]+2.5), 1e-10);
        instance = new Exponential(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.logPdf(range[i]+2.5), 1e-10);
        instance = new Exponential(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.logPdf(range[i]+2.5), 1e-10);
    }

    /**
     * Test of pdf method, of class Exponential.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        ContinuousDistribution instance = null;
        
        double[] param0p5 = new double[]{0,0,0.5,0.4412484512922977,0.38940039153570244,0.3436446393954861,0.3032653298563167,0.26763071425949514,0.23618327637050734,0.2084310098392542,0.18393972058572117,0.16232623367917487,0.14325239843009505,0.12641979790237323,0.11156508007421491,0.09845583760209703,0.08688697172522257,0.07667748342246423,0.06766764161830635,0.05971648413335981,0.052699612280932166,0.046507244605331746,0.0410424993119494,0.03621987851712573,0.031963930603353785};
        double[] param2 = new double[]{0,0,2.,1.2130613194252668,0.7357588823428847,0.44626032029685964,0.2706705664732254,0.1641699972477976,0.09957413673572789,0.060394766844637,0.03663127777746836,0.022217993076484612,0.013475893998170934,0.008173542876928133,0.004957504353332717,0.0030068783859551447,0.0018237639311090325,0.0011061687402956673,0.0006709252558050237,0.00040693673802128834,0.0002468196081733591,0.0001497036597754012,0.00009079985952496971,0.000055072898699494316,0.00003340340158049132};
        double[] param12 = new double[]{0,0,12.,0.5974448204143673,0.029745026119996302,0.0014809176490401547,0.00007373054823993851,3.6708278460219094e-6,1.8275975693655156e-7,9.099072513494288e-9,4.5301614531349173e-10,2.2554345798469e-11,1.122914756260821e-12,5.590663374124077e-14,2.7834273962922836e-15,1.3857869007618943e-16,6.899426717152272e-18,3.4350222966592726e-19,1.7101968992891223e-20,8.514568994741645e-22,4.2391542866409685e-23,2.110550642909174e-24,1.0507812915235825e-25,5.231532000075697e-27,2.6046264135643672e-28};
        
        instance = new Exponential(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(param0p5[i], instance.pdf(range[i]+2.5), 1e-10);
        instance = new Exponential(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.pdf(range[i]+2.5), 1e-10);
        instance = new Exponential(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.pdf(range[i]+2.5), 1e-10);
    }

    /**
     * Test of cdf method, of class Exponential.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        ContinuousDistribution instance = null;
        
        double[] param0p5 = new double[]{0,0,0.,0.11750309741540454,0.22119921692859512,0.31271072120902776,0.3934693402873666,0.4647385714810097,0.5276334472589853,0.5831379803214916,0.6321205588285577,0.6753475326416503,0.7134952031398099,0.7471604041952535,0.7768698398515702,0.8030883247958059,0.8262260565495548,0.8466450331550716,0.8646647167633873,0.8805670317332803,0.8946007754381357,0.9069855107893365,0.9179150013761012,0.9275602429657486,0.9360721387932924};
        double[] param2 = new double[]{0,0,0.,0.3934693402873666,0.6321205588285577,0.7768698398515702,0.8646647167633873,0.9179150013761012,0.950212931632136,0.9698026165776815,0.9816843611112658,0.9888910034617577,0.9932620530009145,0.995913228561536,0.9975212478233336,0.9984965608070224,0.9990881180344455,0.9994469156298522,0.9996645373720975,0.9997965316309894,0.9998765901959134,0.9999251481701124,0.9999546000702375,0.9999724635506503,0.9999832982992097};
        double[] param12 = new double[]{0,0,0.,0.950212931632136,0.9975212478233336,0.9998765901959134,0.9999938557876467,0.9999996940976795,0.9999999847700203,0.999999999241744,0.9999999999622486,0.9999999999981205,0.9999999999999064,0.9999999999999953,0.9999999999999998,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.};
        
        instance = new Exponential(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(param0p5[i], instance.cdf(range[i]+2.5), 1e-10);
        instance = new Exponential(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.cdf(range[i]+2.5), 1e-10);
        instance = new Exponential(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.cdf(range[i]+2.5), 1e-10);
    }

    /**
     * Test of invCdf method, of class Exponential.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        ContinuousDistribution instance;
        
        double[] param0p5 = new double[]
        {0.016460998273030734,0.10086170725378371,0.18898168684184483,0.28116390124237867,0.37780105578399414,0.4793457065308407,0.5863230764328129,0.6993474969594976,0.8191437801216357,0.9465754088938508,1.0826823353838824,1.2287326054136622,1.3862943611198906,1.5573387079962149,1.7443908240178614,1.9507592964883234,2.180897956057897,2.4410042125542932,2.740067680496221,3.091849013423549,3.5189972140196675,4.062864644986952,4.812251543869772,6.024523151010405,9.608042089466538};
        double[] param2 = new double[]
        {0.004115249568257684,0.025215426813445928,0.047245421710461206,0.07029097531059467,0.09445026394599854,0.11983642663271017,0.14658076910820322,0.1748368742398744,0.20478594503040892,0.2366438522234627,0.2706705838459706,0.30718315135341556,0.34657359027997264,0.3893346769990537,0.43609770600446535,0.48768982412208084,0.5452244890144743,0.6102510531385733,0.6850169201240552,0.7729622533558872,0.8797493035049169,1.015716161246738,1.203062885967443,1.5061307877526013,2.4020105223666346};
        double[] param12 = new double[]
        {0.0006858749280429472,0.004202571135574321,0.007874236951743534,0.011715162551765777,0.01574171065766642,0.01997273777211836,0.024430128184700535,0.029139479039979065,0.03413099083840149,0.03944064203724378,0.04511176397432843,0.05119719189223593,0.057762265046662105,0.06488911283317561,0.07268295100074422,0.08128163735368013,0.09087074816907904,0.10170850885642888,0.11416948668734253,0.12882704222598118,0.14662488391748613,0.16928602687445632,0.20051048099457383,0.2510217979587669,0.40033508706110577};

        instance = new Exponential(0.5);
        for(int i = 0; i < range.length-2; i++)//-2 b/c it enters a numerically unstable range that isnt fair
            assertEquals(param0p5[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Exponential(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Exponential(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
    }

    /**
     * Test of min method, of class Exponential.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        ContinuousDistribution dist = new Exponential(0.5);
        assertTrue(0 == dist.min());
    }

    /**
     * Test of max method, of class Exponential.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        ContinuousDistribution dist = new Exponential(0.5);
        assertTrue(Double.POSITIVE_INFINITY == dist.max());
    }

    /**
     * Test of mean method, of class Exponential.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        ContinuousDistribution dist = new Exponential(0.5);
        assertEquals(2, dist.mean(), 1e-10);
        dist = new Exponential(2);
        assertEquals(0.5, dist.mean(), 1e-10);
        dist = new Exponential(12);
        assertEquals(0.08333333333333333, dist.mean(), 1e-10);
    }

    /**
     * Test of median method, of class Exponential.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        ContinuousDistribution dist = new Exponential(0.5);
        assertEquals(1.3862943611198906, dist.median(), 1e-10);
        dist = new Exponential(2);
        assertEquals(0.34657359027997264, dist.median(), 1e-10);
        dist = new Exponential(12);
        assertEquals(0.057762265046662105, dist.median(), 1e-10);
    }

    /**
     * Test of mode method, of class Exponential.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        ContinuousDistribution dist = new Exponential(0.5);
        assertEquals(0, dist.mode(), 1e-10);
        dist = new Exponential(2);
        assertEquals(0, dist.mode(), 1e-10);
        dist = new Exponential(12);
        assertEquals(0, dist.mode(), 1e-10);
    }

    /**
     * Test of variance method, of class Exponential.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        ContinuousDistribution dist = new Exponential(0.5);
        assertEquals(4, dist.variance(), 1e-10);
        dist = new Exponential(2);
        assertEquals(0.25, dist.variance(), 1e-10);
        dist = new Exponential(12);
        assertEquals(0.006944444444444444, dist.variance(), 1e-10);
    }

    /**
     * Test of skewness method, of class Exponential.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        ContinuousDistribution dist = new Exponential(0.5);
        assertEquals(2, dist.skewness(), 1e-10);
        dist = new Exponential(2);
        assertEquals(2, dist.skewness(), 1e-10);
        dist = new Exponential(12);
        assertEquals(2, dist.skewness(), 1e-10);
    }
	@Test
	public void testEquals(){
		System.out.println("equals");
		ContinuousDistribution d1 = new Exponential(0.5);
		ContinuousDistribution d2 = new Exponential(0.6);
		ContinuousDistribution d4 = new Exponential(0.5);
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
		ContinuousDistribution d1 = new Exponential(0.5);
		ContinuousDistribution d2 = new Exponential(0.6);
		ContinuousDistribution d4 = new Exponential(0.5);
		assertEquals(d1.hashCode(), d4.hashCode());
		assertFalse(d1.hashCode()==d2.hashCode());
	}
}
