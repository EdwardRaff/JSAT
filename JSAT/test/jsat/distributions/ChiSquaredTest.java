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
public class ChiSquaredTest
{
    double[] range = new double[]
    {
        -3., -2.75, -2.5, -2.25, -2., -1.75, 
        -1.5, -1.25, -1., -0.75, -0.5, -0.25,
        0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 
        1.75, 2., 2.25, 2.5, 2.75, 3.
    };
    
    public ChiSquaredTest()
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
     * Test of pdf method, of class ChiSquared.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        ChiSquared instance = null;
        
        double[] df0p5 = new double[]{0,0.57892140748046,0.303780786595025,0.19779032668795404,0.14067411288159115,0.10501343513705805,0.08082991466920406,0.06354409778550153,0.05073345595415418,0.04098672759009646,0.03342245268591643,0.027460408471784374,0.02270276544056991,0.01886776125462765,0.015750525546907305,0.013198841745077544,0.011097559156248158,0.00935823516479523,0.00791205789422622,0.006704892579220859,0.005693741133655941,0.004844165066959195,0.0041283792228703564,0.003523821549028722,0.0030120664071556255};
        double[] df2 = new double[]{0,0.44124845129229767,0.38940039153570244,0.3436446393954861,0.3032653298563167,0.26763071425949514,0.23618327637050734,0.2084310098392542,0.18393972058572117,0.16232623367917487,0.14325239843009505,0.12641979790237323,0.11156508007421492,0.09845583760209702,0.08688697172522256,0.07667748342246423,0.06766764161830635,0.05971648413335981,0.052699612280932166,0.046507244605331746,0.0410424993119494,0.036219878517125735,0.031963930603353785,0.02820806975188867,0.024893534183931972};
        double[] df12 = new double[]{0,1.1221528404039961e-7,3.1689484988257036e-6,0.00002123658431322813,0.00007897534631674914,0.0002126937820589504,0.0004670616549319115,0.0008908843949301007,0.0015328310048810098,0.002437642866140136,0.0036430968839033773,0.005177824623610207,0.007059977723446413,0.00929666221893194,0.011884027781460088,0.014807882683800707,0.018044704431548358,0.021562924197423262,0.025324376672987984,0.029285823023482597,0.03340047144527132,0.03761943615510857,0.041893090719406986,0.04617228502682505,0.05040940672246224};
        
        instance = new ChiSquared(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(df0p5[i], instance.pdf(range[i]+3), 1e-10);
        instance = new ChiSquared(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(df2[i], instance.pdf(range[i]+3), 1e-10);
        instance = new ChiSquared(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(df12[i], instance.pdf(range[i]+3), 1e-10);
    }

    /**
     * Test of cdf method, of class ChiSquared.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        ChiSquared instance = null;
        
        double[] df0p5 = new double[]{0,0.640157206083084,0.7436779447314611,0.8047991126008802,0.8464864041916775,0.87688573996339,0.8999365132844983,0.9178700519520441,0.932078867989891,0.9434907075698542,0.9527532988560908,0.9603349624798424,0.9665835558410207,0.9717630228513373,0.9760771069754204,0.9796853136500356,0.9827139881404834,0.9852642025957218,0.9874174943013079,0.9892401185815257,0.9907862515204601,0.992100435436148,0.9932194688421462,0.9941738826536312,0.9949891040512912};
        double[] df2 = new double[]{0,0.11750309741540454,0.22119921692859512,0.31271072120902776,0.3934693402873666,0.4647385714810097,0.5276334472589853,0.5831379803214916,0.6321205588285577,0.6753475326416503,0.7134952031398099,0.7471604041952535,0.7768698398515702,0.8030883247958059,0.8262260565495548,0.8466450331550716,0.8646647167633873,0.8805670317332803,0.8946007754381357,0.9069855107893365,0.9179150013761012,0.9275602429657486,0.9360721387932924,0.9435838604962227,0.950212931632136};
        double[] df12 = new double[]{0,4.7604532844419965e-9,2.7381356338284025e-7,2.803736903342154e-6,0.00001416493732234249,0.00004859953912795168,0.00013055446292196965,0.0002962521561470425,0.0005941848175816929,0.0010845927305314122,0.0018380854505885185,0.0029336100162652462,0.0044559807752478486,0.006493173769154503,0.009133564163468133,0.01246325444683668,0.016563608480614434,0.021509074844324818,0.02736535406330157,0.03418793918529287,0.04202103819530612,0.05089686994361906,0.06083531238569402,0.0718438726220419,0.08391794203130346};
        
        instance = new ChiSquared(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(df0p5[i], instance.cdf(range[i]+3), 1e-10);
        instance = new ChiSquared(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(df2[i], instance.cdf(range[i]+3), 1e-10);
        instance = new ChiSquared(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(df12[i], instance.cdf(range[i]+3), 1e-10);
    }
        

    /**
     * Test of invCdf method, of class ChiSquared.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        ChiSquared instance = null;

        double[] df0p5 = new double[]
        {
            6.093614961311839e-9, 7.897349917808368e-6, 0.0000892198004181562, 0.0003994149546910942, 0.0011856544464383886, 
            0.002787741883826599, 0.005640285791335964, 0.010277055179909702, 0.017338752494932063, 0.027585874814306646, 
            0.041918981404673365, 0.061409712302900245, 0.08734760470574683, 0.1213106759556879, 0.16527293485434283, 
            0.22177158982679546, 0.29417538482483335, 0.38713399406906546, 0.5073739757420875, 0.6652156884578238, 
            0.8777617025944231, 1.1765830034154225, 1.6305260036139653, 2.4434484059336072, 5.189836396971451
        };
        double[] df2 = new double[]
        {
            0.016460998273030734, 0.10086170725378371, 0.18898168684184483, 0.28116390124237867, 0.37780105578399414, 
            0.4793457065308407, 0.5863230764328129, 0.6993474969594976, 0.8191437801216357, 0.9465754088938508, 
            1.0826823353838824, 1.2287326054136622, 1.3862943611198906, 1.5573387079962149, 1.7443908240178614, 
            1.9507592964883234, 2.180897956057897, 2.4410042125542932, 2.740067680496221, 3.091849013423549, 
            3.5189972140196675, 4.062864644986952, 4.812251543869772, 6.024523151010405, 9.608042089466538
        };
        double[] df12 = new double[]
        {
            3.4180950469151483, 5.203890364257783, 6.1214555157081225, 6.8266639318647435, 7.430985599781199, 
            7.97751232692583, 8.488313547594942, 8.976714493919122, 9.451780012809346, 9.920320955923309, 
            10.387938467434852, 10.859649089467469, 11.34032237742414, 11.835049834156907, 12.349524382578897, 
            12.890506293567935, 13.466478617419032, 14.08866824134716, 14.772779308969527, 15.5422029995657, 
            16.434605452689, 17.51743977608438, 18.932767402341046, 21.08284981490997, 26.821781360074844
        };

        instance = new ChiSquared(0.5);
        for(int i = 0; i < range.length-2; i++)//-2 b/c it enters a numerically unstable range that isnt fair
            assertEquals(df0p5[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new ChiSquared(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(df2[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new ChiSquared(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(df12[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
    }

    /**
     * Test of min method, of class ChiSquared.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        ChiSquared dist = new ChiSquared(0.5);
        assertTrue(0 == dist.min());
    }

    /**
     * Test of max method, of class ChiSquared.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        ChiSquared dist = new ChiSquared(0.5);
        assertTrue(Double.POSITIVE_INFINITY == dist.max());
    }
    
    /**
     * Test of mean method, of class ChiSquared.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        ChiSquared dist = new ChiSquared(12);
        assertEquals(12, dist.mean(), 1e-10);
        dist = new ChiSquared(2);
        assertEquals(2, dist.mean(), 1e-10);
        dist = new ChiSquared(0.5);
        assertEquals(0.5, dist.mean(), 1e-10);
    }

    /**
     * Test of median method, of class ChiSquared.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        ChiSquared dist = new ChiSquared(12);
        assertEquals(11.34032237742414, dist.median(), 1e-10);
        dist = new ChiSquared(2);
        assertEquals(1.3862943611198906, dist.median(), 1e-10);
        dist = new ChiSquared(0.5);
        assertEquals(0.08734760470574683, dist.median(), 1e-10);
    }

    /**
     * Test of mode method, of class ChiSquared.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        ChiSquared dist = new ChiSquared(12);
        assertEquals(10, dist.mode(), 1e-10);
        dist = new ChiSquared(2);
        assertEquals(0, dist.mode(), 1e-10);
        dist = new ChiSquared(0.5);
        assertEquals(0, dist.mode(), 1e-10);
    }

    /**
     * Test of variance method, of class ChiSquared.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        ChiSquared dist = new ChiSquared(12);
        assertEquals(24, dist.variance(), 1e-10);
        dist = new ChiSquared(2);
        assertEquals(4, dist.variance(), 1e-10);
        dist = new ChiSquared(0.5);
        assertEquals(1, dist.variance(), 1e-10);
    }

    /**
     * Test of skewness method, of class ChiSquared.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        ChiSquared dist = new ChiSquared(12);
        assertEquals(0.816496580927726, dist.skewness(), 1e-10);
        dist = new ChiSquared(2);
        assertEquals(2, dist.skewness(), 1e-10);
        dist = new ChiSquared(0.5);
        assertEquals(4, dist.skewness(), 1e-10);
    }


	@Test
	public void testEquals(){
		System.out.println("equals");
		ContinuousDistribution d1 = new ChiSquared(0.5);
		ContinuousDistribution d2 = new ChiSquared(0.6);
		ContinuousDistribution d4 = new ChiSquared(0.5);
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
		ContinuousDistribution d1 = new ChiSquared(0.5);
		ContinuousDistribution d2 = new ChiSquared(0.6);
		ContinuousDistribution d4 = new ChiSquared(0.5);
		assertEquals(d1.hashCode(), d4.hashCode());
		assertFalse(d1.hashCode()==d2.hashCode());
	}
}
