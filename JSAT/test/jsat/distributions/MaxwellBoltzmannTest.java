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
public class MaxwellBoltzmannTest
{
    double[] range = new double[]
    {
        -3., -2.75, -2.5, -2.25, -2., -1.75, 
        -1.5, -1.25, -1., -0.75, -0.5, -0.25,
        0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 
        1.75, 2., 2.25, 2.5, 2.75, 3.
    };
    
    public MaxwellBoltzmannTest()
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
     * Test of setShape method, of class MaxwellBoltzmann.
     */
    @Test
    public void testSetShape()
    {
        System.out.println("setSigma");
        MaxwellBoltzmann instance = new MaxwellBoltzmann();
        instance.setShape(1);
        try
        {
            instance.setShape(0);
            fail("invalid value");
        }
        catch(Exception ex)
        {
            
        }
        
        try
        {
            instance.setShape(Double.POSITIVE_INFINITY);
            fail("invalid value");
        }
        catch(Exception ex)
        {
            
        }
    }

    
    @Test
    public void testLogPdf()
    {
        System.out.println("pdf");
        ContinuousDistribution instance = null;

        double[] param0p5 = new double[]{0,0,0,-1.0439385332046727,-0.032644172084782055,0.15328604413154664,-0.1463498109648913,-0.8250627083364721,-1.8354195947485628,-3.152118235094046,-4.760055449845001,-6.649489378532234,-8.813768347216582,-11.24814798760793,-13.949125233628672,-16.914039818281598,-20.140823873974156,-23.627838131000253,-27.37376108872511,-31.37751184509224,-35.63819501741234,-40.15506057487179,-44.92747398609669,-49.95489365775783,-55.23685362648804};
        double[] param2 = new double[]{0,0,0,-5.085634116564345,-3.7227772554444543,-2.950909539228125,-2.4302328943245635,-2.0542582916961436,-1.7755526781082347,-1.568813818453718,-1.4189385332046727,-1.316184961891906,-1.2539014305762533,-1.2273435709676035,-1.233008316988344,-1.2682354016412711,-1.3309569573338274,-1.4195337143599245,-1.5326441720847819,-1.6692074284519127,-1.8283281007720154,-2.0092561582314636,-2.211357069456363,-2.4340892411174986,-2.676986709847713};
        double[] param12 = new double[]{0,0,0,-10.4533170381374,-9.067673718684174,-8.25782857191229,-7.683983524230951,-7.239649546602531,-6.8773935857924,-6.571913406693438,-6.308105829777727,-6.076228994576071,-5.869631227149307,-5.683568159207324,-5.514536724672509,-5.359876656547658,-5.2175200872402145,-5.08582774704409,-4.963478135324503,-4.849390350024966,-4.74266900845618,-4.642564079804518,-4.548441032696084,-4.459758273801663,-4.3760498397541};
        
        instance = new MaxwellBoltzmann(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(param0p5[i], instance.logPdf(range[i]+2.5), 1e-10);
        instance = new MaxwellBoltzmann(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.logPdf(range[i]+2.5), 1e-10);
        instance = new MaxwellBoltzmann(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.logPdf(range[i]+2.5), 1e-10);
    }
    
    /**
     * Test of pdf method, of class MaxwellBoltzmann.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        ContinuousDistribution instance = null;
        
        double[] param0p5 = new double[]{0,0,0,0.35206532676429947,0.9678828980765735,1.1656583609930256,0.863855464211009,0.4382075123392134,0.15954654282976827,0.042761452057242244,0.008565134448952664,0.0012946830296593434,0.00014867195147342976,0.000013031409651477368,8.749271303745532e-7,4.511550678949219e-8,1.7904052000394604e-9,5.477221199315272e-11,1.2933813973854447e-12,2.3600420975525e-14,3.3306466372207283e-16,3.638134677342304e-18,3.0778394506825684e-20,2.0177406354196752e-22,1.0255085186985274e-24};
        double[] param2 = new double[]{0,0,0,0.00618496385851171,0.024166757300178074,0.05229212257543626,0.08801633169107487,0.12818787833999026,0.16938980558707747,0.20829210813357238,0.24197072451914337,0.2681563798098696,0.2853891959203467,0.2930700643820856,0.2914145902482564,0.28132761428232617,0.26422428890619154,0.2418267509532137,0.21596386605275225,0.18839632404815398,0.16068198741806627,0.13408837811873256,0.10955187808480335,0.08767756381566659,0.06877006641828234};
        double[] param12 = new double[]{0,0,0,0.000028852409850921587,0.00011533452737288349,0.00025922126086385847,0.00046013827113234023,0.0007175631885070832,0.0010308273097519517,0.0013991177680274776,0.0018214801671386106,0.0022968216694486633,0.002823914525031556,0.003401400027886845,0.004027792883363012,0.004701485969332062,0.005420755472140021,0.00618376637693079,0.006988578290610984,0.00783315157449754,0.008715353762572708,0.009632966240267904,0.01058369115781257,0.011565158551420455,0.012574933644946164};
        
        instance = new MaxwellBoltzmann(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(param0p5[i], instance.pdf(range[i]+2.5), 1e-10);
        instance = new MaxwellBoltzmann(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.pdf(range[i]+2.5), 1e-10);
        instance = new MaxwellBoltzmann(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.pdf(range[i]+2.5), 1e-10);
    }

    /**
     * Test of cdf method, of class MaxwellBoltzmann.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        ContinuousDistribution instance = null;
        
        double[] param0p5 = new double[]{0,0,0,0.030859595783726712,0.19874804309879912,0.4778328104646086,0.7385358700508893,0.8999391668806049,0.9707091134651118,0.9934259629766086,0.9988660157102147,0.9998493509837884,0.999984559501709,0.9999987773472704,0.9999999251162305,0.9999999964492564,0.9999999998695543,0.9999999999962846,0.999999999999918,0.9999999999999986,1.,1.,1.,1.,1.};
        double[] param2 = new double[]{0,0,0,0.0005170279240384046,0.004078592964422811,0.013448212943120874,0.030859595783726712,0.057827731214630085,0.09503914701405697,0.1423298471325275,0.19874804309879912,0.26268851273105853,0.33207773919373457,0.40458482528702217,0.4778328104646086,0.5495880697451594,0.6179110703795754,0.6812587421193825,0.7385358700508893,0.7890991997636632,0.8327226214294071,0.8695345208664327,0.8999391668806049,0.9245331498906099,0.94402587977114};
        double[] param12 = new double[]{0,0,0,2.4045762129741577e-6,0.00001922909733632211,0.00006485597263361509,0.00015359266218761825,0.0002996345070972878,0.0005170279240384046,0.0008196340833463434,0.0012210931878273434,0.0017347894670798891,0.002373816999227285,0.0031509464686337862,0.004078592964422811,0.005168784920450115,0.0064331342928277135,0.00788280806617181,0.009528501174477233,0.01138041091693437,0.013448212943120874,0.015741038875855595,0.0182674556336187,0.02103544650785194,0.024052394043691305};
        
        instance = new MaxwellBoltzmann(0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(param0p5[i], instance.cdf(range[i]+2.5), 1e-10);
        instance = new MaxwellBoltzmann(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.cdf(range[i]+2.5), 1e-10);
        instance = new MaxwellBoltzmann(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.cdf(range[i]+2.5), 1e-10);
    }

    /**
     * Test of invCdf method, of class MaxwellBoltzmann.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        ContinuousDistribution instance;
        
        double[] param0p5 = new double[]
        {0.15833852261996845,0.29483399443273883,0.367702453369813,0.4238115493761005,0.47166773202109047,0.514633536906765,0.5544484063052013,0.5921637017354954,0.628489130214972,0.663948955783709,0.6989632279600045,0.7338956874781447,0.7690861272275261,0.8048761358046771,0.8416338247492682,0.879782418021023,0.9198388029472511,0.9624720066447157,1.0086008009162615,1.0595722524099012,1.1175243646484285,1.1862308058829834,1.2735108480574397,1.4010354011075703,1.7157036426968402};
        double[] param2 = new double[]
        {0.6333540904798738,1.1793359777309553,1.470809813479252,1.695246197504402,1.8866709280843619,2.05853414762706,2.2177936252208053,2.3686548069419815,2.513956520859888,2.655795823134836,2.795852911840018,2.935582749912579,3.0763445089101045,3.2195045432187084,3.3665352989970727,3.519129672084092,3.6793552117890043,3.8498880265788626,4.034403203665046,4.238289009639605,4.470097458593714,4.744923223531933,5.094043392229759,5.604141604430281,6.862814570787361};
        double[] param12 = new double[]
        {3.8001245428792427,7.076015866385732,8.824858880875512,10.171477185026411,11.320025568506171,12.35120488576236,13.306761751324832,14.21192884165189,15.083739125159328,15.934774938809017,16.775117471040108,17.613496499475474,18.458067053460628,19.31702725931225,20.199211793982435,21.114778032504553,22.076131270734024,23.099328159473174,24.206419221990274,25.429734057837628,26.820584751562286,28.4695393411916,30.564260353378554,33.624849626581685,41.176887424724164};

        instance = new MaxwellBoltzmann(0.5);
        for(int i = 0; i < range.length-2; i++)//-2 b/c it enters a numerically unstable range that isnt fair
            assertEquals(param0p5[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new MaxwellBoltzmann(2);
        for(int i = 0; i < range.length; i++)
            assertEquals(param2[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new MaxwellBoltzmann(12);
        for(int i = 0; i < range.length; i++)
            assertEquals(param12[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
    }

    /**
     * Test of min method, of class MaxwellBoltzmann.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        ContinuousDistribution dist = new MaxwellBoltzmann(0.5);
        assertTrue(0 == dist.min());
    }

    /**
     * Test of max method, of class MaxwellBoltzmann.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        ContinuousDistribution dist = new MaxwellBoltzmann(0.5);
        assertTrue(Double.POSITIVE_INFINITY == dist.max());
    }

    /**
     * Test of mean method, of class MaxwellBoltzmann.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        ContinuousDistribution dist = new MaxwellBoltzmann(0.5);
        assertEquals(0.7978845608028654, dist.mean(), 1e-10);
        dist = new MaxwellBoltzmann(2);
        assertEquals(3.1915382432114616, dist.mean(), 1e-10);
        dist = new MaxwellBoltzmann(12);
        assertEquals(19.14922945926877, dist.mean(), 1e-10);
    }
    
    @Test
    public void testMedian()
    {
        System.out.println("median");
        ContinuousDistribution dist = new MaxwellBoltzmann(0.5);
        assertEquals(0.7690861272275261, dist.median(), 1e-10);
        dist = new MaxwellBoltzmann(2);
        assertEquals(3.0763445089101045, dist.median(), 1e-10);
        dist = new MaxwellBoltzmann(12);
        assertEquals(18.458067053460628, dist.median(), 1e-10);
    }

    /**
     * Test of mode method, of class MaxwellBoltzmann.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        ContinuousDistribution dist = new MaxwellBoltzmann(0.5);
        assertEquals(0.7071067811882024, dist.mode(), 1e-10);
        dist = new MaxwellBoltzmann(2);
        assertEquals(2.828427124743843, dist.mode(), 1e-10);
        dist = new MaxwellBoltzmann(12);
        assertEquals(16.970562749522642, dist.mode(), 1e-8);//Hitting the bounds of what we can numericaly caluate well. But relative error is still in 1e-10
    }

    /**
     * Test of variance method, of class MaxwellBoltzmann.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        ContinuousDistribution dist = new MaxwellBoltzmann(0.5);
        assertEquals(0.11338022763241863, dist.variance(), 1e-10);
        dist = new MaxwellBoltzmann(2);
        assertEquals(1.814083642118698, dist.variance(), 1e-10);
        dist = new MaxwellBoltzmann(12);
        assertEquals(65.30701111627313, dist.variance(), 1e-10);
    }

    /**
     * Test of skewness method, of class MaxwellBoltzmann.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        ContinuousDistribution dist = new MaxwellBoltzmann(0.5);
        assertEquals(0.48569282804959213, dist.skewness(), 1e-10);
        dist = new MaxwellBoltzmann(2);
        assertEquals(0.48569282804959213, dist.skewness(), 1e-10);
        dist = new MaxwellBoltzmann(12);
        assertEquals(0.48569282804959213, dist.skewness(), 1e-10);
    }
	@Test
	public void testEquals(){
		System.out.println("equals");
		ContinuousDistribution d1 = new MaxwellBoltzmann(0.5);
		ContinuousDistribution d2 = new MaxwellBoltzmann(0.6);
		ContinuousDistribution d4 = new MaxwellBoltzmann(0.5);
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
		ContinuousDistribution d1 = new MaxwellBoltzmann(0.5);
		ContinuousDistribution d2 = new MaxwellBoltzmann(0.6);
		ContinuousDistribution d4 = new MaxwellBoltzmann(0.5);
		assertEquals(d1.hashCode(), d4.hashCode());
		assertFalse(d1.hashCode()==d2.hashCode());
	}
}
