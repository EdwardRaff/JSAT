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
public class BetaTest
{
    double[] range = new double[]
    {
        -3., -2.75, -2.5, -2.25, -2., -1.75, 
        -1.5, -1.25, -1., -0.75, -0.5, -0.25,
        0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 
        1.75, 2., 2.25, 2.5, 2.75, 3.
    };
    
    public BetaTest()
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
     * Test of pdf method, of class Beta.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0,1.7589346862069402,1.1992165982748448,0.9843543098421845,0.8660395722419083,0.7907925823252256,0.7393296651378021,0.7028344377314711,0.6766845883748794,0.6582485884902135,0.6459658355538042,0.63891821212204,0.6366197723675814,0.63891821212204,0.6459658355538042,0.6582485884902135,0.6766845883748794,0.7028344377314711,0.7393296651378021,0.7907925823252256,0.8660395722419083,0.9843543098421845,1.1992165982748448,1.7589346862069402,0};
        double[] paramTwo1 = new double[]{0,4.752563983397751,2.8965431767081506,2.1142236480652308,1.6445302034980571,1.3191626662009392,1.0757944013125078,0.8850491290558684,0.7309205031500751,0.6037892363480897,0.49744882216410985,0.407672421967636,0.33145630368119416,0.2665911929745277,0.21140530443767036,0.1646030046584622,0.12515953295020701,0.09225001734633169,0.06520024954176369,0.04345170496780734,0.026536147348631647,0.014056838886003111,0.005674401156432602,0.0010960132952870466,0};
        double[] paramTwo2 = new double[]{0,0.0010960132952870466,0.005674401156432602,0.014056838886003111,0.026536147348631647,0.04345170496780734,0.06520024954176369,0.09225001734633169,0.12515953295020701,0.1646030046584622,0.21140530443767036,0.2665911929745277,0.33145630368119416,0.407672421967636,0.49744882216410985,0.6037892363480897,0.7309205031500751,0.8850491290558684,1.0757944013125078,1.3191626662009392,1.6445302034980571,2.1142236480652308,2.8965431767081506,4.752563983397751,0};
        double[] paramTwo3 = new double[]{0,0.03217532266307823,0.14891252930402912,0.32803182145023146,0.5474832246889396,0.7875378145455926,1.0307877164838113,1.2621461059054015,1.4688472081503543,1.6404462984968413,1.7688197021612204,1.8481647942980308,1.8749999999999987,1.8481647942980308,1.7688197021612204,1.6404462984968413,1.4688472081503543,1.2621461059054015,1.0307877164838113,0.7875378145455926,0.5474832246889396,0.32803182145023146,0.14891252930402912,0.03217532266307823,0};
        
        instance = new Beta(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.pdf(range[i]/5.9+0.5), 1e-10);
        instance = new Beta(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.pdf(range[i]/5.9+0.5), 1e-10);
        instance = new Beta(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.pdf(range[i]/5.9+0.5), 1e-10);
        instance = new Beta(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.pdf(range[i]/5.9+0.5), 1e-10);
    }

    /**
     * Test of cdf method, of class Beta.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0,0.11788372107011924,0.17813214287521345,0.22386746294904356,0.2628616068083092,0.29785628990636376,0.3302095813792621,0.3607209558369871,0.3899170658416282,0.4181755799950677,0.44578746335898967,0.4729921922455666,0.5000000000000001,0.5270078077544333,0.5542125366410103,0.5818244200049323,0.6100829341583718,0.639279044163013,0.669790418620738,0.7021437100936363,0.7371383931916908,0.7761325370509564,0.8218678571247866,0.8821162789298808,1.00000000000000};
        double[] paramTwo1 = new double[]{0,0.33749333898201467,0.49209569417277615,0.5965739352074145,0.6755170848657505,0.7379399861678736,0.788455195917326,0.8298465182750109,0.863973724796828,0.8921687214544558,0.9154350625005302,0.9345587974142613,0.9501747372194232,0.962808391664012,0.9729037492836445,0.9808423820196484,0.9869570017772313,0.9915413394571007,0.9948575107152673,0.9971416182210584,0.998608087714689,0.9994530763322207,0.999857188867766,0.9999876694226759,1.00000000000000};
        double[] paramTwo2 = new double[]{0,0.000012330577324056844,0.00014281113223401983,0.0005469236677792935,0.0013919122853110215,0.0028583817789416396,0.005142489284732719,0.008458660542899338,0.01304299822276872,0.019157617980351684,0.02709625071635551,0.037191608335987954,0.04982526278057676,0.0654412025857387,0.08456493749946976,0.10783127854554417,0.13602627520317195,0.1701534817249891,0.21154480408267395,0.2620600138321264,0.32448291513424954,0.40342606479258547,0.5079043058272239,0.6625066610179853,1.00000000000000};
        double[] paramTwo3 = new double[]{0,0.00036998602561136404,0.003944791957616764,0.013869702028410147,0.03231266867319054,0.06055649816429026,0.09909720018482686,0.14774233740235485,0.2057093750425177,0.27172403046269944,0.34411862272567684,0.42093042217327137,0.5000000000000001,0.5790695778267286,0.6558813772743232,0.7282759695373006,0.7942906249574823,0.8522576625976451,0.9009027998151732,0.9394435018357097,0.9676873313268095,0.9861302979715898,0.9960552080423832,0.9996300139743887,1.00000000000000};
        
        instance = new Beta(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.cdf(range[i]/5.9+0.5), 1e-10);
        instance = new Beta(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.cdf(range[i]/5.9+0.5), 1e-10);
        instance = new Beta(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.cdf(range[i]/5.9+0.5), 1e-10);
        instance = new Beta(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.cdf(range[i]/5.9+0.5), 1e-10);
    }

    /**
     * Test of invCdf method, of class Beta.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0.00016576624284345113,0.005956051954461413,0.019925063164199123,0.04184154775649763,0.07134268596183385,0.10794009671164481,0.1510279226168282,0.19989285972581594,0.2537259660230678,0.31163605318240706,0.3726644398793855,0.4358008224267242,0.5000000000000002,0.5641991775732754,0.6273355601206145,0.6883639468175929,0.746274033976932,0.8001071402741841,0.8489720773831718,0.8920599032883552,0.9286573140381662,0.9581584522435024,0.9800749368358008,0.9940439480455386,0.9998342337571565};
        double[] paramTwo1 = new double[]{0.000019111239764582307,0.0006886190862176469,0.0023195643945001826,0.0049245874336406245,0.008524203064237706,0.013147431007302195,0.01883272167757733,0.025629245127778394,0.03359864352684645,0.04281739498704826,0.053380007394563556,0.06540337027538033,0.079032767076173,0.0944503376606125,0.11188727260931457,0.13164189883735045,0.15410746275722342,0.17981669319696458,0.20951722718400426,0.24430834148810293,0.2859123406197187,0.33728555405656346,0.40428132830521096,0.5019552958188651,0.7147352141973307};
        double[] paramTwo2 = new double[]{0.28526478580267,0.4980447041811349,0.5957186716947891,0.6627144459434365,0.7140876593802813,0.7556916585118971,0.7904827728159958,0.8201833068030354,0.8458925372427766,0.8683581011626496,0.8881127273906855,0.9055496623393875,0.920967232923827,0.9345966297246198,0.9466199926054364,0.9571826050129517,0.9664013564731535,0.9743707548722216,0.9811672783224227,0.9868525689926978,0.9914757969357623,0.9950754125663593,0.9976804356054998,0.9993113809137824,0.9999808887602354};
        double[] paramTwo3 = new double[]{0.09848468152143303,0.18808934769778385,0.23687906456075017,0.2746049499190648,0.30675508011185254,0.3355120289556864,0.36200518342617627,0.3869129426302727,0.41068664797302484,0.4336514783763557,0.45605842357305015,0.4781141375560683,0.5,0.5218858624439316,0.5439415764269498,0.5663485216236444,0.5893133520269751,0.6130870573697274,0.6379948165738237,0.6644879710443137,0.6932449198881474,0.7253950500809352,0.7631209354392499,0.8119106523022162,0.9015153184785673};
        
        instance = new Beta(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Beta(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Beta(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new Beta(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
    }

    /**
     * Test of min method, of class Beta.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        ContinuousDistribution dist = new Beta(0.5, 3);
        assertTrue(0 == dist.min());
    }

    /**
     * Test of max method, of class Beta.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        ContinuousDistribution dist = new Beta(0.5, 3);
        assertTrue(1 == dist.max());
    }

    /**
     * Test of mean method, of class Beta.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        ContinuousDistribution dist = new Beta(0.5, 0.5);
        assertEquals(0.5, dist.mean(), 1e-10);
        dist = new Beta(0.5, 3);
        assertEquals(0.14285714285714285, dist.mean(), 1e-10);
        dist = new Beta(3, 0.5);
        assertEquals(0.8571428571428571, dist.mean(), 1e-10);
        dist = new Beta(3, 3);
        assertEquals(0.5, dist.mean(), 1e-10);
    }

    /**
     * Test of median method, of class Beta.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        ContinuousDistribution dist = new Beta(0.5, 0.5);
        assertEquals(0.5, dist.median(), 1e-10);
        dist = new Beta(0.5, 3);
        assertEquals(0.079032767076173, dist.median(), 1e-10);
        dist = new Beta(3, 0.5);
        assertEquals(0.920967232923827, dist.median(), 1e-10);
        dist = new Beta(3, 3);
        assertEquals(0.5, dist.median(), 1e-10);
    }

    /**
     * Test of mode method, of class Beta.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        ContinuousDistribution dist = new Beta(0.5, 0.5);
        assertTrue(Double.isNaN(dist.mode()));
        dist = new Beta(0.5, 3);
        assertTrue(Double.isNaN(dist.mode()));
        dist = new Beta(3, 0.5);
        assertTrue(Double.isNaN(dist.mode()));
        dist = new Beta(3, 3);
        assertEquals(0.5, dist.mode(), 1e-10);
    }

    /**
     * Test of variance method, of class Beta.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        ContinuousDistribution dist = new Beta(0.5, 0.5);
        assertEquals(0.125, dist.variance(), 1e-10);
        dist = new Beta(0.5, 3);
        assertEquals(0.0272108843537415, dist.variance(), 1e-10);
        dist = new Beta(3, 0.5);
        assertEquals(0.0272108843537415, dist.variance(), 1e-10);
        dist = new Beta(3, 3);
        assertEquals(0.03571428571428571, dist.variance(), 1e-10);
    }

    /**
     * Test of skewness method, of class Beta.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        ContinuousDistribution dist = new Beta(0.5, 0.5);
        assertEquals(0, dist.skewness(), 1e-10);
        dist = new Beta(0.5, 3);
        assertEquals(1.5745916432444336, dist.skewness(), 1e-10);
        dist = new Beta(3, 0.5);
        assertEquals(-1.5745916432444336, dist.skewness(), 1e-10);
        dist = new Beta(3, 3);
        assertEquals(0, dist.skewness(), 1e-10);
    }
    
    @Test
    public void testEquals(){
    	System.out.println("equals");
    	ContinuousDistribution d1 = new Beta(0.5, 0.5);
    	ContinuousDistribution d2 = new Beta(0.6, 0.5);
    	ContinuousDistribution d3 = new Beta(0.5, 0.6);
    	ContinuousDistribution d4 = new Beta(0.5, 0.5);
    	Integer i = new Integer(1);
    	assertFalse(d1.equals(d2));
    	assertFalse(d1.equals(d3));
    	assertFalse(d2.equals(d3));
    	assertFalse(d1.equals(i));
    	assertFalse(d1.equals(null));
    	assertEquals(d1, d1);
    	assertEquals(d1, d4);
    	assertEquals(d1, d1.clone());
    }
    
    @Test
    public void testHashCode(){
    	System.out.println("hashCode");
    	ContinuousDistribution d1 = new Beta(0.5, 0.5);
    	ContinuousDistribution d2 = new Beta(0.6, 0.5);
    	ContinuousDistribution d4 = new Beta(0.5, 0.5);
    	assertEquals(d1.hashCode(), d4.hashCode());
    	assertFalse(d1.hashCode()==d2.hashCode());
    }
    
}
